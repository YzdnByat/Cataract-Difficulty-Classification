"""Shared deterministic geometry for training, inference and exports.

Pillow/NumPy only: preparation and reports do not require PyTorch. Padding is
generated at the final resolution; the resized image itself is never blurred.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageFilter, ImageStat
from PIL import __version__ as PILLOW_VERSION
from . import color_constancy

MODEL_NAME = 'swin_small_patch4_window7_224.ms_in22k_ft_in1k'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALGORITHM_VERSION = 'aspect_fit_edge_mean_blend_v1'
DEFAULT_PREPROCESSING = {
    'input_size': 448,
    'interpolation': 'bicubic',
    'normalization': 'imagenet',
    'padding': {'method': 'edge_blend', 'transition_fraction': 0.10, 'blur_radius': 2.0},
}


# Shared with the pipeline builder and recorded in preprocessing.json.
TRAIN_AUGMENTATION = {
    'RandomResizedCrop': {'scale': [0.80, 1.0], 'ratio': [0.9, 1.11], 'p': 1.0},
    'ShiftScaleRotate': {
        'shift_limit': 0.05, 'scale_limit': 0, 'rotate_limit': 15, 'fill': 0, 'p': 0.7,
    },
    'HorizontalFlip': {'p': 0.5},
    'VerticalFlip': {'p': 0.3},
    'ColorJitter': {
        'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.05, 'p': 0.8,
    },
}


def validate_preprocessing(value=None):
    cfg = deepcopy(DEFAULT_PREPROCESSING if value is None else value)
    if not isinstance(cfg, dict) or set(cfg) - {'color_constancy'} != set(DEFAULT_PREPROCESSING):
        raise ValueError('preprocessing requires input_size, interpolation, normalization and padding.')
    if cfg.get('color_constancy', 'none') not in color_constancy.COLOR_METHODS:
        raise ValueError('color_constancy must be none, gray_world or white_patch.')
    if type(cfg['input_size']) is not int or cfg['input_size'] not in {224, 448}:
        raise ValueError('input_size must be 224 or 448.')
    if cfg['interpolation'] != 'bicubic' or cfg['normalization'] != 'imagenet':
        raise ValueError('Use bicubic interpolation and imagenet normalization for the pinned pretrained checkpoint.')
    pad = cfg['padding']
    if not isinstance(pad, dict) or set(pad) != {'method', 'transition_fraction', 'blur_radius'}:
        raise ValueError('padding requires method, transition_fraction and blur_radius.')
    if pad['method'] not in {'edge_blend', 'mean'}:
        raise ValueError('padding.method must be edge_blend or mean.')
    for key in ['transition_fraction', 'blur_radius']:
        v = pad[key]
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
            raise ValueError(f'padding.{key} must be a finite number.')
        pad[key] = float(v)
    if not 0 < pad['transition_fraction'] <= 0.5 or not 0 <= pad['blur_radius'] <= 16:
        raise ValueError('Use 0 < transition_fraction <= 0.5 and 0 <= blur_radius <= 16.')
    return cfg


def geometry(width, height, size=448):
    if any(type(v) is not int or v <= 0 for v in (width, height, size)):
        raise ValueError('Image dimensions and target size must be positive integers.')
    if width >= height:
        w, h = size, max(1, int(round(height * size / width)))
    else:
        w, h = max(1, int(round(width * size / height))), size
    left, top = (size-w)//2, (size-h)//2
    return {'original_width': width, 'original_height': height,
            'resized_width': w, 'resized_height': h,
            'pad_left': left, 'pad_right': size-w-left, 'pad_top': top, 'pad_bottom': size-h-top,
            'padding_fraction': float(1-(w*h)/(size*size))}


def mean_rgb(image):
    """Mean over all image pixels, in the original 0..255 RGB scale."""
    return tuple(int(round(v)) for v in ImageStat.Stat(image).mean[:3])


def _fit_pad(image, cfg):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    size = cfg['input_size']
    g = geometry(*image.size, size)
    w, h = g['resized_width'], g['resized_height']
    resized = image.resize((w, h), Image.Resampling.BICUBIC)
    if w == h == size:
        return resized
    colour = mean_rgb(image)
    left, right, top, bottom = (g[k] for k in ['pad_left', 'pad_right', 'pad_top', 'pad_bottom'])
    if cfg['padding']['method'] == 'mean':
        canvas = Image.new('RGB', (size, size), colour)
        canvas.paste(resized, (left, top))
        return canvas
    pixels = np.asarray(resized)
    edge = np.pad(pixels, ((top, bottom), (left, right), (0, 0)), mode='edge')
    blurred = np.asarray(Image.fromarray(edge).filter(
        ImageFilter.GaussianBlur(cfg['padding']['blur_radius'])), dtype=np.float32)
    # First padding pixel equals the adjacent edge. Transition reaches the global
    # image mean within ~10% of the canvas, compressed for narrower padding bands.
    ramp = max(1, int(round(size * cfg['padding']['transition_fraction'])))
    xs, ys = np.arange(size)[None, :], np.arange(size)[:, None]
    def weight(distance, band):
        return np.clip((distance-1)/max(1, min(band-1, ramp)), 0, 1)
    t = np.maximum(np.maximum(weight(left-xs, left), weight(xs-(left+w-1), right)),
                   np.maximum(weight(top-ys, top), weight(ys-(top+h-1), bottom)))
    smooth = (t*t*(3-2*t)).astype(np.float32)[..., None]
    canvas = (1-smooth)**2 * edge + smooth*(1-smooth)*blurred + smooth*np.asarray(colour, dtype=np.float32)
    canvas = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
    # Preserve every resized frame pixel; blur/blending applies only to pads.
    canvas[top:top+h, left:left+w] = pixels
    return Image.fromarray(canvas)


def fit_pad(image, config=None, *, source=None):
    return FitPad(config)(image, source=source)


class FitPad:
    """Picklable torchvision-compatible callable, with validation done once."""
    def __init__(self, config=None):
        self.config = validate_preprocessing(config)
        self.color_method = self.config.get('color_constancy', 'none')

    def __call__(self, image, *, source=None):
        if self.color_method != 'none':
            if not isinstance(source, str) or not source:
                raise ValueError('A frame source is required when color constancy is enabled.')
            if source == 'Noor':
                image = color_constancy.correct_rgb(image, self.color_method)
        return _fit_pad(image, self.config)


def preprocessing_metadata(config=None):
    cfg = validate_preprocessing(config)
    identity = {'algorithm_version': ALGORITHM_VERSION,
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'configuration': cfg, 'runtime_versions': {'Pillow': PILLOW_VERSION, 'numpy': np.__version__}}
    method = cfg.get('color_constancy', 'none')
    if method != 'none':
        identity['color_constancy'] = {
            'method': method, 'source': 'Noor', 'stage': 'original RGB before resize and padding',
            'implementation_sha256': hashlib.sha256(Path(color_constancy.__file__).read_bytes()).hexdigest(),
        }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {**identity, 'fingerprint': signature, 'pretrained_checkpoint': MODEL_NAME,
            'tensor_normalization': {'mean': list(IMAGENET_MEAN), 'std': list(IMAGENET_STD),
                                     'formula': '(RGB / 255 - mean) / std'},
            'training_augmentation': {
                'backend': 'albumentations',
                'input': 'FitPad RGB square',
                'order': [*TRAIN_AUGMENTATION, 'Normalize', 'ToTensorV2'],
                'parameters': deepcopy(TRAIN_AUGMENTATION),
                'crop_output_size': [cfg['input_size'], cfg['input_size']],
                'geometric_interpolation': 'OpenCV INTER_LINEAR',
                'affine_border_mode': 'OpenCV BORDER_CONSTANT',
                'newly_exposed_pixel_fill': 'black RGB during ShiftScaleRotate, before ColorJitter',
            },
            'evaluation_augmentation': False,
            'geometry': 'Full frame resized proportionally and centered on a square before training augmentation.',
            'limits': 'Training crop, rotation and translation can exclude parts of the original frame.'}



def validate_pretrained_contract(pretrained_cfg, config=None):
    cfg = validate_preprocessing(config)
    # Native checkpoint metadata remains 224 even when fine-tuning at 448.
    expected = {'input_size': [3, 224, 224],
                'mean': list(IMAGENET_MEAN), 'std': list(IMAGENET_STD), 'interpolation': cfg['interpolation']}
    for key, value in expected.items():
        actual = pretrained_cfg.get(key)
        if key in {'input_size', 'mean', 'std'} and actual is not None:
            actual = list(actual)
        if actual != value:
            raise ValueError(f'Pretrained checkpoint {key}={actual!r} differs from preprocessing {value!r}.')
