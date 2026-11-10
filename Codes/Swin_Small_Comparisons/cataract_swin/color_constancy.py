"""Per-frame channel gains on decoded RGB pixels."""
import numpy as np
from PIL import Image

COLOR_METHODS = ('none', 'gray_world', 'white_patch')


def correct_rgb(image, method):
    """Apply Gray World or White Patch to the original, unpadded frame."""
    if method == 'none':
        return image.convert('RGB')
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    if method == 'gray_world':
        estimate = pixels.mean(axis=(0, 1), dtype=np.float64)
        target = estimate.mean()
    elif method == 'white_patch':
        estimate = pixels.max(axis=(0, 1)).astype(np.float64)
        target = 255.0
    else:
        raise ValueError(f'Unknown color constancy method {method!r}.')
    gains = np.ones(3, dtype=np.float64)
    np.divide(target, estimate, out=gains, where=estimate > 0)
    corrected = np.rint(pixels * gains.astype(np.float32))
    return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))
