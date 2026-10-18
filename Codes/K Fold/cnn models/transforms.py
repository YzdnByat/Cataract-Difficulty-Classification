"""Build notebook-exact stochastic training and deterministic validation transforms."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def _mean_rgb(image: Image.Image) -> Tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    return tuple(arr.reshape(-1, 3).mean(axis=0).astype(np.uint8).tolist())


class PadToSquareWithMean:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return TF.pad(image, (left, top, right, bottom), fill=_mean_rgb(image))


class RotateWithMeanFill:
    def __init__(self, degrees: float):
        self.degrees = float(degrees)

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = transforms.RandomRotation.get_params([-self.degrees, self.degrees])
        return TF.rotate(image, angle, fill=_mean_rgb(image))


class AffineWithMeanFill:
    def __init__(self, translate, scale, shear):
        self.translate = tuple(translate)
        self.scale = tuple(scale)
        self.shear = tuple(shear)

    def __call__(self, image: Image.Image) -> Image.Image:
        angle, translation, scale, shear = transforms.RandomAffine.get_params(
            degrees=(0.0, 0.0),
            translate=self.translate,
            scale_ranges=self.scale,
            shears=self.shear,
            img_size=image.size,
        )
        return TF.affine(
            image,
            angle=angle,
            translate=translation,
            scale=scale,
            shear=shear,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=_mean_rgb(image),
        )


def build_transforms(model_config: Dict[str, Any]):
    augmentation = model_config["augmentation"]
    size = int(model_config["input_resolution"])
    mean = augmentation["normalize_mean"]
    std = augmentation["normalize_std"]
    color = augmentation["color_jitter"]
    blur = augmentation["gaussian_blur"]

    evaluation = transforms.Compose(
        [
            PadToSquareWithMean(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    training = transforms.Compose(
        [
            PadToSquareWithMean(),
            transforms.Resize((size, size)),
            RotateWithMeanFill(augmentation["rotation_degrees"]),
            transforms.RandomHorizontalFlip(
                p=float(augmentation["horizontal_flip_probability"])
            ),
            AffineWithMeanFill(
                translate=augmentation["affine_translate"],
                scale=augmentation["affine_scale"],
                shear=augmentation["affine_shear"],
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=float(color["brightness"]),
                        contrast=float(color["contrast"]),
                        saturation=float(color["saturation"]),
                    )
                ],
                p=float(color["probability"]),
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=int(blur["kernel_size"]),
                        sigma=tuple(blur["sigma"]),
                    )
                ],
                p=float(blur["probability"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return training, evaluation

