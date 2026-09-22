"""
transforms.py
-------------
Data augmentation utilities and torchvision transform pipelines
with mean-color background padding to preserve surgical image aspect ratios.
"""

from typing import Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def _mean_rgb(image: Image.Image) -> Tuple[int, int, int]:
    """Computes the mean RGB value of a PIL image."""
    arr = np.array(image.convert("RGB"))
    return tuple(arr.reshape(-1, 3).mean(axis=0).astype(np.uint8).tolist())


class PadToSquareWithMean:
    """Pads a rectangular image into a square using its mean RGB value."""

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        mean_color = _mean_rgb(image)

        return TF.pad(
            image,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=mean_color,
            padding_mode="constant",
        )


class RotateWithMeanFill:
    """Rotates an image randomly within [-degrees, degrees] using mean-fill background."""

    def __init__(self, degrees: int = 12):
        self.degrees = degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = transforms.RandomRotation.get_params([-self.degrees, self.degrees])
        mean_color = _mean_rgb(image)
        return TF.rotate(image, angle=angle, expand=False, fill=mean_color)


class AffineWithMeanFill:
    """Applies random affine transformations with mean-fill background padding."""

    def __init__(
        self,
        degrees: int = 0,
        translate: Tuple[float, float] = (0.10, 0.10),
        scale: Tuple[float, float] = (0.95, 1.05),
        shear: Tuple[int, int] = (-4, 4),
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, image: Image.Image) -> Image.Image:
        angle, translations, scale_factor, shear_factor = (
            transforms.RandomAffine.get_params(
                degrees=(-self.degrees, self.degrees),
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=image.size,
            )
        )
        mean_color = _mean_rgb(image)
        return TF.affine(
            image,
            angle=angle,
            translate=translations,
            scale=scale_factor,
            shear=shear_factor,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=mean_color,
        )


def get_transforms(
    task: str,
    img_size: int = 224
) -> Tuple[transforms.Compose, transforms.Compose]:

    """
    Generates train and evaluation transform pipelines.

    Args:
        task: 'severity' or 'poor_dilation'
        img_size: Target square image size

    Returns:
        train_transform, eval_transform
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # ============================================================
    # Validation / Test
    # ============================================================

    eval_transform = transforms.Compose(
        [
            PadToSquareWithMean(),

            transforms.Resize(
                (img_size, img_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std,
            ),
        ]
    )

    # ============================================================
    # Task-specific photometric augmentation
    # ============================================================

    if task == "severity":

        # More conservative augmentation because brightness,
        # contrast and sharpness can contain useful information
        # about cataract severity.

        brightness = 0.15
        contrast = 0.15
        saturation = 0.05

        color_jitter_probability = 0.40

        blur_kernel = 5
        blur_sigma = (0.1, 0.8)
        blur_probability = 0.15

    elif task == "poor_dilation":

        # Poor dilation depends more strongly on pupil geometry,
        # so somewhat stronger photometric augmentation is acceptable.

        brightness = 0.25
        contrast = 0.25
        saturation = 0.10

        color_jitter_probability = 0.60

        blur_kernel = 3
        blur_sigma = (0.1, 0.4)
        blur_probability = 0.30

    else:
        raise ValueError(
            f"Unsupported task: {task}"
        )

    # ============================================================
    # Training augmentation
    # ============================================================

    train_transform = transforms.Compose(
        [
            # Preserve original aspect ratio
            PadToSquareWithMean(),

            # Resize to model input resolution
            transforms.Resize(
                (img_size, img_size)
            ),

            # Small rotation
            RotateWithMeanFill(
                degrees=12
            ),

            # Horizontal reflection
            transforms.RandomHorizontalFlip(
                p=0.30
            ),

            # Small geometric perturbations
            AffineWithMeanFill(
                degrees=0,
                translate=(0.10, 0.10),
                scale=(0.95, 1.05),
                shear=(-4, 4),
            ),

            # ----------------------------------------------------
            # Photometric augmentation
            # ----------------------------------------------------

            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                    )
                ],
                p=color_jitter_probability,
            ),

            # ----------------------------------------------------
            # Mild blur augmentation
            # ----------------------------------------------------

            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=blur_kernel,
                        sigma=blur_sigma,
                    )
                ],
                p=blur_probability,
            ),

            # ----------------------------------------------------
            # Tensor + ImageNet normalization
            # ----------------------------------------------------

            transforms.ToTensor(),

            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std,
            ),
        ]
    )

    return train_transform, eval_transform