"""Padded evaluation inputs and the shared Albumentations training pipeline."""
import os
import random

# The augmentation version is pinned. Avoid a network check at import time.
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from .preprocessing import FitPad, IMAGENET_MEAN, IMAGENET_STD, TRAIN_AUGMENTATION


def build_training_pipeline(input_size, *, replay=False):
    """Build identical operators for training and recorded inspection draws."""
    settings = TRAIN_AUGMENTATION
    ops = [
        A.RandomResizedCrop(
            size=(input_size, input_size),
            scale=tuple(settings['RandomResizedCrop']['scale']),
            ratio=tuple(settings['RandomResizedCrop']['ratio']),
            interpolation=cv2.INTER_LINEAR,
            p=settings['RandomResizedCrop']['p'],
        ),
        A.ShiftScaleRotate(
            **settings['ShiftScaleRotate'],
            border_mode=cv2.BORDER_CONSTANT,
            interpolation=cv2.INTER_LINEAR,
        ),
        A.HorizontalFlip(**settings['HorizontalFlip']),
        A.VerticalFlip(**settings['VerticalFlip']),
        A.ColorJitter(**settings['ColorJitter']),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ]
    return A.ReplayCompose(ops) if replay else A.Compose(ops, strict=True)


def apply_training_pipeline(pipeline, padded_image, *, seed):
    """Run a draw on the square RGB canvas with a shared random stream."""
    # Giving every operator the same seed would correlate their flip/apply flags.
    pipeline.set_random_state(np.random.default_rng(seed), random.Random(seed))
    return pipeline(image=np.asarray(padded_image))


class TrainingTransform:
    def __init__(self, preprocessing=None, *, generator=None):
        self.pad = FitPad(preprocessing)
        self.pipeline = build_training_pipeline(self.pad.config['input_size'])
        self.generator = generator

    def __call__(self, image, *, source=None):
        # With workers, DataLoader seeds each process. With zero workers use the
        # data generator directly to keep augmentation separate from the model.
        seed = torch.randint(0, 2**32, (), dtype=torch.int64, generator=self.generator).item()
        return apply_training_pipeline(self.pipeline, self.pad(image, source=source), seed=seed)['image']


class EvaluationTransform:
    def __init__(self, preprocessing=None):
        self.pad = FitPad(preprocessing)
        self.to_normalized_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, image, *, source=None):
        return self.to_normalized_tensor(self.pad(image, source=source))


class FrameDataset(Dataset):
    def __init__(self, df, augment, preprocessing=None, *, generator=None):
        self.df = df.reset_index(drop=True)
        self.paths = tuple(self.df.image_path)
        self.labels = tuple(int(value) for value in self.df.label)
        self.sources = tuple(self.df.source) if 'source' in self.df else (None,) * len(self.df)
        self.transform = (TrainingTransform(preprocessing, generator=generator) if augment
                          else EvaluationTransform(preprocessing))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image, source=self.sources[index])
        return tensor, self.labels[index]


def make_loaders(df, cfg, preprocessing=None, *, data_seed=None):
    loaders = {}
    for split in ['train', 'val', 'test']:
        generator = None
        if split == 'train' and data_seed is not None:
            generator = torch.Generator().manual_seed(data_seed)
        dataset = FrameDataset(df[df.split == split], split == 'train', preprocessing,
                               generator=generator if cfg['workers'] == 0 else None)
        sampler = None
        if split == 'train':
            counts = dataset.df.label.value_counts()
            weights = dataset.df.label.map(lambda value: 1 / counts[value]).to_numpy()
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double), len(dataset), replacement=True, generator=generator)
        loaders[split] = DataLoader(
            dataset, batch_size=cfg['batch_size'], sampler=sampler, shuffle=False,
            num_workers=cfg['workers'], pin_memory=True, drop_last=False, generator=generator)
    return loaders
