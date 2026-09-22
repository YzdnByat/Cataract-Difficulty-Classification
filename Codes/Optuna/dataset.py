"""
dataset.py
----------
Dataset abstractions, video-wise splitting with zero-leakage validation,
quota-based class sampling, and PyTorch DataLoader factories.
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import GroupShuffleSplit

from config import PathConfig, TaskConfig, ModelConfig
from transforms import get_transforms
from metadata_utils import load_metadata, resolve_image


# =============================================================
# CLASS SAMPLER
# =============================================================

class BalancedClassSampler(Sampler):
    """
    Ensures a fixed target number of samples per class per epoch.

    Sampling uses replacement only when the requested target for a
    class is larger than the number of available training frames.
    """

    def __init__(
        self,
        labels: np.ndarray,
        target_per_class: Dict[int, int]
    ):
        self.labels = np.asarray(labels)
        self.target_per_class = target_per_class
        self.classes = np.unique(self.labels)

        # Verify all expected classes exist in training split
        expected_classes = set(self.target_per_class.keys())
        present_classes = set(self.classes.tolist())

        missing_classes = expected_classes - present_classes

        if missing_classes:
            raise ValueError(
                f"Training split is missing required classes: "
                f"{sorted(missing_classes)}"
            )

        # Store indices belonging to each class
        self.class_indices = {
            int(c): np.where(self.labels == c)[0]
            for c in self.classes
        }

    def __iter__(self):
        indices = []

        for c in self.classes:
            c = int(c)

            target_n = self.target_per_class[c]
            class_indices = self.class_indices[c]

            # Only use replacement when oversampling is required
            replace = target_n > len(class_indices)

            chosen = np.random.choice(
                class_indices,
                size=target_n,
                replace=replace
            )

            indices.extend(chosen.tolist())

        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        return sum(self.target_per_class.values())


# =============================================================
# PYTORCH DATASET
# =============================================================

class CataractDataset(Dataset):
    """
    PyTorch dataset reading images using:

        video_id
        file_name
        label

    Labels are already numeric in the metadata CSV.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        transform=None
    ):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [
            resolve_image(image_dir, row.video_id, row.file_name)
            for row in self.data.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        row = self.data.iloc[idx]

        with Image.open(self.image_paths[idx]) as source:
            image = source.convert("RGB")

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        if self.transform:
            image = self.transform(image)

        return image, label


# =============================================================
# DATA PREPARATION AND SPLITTING
# =============================================================

class PatientDataSplitter:
    """
    Performs video-isolated 70/15/15 train/val/test splits.

    Frames from the same video_id can never appear in more than one
    split.
    """

    @staticmethod
    def load_and_prep_df(
        task: str,
        csv_path: str,
        img_dir: str
    ) -> pd.DataFrame:

        df = load_metadata(task, csv_path)
        # Fail before starting Optuna if any metadata image is missing/ambiguous.
        for row in df.itertuples(index=False):
            resolve_image(img_dir, row.video_id, row.file_name)
        print(f"Validated {task}: {len(df)} frames, {df.video_id.nunique()} videos")
        return df

    @classmethod
    def split(
        cls,
        df: pd.DataFrame,
        random_state: int = 42
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame
    ]:

        # ---------------------------------------------------------
        # Stage 1:
        # 70% Train
        # 30% Temporary (Validation + Test)
        # ---------------------------------------------------------

        gss_train = GroupShuffleSplit(
            n_splits=1,
            test_size=0.30,
            random_state=random_state
        )

        train_idx, temp_idx = next(
            gss_train.split(
                df,
                groups=df["video_id"]
            )
        )

        train_df = (
            df.iloc[train_idx]
            .reset_index(drop=True)
        )

        temp_df = (
            df.iloc[temp_idx]
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------
        # Stage 2:
        # Split temporary data 50/50
        #
        # 15% Validation
        # 15% Test
        # ---------------------------------------------------------

        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=random_state
        )

        val_idx, test_idx = next(
            gss_val.split(
                temp_df,
                groups=temp_df["video_id"]
            )
        )

        val_df = (
            temp_df.iloc[val_idx]
            .reset_index(drop=True)
        )

        test_df = (
            temp_df.iloc[test_idx]
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------
        # Strict video-leakage checks
        # ---------------------------------------------------------

        train_vids = set(
            train_df["video_id"]
        )

        val_vids = set(
            val_df["video_id"]
        )

        test_vids = set(
            test_df["video_id"]
        )

        assert len(train_vids & val_vids) == 0, (
            "Video leakage between Train and Val sets!"
        )

        assert len(train_vids & test_vids) == 0, (
            "Video leakage between Train and Test sets!"
        )

        assert len(val_vids & test_vids) == 0, (
            "Video leakage between Val and Test sets!"
        )

        return train_df, val_df, test_df


# =============================================================
# DATALOADER FACTORY
# =============================================================

def build_dataloaders(
    task: str,
    img_size: int,
    batch_size: int = ModelConfig.BATCH_SIZE,
    num_workers: int = ModelConfig.NUM_WORKERS,
    csv_path: Optional[str] = None,
    img_dir: Optional[str] = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Dict[str, pd.DataFrame]
]:

    """
    Prepares video-isolated datasets and returns
    train/validation/test DataLoaders.
    """

    # ---------------------------------------------------------
    # Select dataset
    # ---------------------------------------------------------

    if task == "severity":

        csv_file = (
            csv_path
            or PathConfig.SEVERITY_CSV_PATH
        )

        img_folder = (
            img_dir
            or PathConfig.SEVERITY_IMG_DIR
        )

        target_per_class = (
            TaskConfig.SEVERITY_TARGET_PER_CLASS
        )

    elif task == "poor_dilation":

        csv_file = (
            csv_path
            or PathConfig.DILATION_CSV_PATH
        )

        img_folder = (
            img_dir
            or PathConfig.DILATION_IMG_DIR
        )

        target_per_class = (
            TaskConfig.DILATION_TARGET_PER_CLASS
        )

    else:
        raise ValueError(
            f"Unknown task: {task}"
        )

    # ---------------------------------------------------------
    # Load metadata and split
    # ---------------------------------------------------------

    df = PatientDataSplitter.load_and_prep_df(
        task,
        csv_file,
        img_folder
    )

    train_df, val_df, test_df = (
        PatientDataSplitter.split(
            df,
            random_state=ModelConfig.RANDOM_STATE
        )
    )

    # ---------------------------------------------------------
    # Transforms
    # ---------------------------------------------------------

    train_transform, eval_transform = (
        get_transforms(
            task=task,
            img_size=img_size
        )
    )

    # ---------------------------------------------------------
    # Dataset objects
    # ---------------------------------------------------------

    train_dataset = CataractDataset(
        train_df,
        img_folder,
        transform=train_transform
    )

    val_dataset = CataractDataset(
        val_df,
        img_folder,
        transform=eval_transform
    )

    test_dataset = CataractDataset(
        test_df,
        img_folder,
        transform=eval_transform
    )

    # ---------------------------------------------------------
    # Training sampler
    # ---------------------------------------------------------

    y_train = (
        train_df["label"]
        .astype(int)
        .to_numpy()
    )

    train_sampler = BalancedClassSampler(
        labels=y_train,
        target_per_class=target_per_class
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    return (
        train_loader,
        val_loader,
        test_loader,
        split_dfs
    )