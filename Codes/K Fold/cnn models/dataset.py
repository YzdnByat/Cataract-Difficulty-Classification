"""Datasets, notebook-exact class sampling, and fold DataLoaders."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from transforms import build_transforms


class BalancedClassSampler(Sampler[int]):
    """Draw each configured class quota with replacement, matching the notebooks."""

    def __init__(self, labels: np.ndarray, target_per_class: Dict[int, int], seed: int = 42):
        self.labels = np.asarray(labels, dtype=int)
        self.target_per_class = {int(k): int(v) for k, v in target_per_class.items()}
        self.seed = int(seed)
        self.epoch = 0
        present = set(np.unique(self.labels).tolist())
        expected = set(self.target_per_class)
        if expected != present:
            raise ValueError(
                f"Training-fold classes do not match sampler quotas: "
                f"present={sorted(present)}, quotas={sorted(expected)}"
            )
        self.class_indices = {
            label: np.flatnonzero(self.labels == label) for label in sorted(expected)
        }
        if any(value <= 0 for value in self.target_per_class.values()):
            raise ValueError("Every class quota must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        selected = []
        for label, indices in self.class_indices.items():
            target = self.target_per_class[label]
            # The source notebooks explicitly use replace=True for every class.
            selected.extend(rng.choice(indices, size=target, replace=True).tolist())
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return sum(self.target_per_class.values())


def _candidate_image_paths(image_dir: Path, video_id: str, file_name: str) -> list[Path]:
    """Support both current flat archives and older notebook video subfolders."""
    candidates = [
        image_dir / file_name,
        image_dir / video_id / file_name,
        image_dir / "images" / file_name,
        image_dir / "images" / video_id / file_name,
    ]
    unique = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


class CataractDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_dir: str | Path, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = []
        missing = []
        for row in self.data.itertuples(index=False):
            candidates = _candidate_image_paths(
                self.image_dir, str(row.video_id), str(row.file_name)
            )
            found = next((path for path in candidates if path.is_file()), None)
            if found is None:
                missing.append((str(row.video_id), str(row.file_name), candidates))
            else:
                self.image_paths.append(found)
        if missing:
            example = missing[0]
            attempted = ", ".join(str(path) for path in example[2])
            raise FileNotFoundError(
                f"Could not resolve {len(missing)} image(s). First missing record: "
                f"video_id={example[0]}, file_name={example[1]}. Tried: {attempted}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.image_paths[idx]) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(self.data.iloc[idx]["label"]), dtype=torch.long)
        return image, label


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_fold_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str | Path,
    model_config: Dict[str, Any],
    batch_size: int | None = None,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_transform, eval_transform = build_transforms(model_config)
    train_dataset = CataractDataset(train_df, image_dir, train_transform)
    val_dataset = CataractDataset(val_df, image_dir, eval_transform)
    quotas = {
        int(label): int(quota)
        for label, quota in model_config["sampler"]["target_per_class"].items()
    }
    sampler = BalancedClassSampler(train_df["label"].to_numpy(), quotas, seed=seed)
    actual_batch_size = int(batch_size or model_config["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    return (
        DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            sampler=sampler,
            shuffle=False,
            **common,
        ),
        DataLoader(
            val_dataset,
            batch_size=actual_batch_size,
            shuffle=False,
            **common,
        ),
    )
