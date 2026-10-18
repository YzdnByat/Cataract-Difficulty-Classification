"""Deterministic, video-isolated stratified K-fold generation and auditing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


LABELS: Dict[str, Dict[str, int]] = {
    "severity": {"low": 0, "dense": 1, "mature": 2, "brunescent": 3},
    "poor_dilation": {
        "normal": 0,
        "poor dilation": 1,
        "poor_dilation": 1,
        "poordilation": 1,
    },
}


def load_metadata(csv_path: str | Path, task: str) -> pd.DataFrame:
    """Load required columns and normalize labels without changing row order."""
    task = task.strip().lower()
    if task not in LABELS:
        raise ValueError(f"Unsupported task: {task}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    aliases = {"videoname": "video_id", "filename": "file_name"}
    for old, new in aliases.items():
        if new not in df.columns and old in df.columns:
            df = df.rename(columns={old: new})

    required = {"file_name", "video_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required metadata columns: {sorted(missing)}")

    if not pd.api.types.is_numeric_dtype(df["label"]):
        normalized = df["label"].astype(str).str.strip().str.lower()
        mapped = normalized.map(LABELS[task])
        if mapped.isna().any():
            bad = sorted(normalized[mapped.isna()].unique().tolist())
            raise ValueError(f"Unknown {task} labels: {bad}")
        df["label"] = mapped.astype(int)
    else:
        df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    expected = set(LABELS[task].values())
    actual = set(df["label"].unique().tolist())
    if actual != expected:
        raise ValueError(
            f"Expected labels {sorted(expected)} for {task}; found {sorted(actual)}"
        )
    if df[["file_name", "video_id", "label"]].isna().any().any():
        raise ValueError("Required metadata columns contain missing values")
    if df.duplicated(["video_id", "file_name"]).any():
        raise ValueError("Duplicate (video_id, file_name) rows were found")

    result = df[["file_name", "video_id", "label"]].copy()
    result.insert(0, "row_id", np.arange(len(result), dtype=int))
    return result


def _groups_containing_each_class(df: pd.DataFrame) -> Dict[int, int]:
    pairs = df[["video_id", "label"]].drop_duplicates()
    return {int(k): int(v) for k, v in pairs.groupby("label").size().items()}


def make_grouped_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign one validation fold per video and verify class coverage."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    unique_groups = df["video_id"].nunique()
    if unique_groups < n_splits:
        raise ValueError(
            f"Only {unique_groups} unique videos are available for {n_splits} folds"
        )

    class_group_counts = _groups_containing_each_class(df)
    too_few = {k: v for k, v in class_group_counts.items() if v < n_splits}
    if too_few:
        raise ValueError(
            "Each class must occur in at least n_splits distinct videos. "
            f"Insufficient class/video counts: {too_few}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folded = df.copy()
    folded["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        splitter.split(folded, y=folded["label"], groups=folded["video_id"])
    ):
        folded.loc[folded.index[val_idx], "fold"] = fold

    validate_fold_assignments(folded, n_splits)
    return folded


def validate_fold_assignments(df: pd.DataFrame, n_splits: int) -> None:
    expected_folds = set(range(n_splits))
    actual_folds = set(df["fold"].unique().tolist())
    if actual_folds != expected_folds:
        raise AssertionError(
            f"Expected folds {sorted(expected_folds)}; found {sorted(actual_folds)}"
        )

    folds_per_video = df.groupby("video_id")["fold"].nunique()
    if not (folds_per_video == 1).all():
        raise AssertionError("At least one video_id appears in multiple folds")

    expected_labels = set(df["label"].unique().tolist())
    for fold in range(n_splits):
        validation = df[df["fold"] == fold]
        training = df[df["fold"] != fold]
        overlap = set(validation["video_id"]) & set(training["video_id"])
        if overlap:
            raise AssertionError(f"Video leakage in fold {fold}: {sorted(overlap)[:5]}")
        val_labels = set(validation["label"].unique().tolist())
        train_labels = set(training["label"].unique().tolist())
        if val_labels != expected_labels or train_labels != expected_labels:
            raise ValueError(
                f"Fold {fold} lacks class coverage. "
                f"train={sorted(train_labels)}, val={sorted(val_labels)}"
            )


def build_fold_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Return fold/class frame and unique-video counts in tidy form."""
    records = []
    for fold in sorted(df["fold"].unique()):
        part = df[df["fold"] == fold]
        for label in sorted(df["label"].unique()):
            class_part = part[part["label"] == label]
            records.append(
                {
                    "fold": int(fold),
                    "label": int(label),
                    "frames": int(len(class_part)),
                    "videos_containing_label": int(class_part["video_id"].nunique()),
                    "fold_frames": int(len(part)),
                    "fold_videos": int(part["video_id"].nunique()),
                }
            )
    return pd.DataFrame.from_records(records)
