"""Original-name metadata conversion and explicit image resolution."""
from pathlib import Path
import warnings
import pandas as pd


CLASS_COLUMNS = {
    "severity": ["low nuclear density", "dense", "mature", "brunescent"],
    "poor_dilation": ["normal", "poor dilation"],
}


def load_metadata(task, path):
    if task not in CLASS_COLUMNS:
        raise ValueError(f"Unsupported task: {task}")
    # Read identifiers as text so leading zeros are preserved.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError("Duplicate metadata column names")
    for old, new in [("filename", "file_name"), ("videoname", "video_id")]:
        if old in df and new in df:
            raise ValueError(f"Both {old} and {new} present; use one identifier column")
        df = df.rename(columns={old: new})
    for column in ["file_name", "video_id"]:
        if column not in df:
            raise ValueError(f"Missing metadata column: {column}")
        df[column] = df[column].str.strip()
        if df[column].eq("").any():
            raise ValueError(f"Empty values in {column}")
    columns = CLASS_COLUMNS[task]
    expected = set(range(len(columns)))
    present = [c for c in columns if c in df]
    onehot_labels = None
    if present:
        if len(present) != len(columns):
            raise ValueError(f"One-hot metadata requires all columns: {columns}")
        values = df[columns].apply(pd.to_numeric, errors="raise")
        if not values.isin([0, 1]).all().all() or not values.sum(axis=1).eq(1).all():
            raise ValueError("Each row must have exactly one class set to 1, all others 0")
        onehot_labels = values.idxmax(axis=1).map(dict(zip(columns, range(len(columns)))))
    if "label" in df:
        labels = pd.to_numeric(df["label"], errors="raise")
        if not labels.isin(expected).all():
            raise ValueError(f"Labels must be integers in {sorted(expected)}")
        if onehot_labels is not None and not labels.eq(onehot_labels).all():
            raise ValueError("Numeric labels disagree with one-hot labels")
    elif onehot_labels is not None:
        labels = onehot_labels
    else:
        raise ValueError(f"Provide label or one-hot columns {columns}")
    df["label"] = labels.astype(int)
    df = df[["file_name", "video_id", "label"]].copy()
    if df.groupby(["video_id", "file_name"])["label"].nunique().gt(1).any():
        raise ValueError("Conflicting labels for the same video/image record")
    duplicates = df.duplicated(["video_id", "file_name"]).sum()
    if duplicates:
        warnings.warn(f"Removed {duplicates} repeated video/image record(s) with identical labels")
        df = df.drop_duplicates(["video_id", "file_name"])
    if set(df["label"]) != expected:
        raise ValueError(f"Dataset must contain all labels {sorted(expected)}")
    return df.reset_index(drop=True)


def resolve_image(root, video_id, file_name):
    root = Path(root).resolve()
    candidates = [root / video_id / file_name, root / file_name,
                  root / "images" / video_id / file_name, root / "images" / file_name]
    matches = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Image path escapes image root: {candidate}")
        if candidate.is_file():
            matches.add(candidate)
    if not matches:
        raise FileNotFoundError(f"No image for video={video_id}, file={file_name}, root={root}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous image for {video_id}/{file_name}: {sorted(map(str, matches))}")
    return str(next(iter(matches)))
