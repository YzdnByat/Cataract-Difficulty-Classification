"""Training, validation, out-of-fold prediction, and aggregation for K-fold CV."""

from __future__ import annotations

import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import build_fold_dataloaders
from losses import build_loss
from metrics import compute_metrics, save_confusion_matrix
from models import build_model, build_optimizer_groups, summarize_trainable_parameters


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else value
    return value


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        count = targets.size(0)
        loss_sum += loss.item() * count
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += count
    return loss_sum / total, correct / total


@torch.no_grad()
def _evaluate(model, loader, criterion, device, task):
    model.eval()
    loss_sum = 0.0
    total = 0
    truths, predictions, probabilities = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        count = targets.size(0)
        loss_sum += loss.item() * count
        total += count
        truths.extend(targets.cpu().numpy())
        predictions.extend(probs.argmax(dim=1).cpu().numpy())
        probabilities.extend(probs.cpu().numpy())
    y_true = np.asarray(truths)
    y_pred = np.asarray(predictions)
    y_prob = np.asarray(probabilities)
    return loss_sum / total, compute_metrics(y_true, y_pred, y_prob, task), y_true, y_pred, y_prob


def train_fold(
    model_config: Dict[str, Any],
    task: str,
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    output_dir: str | Path,
    device: torch.device,
    num_workers: int,
    seed: int,
    batch_size_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Train one fold using notebook-exact hyperparameters and selection logic."""
    fold_seed = int(seed) + int(fold)
    set_global_seed(fold_seed)
    fold_dir = Path(output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_fold_dataloaders(
        train_df=train_df,
        val_df=val_df,
        image_dir=image_dir,
        model_config=model_config,
        batch_size=batch_size_override,
        num_workers=num_workers,
        seed=fold_seed,
    )
    model = build_model(
        model_name=model_config["model"],
        num_classes=int(model_config["num_classes"]),
        unfreeze_scopes=model_config["fine_tuning"]["unfreeze_scopes"],
        pretrained=bool(model_config["pretrained"]),
    ).to(device)
    criterion = build_loss(
        model_config["loss"]["class_weights"],
        model_config["loss"]["label_smoothing"],
        device,
    )
    groups = build_optimizer_groups(model, model_config["optimizer"]["parameter_groups"])
    optimizer = torch.optim.AdamW(groups)
    scheduler_kwargs = {
        "mode": model_config["scheduler"]["mode"],
        "factor": float(model_config["scheduler"]["factor"]),
        "patience": int(model_config["scheduler"]["patience"]),
    }
    if model_config["scheduler"].get("min_learning_rate") is not None:
        scheduler_kwargs["min_lr"] = float(model_config["scheduler"]["min_learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    checkpoint_path = fold_dir / "best_model.pth"
    history = []
    best_loss = float("inf")
    best_epoch = 0
    epochs = int(model_config["max_epochs"])
    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_loss, train_accuracy = _train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics, _, _, _ = _evaluate(
            model, val_loader, criterion, device, task
        )
        row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
        }
        for group_index, group in enumerate(optimizer.param_groups):
            safe_name = str(group.get("group_name", group_index)).replace(".", "_").replace("+", "_")
            row[f"lr_{safe_name}"] = float(group["lr"])
        history.append(row)
        print(
            f"fold={fold} epoch={epoch:02d}/{epochs:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    val_loss, val_metrics, y_true, y_pred, y_prob = _evaluate(
        model, val_loader, criterion, device, task
    )
    history_df = pd.DataFrame(history)
    history_df.to_csv(fold_dir / "history.csv", index=False)

    predictions = val_df[["row_id", "file_name", "video_id", "label", "fold"]].copy()
    predictions = predictions.rename(columns={"label": "true_label"})
    predictions["predicted_label"] = y_pred
    for index in range(y_prob.shape[1]):
        predictions[f"probability_{index}"] = y_prob[:, index]
    predictions.to_csv(fold_dir / "validation_predictions.csv", index=False)
    save_confusion_matrix(
        y_true,
        y_pred,
        task,
        str(fold_dir / "confusion_matrix.png"),
    )
    result = {
        "fold": int(fold),
        "model": model_config["model"],
        "task": task,
        "train_frames": int(len(train_df)),
        "train_videos": int(train_df["video_id"].nunique()),
        "val_frames": int(len(val_df)),
        "val_videos": int(val_df["video_id"].nunique()),
        "epochs_trained": epochs,
        "parameter_counts": summarize_trainable_parameters(model),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "selected_checkpoint_validation_loss": val_loss,
        "metrics": val_metrics,
    }
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, indent=2)

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def aggregate_completed_folds(
    model_dir: str | Path,
    task: str,
    expected_folds: Iterable[int],
) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    fold_results = []
    prediction_frames = []
    for fold in sorted(set(int(x) for x in expected_folds)):
        metrics_path = model_dir / f"fold_{fold}" / "metrics.json"
        predictions_path = model_dir / f"fold_{fold}" / "validation_predictions.csv"
        if metrics_path.exists() and predictions_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                fold_results.append(json.load(handle))
            prediction_frames.append(pd.read_csv(predictions_path))
    if not fold_results:
        return {"completed_folds": []}

    flat_rows = []
    for result in fold_results:
        flat_rows.append(
            {
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "best_validation_loss": result["best_validation_loss"],
                **result["metrics"],
            }
        )
    fold_metrics = pd.DataFrame(flat_rows).sort_values("fold")
    fold_metrics.to_csv(model_dir / "fold_metrics.csv", index=False)
    oof = pd.concat(prediction_frames, ignore_index=True).sort_values("row_id")
    oof.to_csv(model_dir / "out_of_fold_predictions.csv", index=False)

    probability_columns = sorted(
        [column for column in oof.columns if column.startswith("probability_")],
        key=lambda value: int(value.split("_")[-1]),
    )
    pooled = compute_metrics(
        oof["true_label"].to_numpy(),
        oof["predicted_label"].to_numpy(),
        oof[probability_columns].to_numpy(),
        task,
    )
    save_confusion_matrix(
        oof["true_label"].to_numpy(),
        oof["predicted_label"].to_numpy(),
        task,
        str(model_dir / "oof_confusion_matrix.png"),
    )
    summary_rows = []
    for metric in ["accuracy", "macro_f1", "weighted_f1", "roc_auc"]:
        values = pd.to_numeric(fold_metrics[metric], errors="coerce")
        summary_rows.append(
            {
                "metric": metric,
                "fold_mean": values.mean(),
                "fold_std": values.std(ddof=1) if len(values) > 1 else np.nan,
                "pooled_oof": pooled.get(metric),
            }
        )
    pd.DataFrame(summary_rows).to_csv(model_dir / "summary_metrics.csv", index=False)
    summary = {
        "completed_folds": sorted(fold_metrics["fold"].astype(int).tolist()),
        "expected_folds": sorted(set(int(x) for x in expected_folds)),
        "complete": set(fold_metrics["fold"].astype(int)) == set(expected_folds),
        "pooled_oof_metrics": pooled,
    }
    with (model_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2)
    return summary
