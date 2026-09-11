"""
metrics.py
----------
Metric evaluation engine and diagnostic visualization tools.
Computes Accuracy, Macro F1, Weighted F1, ROC-AUC, and exports
normalized confusion matrix figures.
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

TASK_CLASS_NAMES = {
    "severity": ["low", "dense", "mature", "brunescent"],
    "poor_dilation": ["normal", "poor_dilation"],
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    task: str,
) -> Dict[str, float]:
    """Calculates all primary classification performance indicators.

    Args:
        y_true: Ground truth integer class array
        y_pred: Predicted class index array
        y_prob: Predicted softmax probability array (N, C)
        task: 'severity' or 'poor_dilation'

    Returns:
        Dictionary containing computed scalar metrics
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    results: Dict[str, float] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": 0.0,
    }

    # Compute ROC-AUC if probabilities are supplied
    if y_prob is not None:
        try:
            if task == "poor_dilation":
                # Binary ROC-AUC using positive class probabilities
                if y_prob.shape[1] == 2:
                    results["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    results["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            elif task == "severity":
                # Multi-class One-vs-Rest ROC-AUC
                results["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except ValueError:
            results["roc_auc"] = float("nan")

    return results


def get_class_names(task: str) -> List[str]:
    """Returns the ordered list of text labels for a task."""
    if task not in TASK_CLASS_NAMES:
        raise ValueError(f"Unknown task: {task}")
    return list(TASK_CLASS_NAMES[task])


def format_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
) -> str:
    """Returns a formatted classification report string."""
    target_names = get_class_names(task)
    labels = list(range(len(target_names)))

    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    output_path: str,
) -> None:
    """Renders and saves raw and normalized confusion matrix heatmaps side-by-side.

    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        task: 'severity' or 'poor_dilation'
        output_path: Target PNG file path
    """
    class_names = get_class_names(task)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized percentages heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
