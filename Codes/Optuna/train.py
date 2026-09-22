"""
train.py
--------
Training loop, validation engine, learning rate scheduler management,
epoch-by-epoch text logging, and plot generation for clinical benchmarks.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import build_model, get_parameter_groups
from losses import get_loss_function
from metrics import compute_metrics, format_classification_report, save_confusion_matrix, get_class_names


def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch with gradient updates."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        task: str,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates the model and computes scalar metrics and predictions."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            total += targets.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    eval_loss = running_loss / total
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob, task)
    return eval_loss, metrics, y_true, y_pred, y_prob


def save_diagnostic_plots(history: Dict[str, List[float]], output_dir: str) -> None:
    """Generates and saves publication-quality loss, accuracy, and learning rate curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss & Accuracy/F1 Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("Training & Validation Loss", fontsize=13)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Accuracy plot
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc (%)", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc (%)", color="#d62728", linewidth=2,
                 linestyle="--")
    axes[1].set_title("Training & Validation Accuracy (%)", fontsize=13)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Macro F1 & LR Schedule plot
    ax2 = axes[2]
    ax2.plot(epochs, [f * 100 for f in history["val_macro_f1"]], label="Val Macro F1 (%)", color="#9467bd", linewidth=2)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Macro F1 (%)", color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Twin axis for Learning Rate
    ax2_lr = ax2.twinx()
    ax2_lr.plot(epochs, history["learning_rate"], label="Head LR", color="#8c564b", linewidth=1.5, linestyle=":")
    ax2_lr.set_ylabel("Learning Rate", color="#8c564b")
    ax2_lr.set_yscale("log")
    ax2_lr.tick_params(axis="y", labelcolor="#8c564b")
    ax2.set_title("Validation F1 & LR Schedule", fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300)
    plt.close(fig)


def save_per_class_performance(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task: str,
        output_dir: str,
) -> None:
    """Plots and saves a per-class Precision, Recall, and F1 comparison bar chart."""
    from sklearn.metrics import precision_recall_fscore_support

    class_names = get_class_names(task)
    labels = list(range(len(class_names)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision * 100, width, label="Precision (%)", color="#4c72b0")
    ax.bar(x, recall * 100, width, label="Recall (%)", color="#55a868")
    ax.bar(x + width, f1 * 100, width, label="F1-Score (%)", color="#c44e52")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Per-Class Test Performance Profile", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=300)
    plt.close(fig)


def train_model(
        model_name: str,
        task: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        lr_head: float,
        lr_backbone: float,
        weight_decay: float,
        label_smoothing: float,
        output_dir: str,
        device: Optional[torch.device] = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
) -> Dict[str, Any]:

    """
    Executes final model training, validation monitoring,
    early stopping, best-checkpoint selection, and
    held-out test evaluation.
    """

    if device is None:

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    num_classes = (
        4 if task == "severity"
        else 2
    )

    # =========================================================
    # MODEL / LOSS / OPTIMIZER
    # =========================================================

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    criterion = get_loss_function(
        task=task,
        label_smoothing=label_smoothing,
        device=device
    )

    param_groups = get_parameter_groups(
        model,
        model_name,
        lr_backbone,
        lr_head,
        weight_decay
    )

    optimizer = torch.optim.AdamW(
        param_groups
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # =========================================================
    # HISTORY
    # =========================================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }

    best_val_f1 = -float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    epochs_trained = 0

    best_model_path = os.path.join(
        output_dir,
        "best_model.pth"
    )

    log_file_path = os.path.join(
        output_dir,
        "training_log.txt"
    )

    with open(
        log_file_path,
        "w"
    ) as log_file:

        log_file.write(
            f"Training Log: "
            f"{model_name} on {task}\n"
        )

        log_file.write(
            "=" * 80 + "\n\n"
        )

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    for epoch in range(
        1,
        num_epochs + 1
    ):

        epochs_trained = epoch

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # -----------------------------------------------------
        # Validate
        # -----------------------------------------------------

        (
            val_loss,
            val_metrics,
            _,
            _,
            _
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
            task
        )

        # -----------------------------------------------------
        # Scheduler
        # -----------------------------------------------------

        current_lr = (
            optimizer.param_groups[-1]["lr"]
        )

        scheduler.step(
            val_loss
        )

        # -----------------------------------------------------
        # History
        # -----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_acc"].append(
            val_metrics["accuracy"]
        )

        history["val_macro_f1"].append(
            val_metrics["macro_f1"]
        )

        history["learning_rate"].append(
            current_lr
        )

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        log_msg = (

            f"Epoch "
            f"[{epoch:02d}/{num_epochs:02d}] | "

            f"Train Loss: "
            f"{train_loss:.4f} | "

            f"Train Acc: "
            f"{train_acc * 100:.2f}% "
            f"({train_acc:.4f}) | "

            f"Val Loss: "
            f"{val_loss:.4f} | "

            f"Val Acc: "
            f"{val_metrics['accuracy'] * 100:.2f}% "
            f"({val_metrics['accuracy']:.4f}) | "

            f"Val F1: "
            f"{val_metrics['macro_f1'] * 100:.2f}% "
            f"({val_metrics['macro_f1']:.4f})"
        )

        print(
            log_msg
        )

        with open(
            log_file_path,
            "a"
        ) as log_file:

            log_file.write(
                log_msg + "\n"
            )

        # =====================================================
        # BEST CHECKPOINT
        # =====================================================

        current_f1 = (
            val_metrics["macro_f1"]
        )

        improvement = (
            current_f1 >
            best_val_f1
            +
            early_stopping_min_delta
        )

        if improvement:

            best_val_f1 = current_f1

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path
            )

            checkpoint_msg = (

                f" --> Saved Best Checkpoint "
                f"at Epoch {epoch:02d} "

                f"(Val Macro F1: "
                f"{best_val_f1:.4f})"
            )

            print(
                checkpoint_msg
            )

            with open(
                log_file_path,
                "a"
            ) as log_file:

                log_file.write(
                    checkpoint_msg + "\n"
                )

        else:

            epochs_without_improvement += 1

        # =====================================================
        # EARLY STOPPING
        # =====================================================

        if (
            epochs_without_improvement
            >= early_stopping_patience
        ):

            stop_msg = (

                f"\nEarly stopping at "
                f"epoch {epoch}. "

                f"No validation Macro-F1 "
                f"improvement for "
                f"{early_stopping_patience} "
                f"epochs.\n"

                f"Best epoch: "
                f"{best_epoch}\n"

                f"Best validation Macro-F1: "
                f"{best_val_f1:.4f}"
            )

            print(
                stop_msg
            )

            with open(
                log_file_path,
                "a"
            ) as log_file:

                log_file.write(
                    stop_msg + "\n"
                )

            break

    # =========================================================
    # TEST BEST MODEL
    # =========================================================

    print(
        "\nEvaluating best model "
        "on held-out test set..."
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    (
        test_loss,
        test_metrics,
        y_true_test,
        y_pred_test,
        _
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
        task
    )

    # =========================================================
    # SAVE DIAGNOSTICS
    # =========================================================

    save_diagnostic_plots(
        history,
        output_dir
    )

    save_confusion_matrix(
        y_true_test,
        y_pred_test,
        task,
        os.path.join(
            output_dir,
            "confusion_matrix.png"
        )
    )

    save_per_class_performance(
        y_true_test,
        y_pred_test,
        task,
        output_dir
    )

    # =========================================================
    # README
    # =========================================================

    report_text = (
        format_classification_report(
            y_true_test,
            y_pred_test,
            task
        )
    )

    readme_path = os.path.join(
        output_dir,
        "README.md"
    )

    with open(
        readme_path,
        "w"
    ) as f:

        f.write(
            f"# Benchmark Results: "
            f"{model_name} - {task}\n\n"
        )

        f.write(
            "## Model & Task Configuration\n"
        )

        f.write(
            f"- **Architecture:** "
            f"`{model_name}`\n"
        )

        f.write(
            f"- **Task:** "
            f"`{task}`\n"
        )

        f.write(
            f"- **Classes:** "
            f"`{get_class_names(task)}`\n"
        )

        f.write(
            f"- **Maximum Epochs:** "
            f"`{num_epochs}`\n"
        )

        f.write(
            f"- **Epochs Actually Trained:** "
            f"`{epochs_trained}`\n"
        )

        f.write(
            f"- **Best Epoch:** "
            f"`{best_epoch}`\n\n"
        )

        f.write(
            "## Selected Hyperparameters\n"
        )

        f.write(
            f"- **Head Learning Rate:** "
            f"`{lr_head:.6f}`\n"
        )

        f.write(
            f"- **Backbone Learning Rate:** "
            f"`{lr_backbone:.6f}`\n"
        )

        f.write(
            f"- **Weight Decay:** "
            f"`{weight_decay:.6e}`\n"
        )

        f.write(
            f"- **Label Smoothing:** "
            f"`{label_smoothing:.4f}`\n"
        )

        f.write(
            "- **Optimizer:** `AdamW`\n"
        )

        f.write(
            "- **LR Scheduler:** "
            "`ReduceLROnPlateau`\n\n"
        )

        f.write(
            "## Final Held-Out Test Metrics\n"
        )

        f.write(
            f"- **Test Accuracy:** "
            f"`{test_metrics['accuracy'] * 100:.2f}%` "
            f"(`{test_metrics['accuracy']:.4f}`)\n"
        )

        f.write(
            f"- **Test Macro F1:** "
            f"`{test_metrics['macro_f1'] * 100:.2f}%` "
            f"(`{test_metrics['macro_f1']:.4f}`)\n"
        )

        f.write(
            f"- **Test Weighted F1:** "
            f"`{test_metrics['weighted_f1'] * 100:.2f}%` "
            f"(`{test_metrics['weighted_f1']:.4f}`)\n"
        )

        f.write(
            f"- **Test ROC-AUC:** "
            f"`{test_metrics['roc_auc']:.4f}`\n\n"
        )

        f.write(
            "## Detailed Classification Report\n"
        )

        f.write(
            "```text\n"
            +
            report_text
            +
            "\n```\n"
        )

    # =========================================================
    # RETURN
    # =========================================================

    return {

        "best_val_macro_f1":
            best_val_f1,

        "best_epoch":
            best_epoch,

        "epochs_trained":
            epochs_trained,

        "test_metrics":
            test_metrics,

        "history":
            history,
    }