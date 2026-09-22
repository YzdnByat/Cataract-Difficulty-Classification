"""
optimize.py
-----------
Optuna Hyperparameter Optimization (HPO) engine with MedianPruner
to search learning rates, weight decay, and label smoothing.
"""

from typing import Dict, Any, Tuple, Optional
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig
from models import build_model, get_parameter_groups
from losses import get_loss_function
from train import train_one_epoch, evaluate


def objective(
    trial: optuna.Trial,
    model_name: str,
    task: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Optuna objective function executed per trial."""
    # Sample hyperparameters
    lr_head = trial.suggest_float("lr_head", 1e-4, 1e-2, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 5e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.10)

    num_classes = 4 if task == "severity" else 2

    # Instantiate Model, Loss, and Optimizer
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=True).to(device)
    criterion = get_loss_function(task=task, label_smoothing=label_smoothing, device=device)
    param_groups = get_parameter_groups(model, model_name, lr_backbone, lr_head, weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    # Fast pruned training loop for HPO
    epochs = ModelConfig.OPTUNA_EPOCHS
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, task)

        # Report metric to Optuna for pruning
        target_score = val_metrics["macro_f1"]
        trial.report(target_score, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return target_score


def run_hpo(
    model_name: str,
    task: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_trials: int = ModelConfig.OPTUNA_TRIALS,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Creates study, executes trials, and returns optimal parameters."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimize for Macro F1 (maximize)
    sampler = TPESampler(seed=ModelConfig.RANDOM_STATE)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_name}_{task}_hpo",
    )

    print(f"\n>>> Starting Optuna Search: {model_name} on {task} ({n_trials} Trials) <<<")
    study.optimize(
        lambda trial: objective(trial, model_name, task, train_loader, val_loader, device),
        n_trials=n_trials,
        timeout=None,
    )

    print(f"\n[Optuna HPO Complete] Best Trial #{study.best_trial.number}")
    print(f"Best Macro F1 Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params