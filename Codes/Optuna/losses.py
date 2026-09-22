"""
losses.py
---------
Loss function registry providing task-specific objectives,
class-weight penalty vectors, and label smoothing.
"""

from typing import Optional
import torch
import torch.nn as nn
from config import TaskConfig


def get_loss_function(
    task: str,
    label_smoothing: float = 0.05,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Builds and returns the criterion for the specified task.

    Args:
        task: 'severity' or 'poor_dilation'
        label_smoothing: float between [0.0, 0.1]
        device: torch.device instance to place class weights on

    Returns:
        Configured nn.CrossEntropyLoss module
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task == "severity":
        # Multi-class nuclear severity with minority class penalty
        weights = torch.tensor(TaskConfig.SEVERITY_LOSS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    elif task == "poor_dilation":
        # Binary pupil dilation task
        weights = torch.tensor(TaskConfig.DILATION_LOSS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    else:
        raise ValueError(f"Unsupported task for loss assignment: {task}")

    return criterion