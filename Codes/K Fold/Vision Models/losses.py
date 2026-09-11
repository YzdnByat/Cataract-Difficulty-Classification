"""Loss builder using the class weights saved with each winning config."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn


def build_loss(
    class_weights: Optional[Iterable[float]],
    label_smoothing: float,
    device: torch.device,
) -> nn.Module:
    weights = None
    if class_weights is not None:
        weights = torch.tensor(list(class_weights), dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(label_smoothing),
    )
