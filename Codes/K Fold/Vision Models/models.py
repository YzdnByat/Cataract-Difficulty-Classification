"""Torchvision model factory and exact optimizer-group construction."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

import torch.nn as nn
from torchvision import models


MODEL_SPECS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "densenet121": (models.densenet121, models.DenseNet121_Weights),
    "densenet169": (models.densenet169, models.DenseNet169_Weights),
    "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights),
    "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights),
    "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights),
    "vgg16": (models.vgg16, models.VGG16_Weights),
}


def _scope_matches(parameter_name: str, scope: str) -> bool:
    """Match canonical config scopes to torchvision named parameters."""
    slice_match = re.fullmatch(r"features\[(\d+):\]", scope)
    if slice_match:
        match = re.match(r"features\.(\d+)(?:\.|$)", parameter_name)
        return bool(match and int(match.group(1)) >= int(slice_match.group(1)))
    return parameter_name == scope or parameter_name.startswith(scope + ".")


def _replace_head(model: nn.Module, model_name: str, num_classes: int) -> None:
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "vgg16":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def build_model(
    model_name: str,
    num_classes: int,
    unfreeze_scopes: Iterable[str],
    pretrained: bool = True,
) -> nn.Module:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model: {model_name}")
    constructor, weights_enum = MODEL_SPECS[model_name]
    model = constructor(weights=weights_enum.DEFAULT if pretrained else None)
    for parameter in model.parameters():
        parameter.requires_grad = False

    scopes = list(unfreeze_scopes)
    for name, parameter in model.named_parameters():
        if any(_scope_matches(name, scope) for scope in scopes):
            parameter.requires_grad = True

    # Source notebooks replace the classification head after the freeze step.
    _replace_head(model, model_name, int(num_classes))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    unmatched = [name for name in trainable if not any(_scope_matches(name, s) for s in scopes)]
    if unmatched:
        raise RuntimeError(f"Trainable parameters fall outside declared scopes: {unmatched[:5]}")
    for scope in scopes:
        if not any(_scope_matches(name, scope) for name in trainable):
            raise RuntimeError(f"Unfreeze scope matches no trainable parameters: {scope}")
    return model


def build_optimizer_groups(
    model: nn.Module, parameter_group_configs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Assign every trainable parameter to exactly one notebook-declared group."""
    named_trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assignments: Dict[str, int] = {}
    result = []
    for group_index, config in enumerate(parameter_group_configs):
        scopes = list(config["scopes"])
        names = [
            name for name in named_trainable
            if any(_scope_matches(name, scope) for scope in scopes)
        ]
        duplicates = [name for name in names if name in assignments]
        if duplicates:
            raise RuntimeError(f"Optimizer scopes overlap for parameters: {duplicates[:5]}")
        if not names:
            raise RuntimeError(f"Optimizer group matches no parameters: {scopes}")
        for name in names:
            assignments[name] = group_index
        result.append({
            "params": [named_trainable[name] for name in names],
            "lr": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "group_name": str(config.get("name", "+".join(scopes))),
        })
    missing = sorted(set(named_trainable) - set(assignments))
    if missing:
        raise RuntimeError(f"Trainable parameters missing from optimizer: {missing[:10]}")
    return result


def summarize_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}
