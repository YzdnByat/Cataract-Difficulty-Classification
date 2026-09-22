"""
models.py
---------
Model factory implementing exact unfreezing depths, head replacements,
and discriminative layer-wise parameter group partitioning.
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Builds the specified CNN architecture, freezes early convolutional stages,
    unfreezes top stages, and replaces the classifier head.

    Args:
        model_name: Name of architecture (e.g., 'resnet18', 'densenet121', 'efficientnet_b3')
        num_classes: Number of target outputs (2 for dilation, 4 for severity)
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        Configured PyTorch nn.Module
    """
    model_name_clean = model_name.lower().replace("-", "_")

    # -------------------------------------------------------------
    # 1. ResNet Family (18, 34, 50)
    # -------------------------------------------------------------
    if "resnet" in model_name_clean:
        if model_name_clean == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        elif model_name_clean == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
        elif model_name_clean == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        else:
            raise NotImplementedError(f"Unsupported ResNet variant: {model_name}")

        # Policy: Freeze conv1, bn1, layer1, layer2; Unfreeze layer3, layer4, fc
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if name.startswith(("layer3", "layer4", "fc")):
                param.requires_grad = True

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # -------------------------------------------------------------
    # 2. DenseNet Family (121, 169)
    # -------------------------------------------------------------
    elif "densenet" in model_name_clean:
        if model_name_clean == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
        elif model_name_clean == "densenet169":
            weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
            model = models.densenet169(weights=weights)
        else:
            raise NotImplementedError(f"Unsupported DenseNet variant: {model_name}")

        # Policy: Freeze up to transition3; Unfreeze denseblock4, norm5, classifier
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if name.startswith(("features.denseblock4", "features.norm5", "classifier")):
                param.requires_grad = True

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    # -------------------------------------------------------------
    # 3. EfficientNet Family (B1 - B5)
    # -------------------------------------------------------------
    elif "efficientnet" in model_name_clean:
        eff_fn = getattr(models, model_name_clean, None)
        if eff_fn is None:
            raise NotImplementedError(f"Unsupported EfficientNet variant: {model_name}")

        # Dynamically fetch default pretrained weights
        variant = model_name_clean.split("_")[-1].upper()
        weight_attr = f"EfficientNet_{variant}_Weights"

        weights_class = getattr(models, weight_attr, None)

        if pretrained and weights_class is None:
            raise ValueError(
                f"Could not find torchvision weights class: {weight_attr}"
            )

        weights = weights_class.DEFAULT if pretrained else None

        model = eff_fn(weights=weights)

        # Policy: Freeze features[0..5]; Unfreeze features.6, features.7, classifier
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if name.startswith(
                    ("features.6", "features.7", "features.8", "classifier")
            ):
                param.requires_grad = True

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unrecognized architecture family for: {model_name}")


def get_parameter_groups(
    model: nn.Module,
    model_name: str,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float = 1e-4
) -> List[Dict[str, Any]]:
    """
    Partitions active parameters into discriminative learning rate groups for AdamW.
    Head parameters receive lr_head; unrolled deep conv layers receive lr_backbone.
    """
    model_name_clean = model_name.lower().replace("-", "_")

    head_params = []
    backbone_params = []

    if "resnet" in model_name_clean:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc"):
                head_params.append(param)
            else:
                backbone_params.append(param)

    elif "densenet" in model_name_clean:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("classifier"):
                head_params.append(param)
            else:
                backbone_params.append(param)

    elif "efficientnet" in model_name_clean:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("classifier"):
                head_params.append(param)
            else:
                backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]