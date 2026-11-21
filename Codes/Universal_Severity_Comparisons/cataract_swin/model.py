"""Identical fine-tuning head for all registered backbones."""
import torch
from torch import nn
from torch.nn import functional as F
import timm
from .registry import MODEL_REGISTRY as REGISTRY, fixed_grid


class ImageClassifier(nn.Module):
    def __init__(self, name, classes, cfg, pretrained=True, input_size=448):
        super().__init__()
        extra = {'img_size': input_size} if fixed_grid(name) else {}
        if not name.startswith(('VGG', 'DenseNet')):
            extra['drop_path_rate'] = 0.0
        self.backbone = timm.create_model(REGISTRY[name], pretrained=pretrained, num_classes=0,
                                          drop_rate=0.0, **extra)
        self.binary = len(classes) == 2
        feature_dim = self.backbone.head_hidden_size if name == 'VGG16' else self.backbone.num_features
        self.classifier = nn.Linear(feature_dim, 1 if self.binary else len(classes))
        self.l2 = cfg['l2_features']

    def embeddings(self, x):
        features = self.backbone(x)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError('Expected one pooled feature vector per image')
        return F.normalize(features, dim=1) if self.l2 else features

    def forward(self, x):
        logits = self.classifier(self.embeddings(x))
        return logits.squeeze(-1) if self.binary else logits


TransformerClassifier = ImageClassifier
