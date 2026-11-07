import torch
from torch import nn
import torch.nn.functional as F
import timm
from .io_utils import require
from .preprocessing import MODEL_NAME

REGISTRY = {'Swin-Small': MODEL_NAME}
TUNE_OVERRIDES = {}

class TransformerClassifier(nn.Module):
    def __init__(self, name, classes, cfg, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(REGISTRY[name], pretrained=pretrained, num_classes=0)
        self.binary = len(classes)==2
        self.classifier = nn.Linear(self.backbone.num_features, 1 if self.binary else len(classes))
        self.l2 = cfg['l2_features']
        self.mode, self.depth = TUNE_OVERRIDES.get(name,(cfg['tune_mode'],cfg['tune_depth']))
        self.unfrozen_modules = []
        if self.mode != 'all':
            self.backbone.requires_grad_(False)
            if self.mode == 'last_blocks':
                require(hasattr(self.backbone,'blocks'), 'last_blocks requires ViT/DeiT')
                require(1<=self.depth<=len(self.backbone.blocks), 'Invalid block depth')
                self.unfrozen_modules = list(self.backbone.blocks[-self.depth:])
            elif self.mode == 'last_stages':
                require(hasattr(self.backbone,'layers'), 'last_stages requires Swin')
                require(1<=self.depth<=len(self.backbone.layers), 'Invalid stage depth')
                self.unfrozen_modules = list(self.backbone.layers[-self.depth:])
            else:
                require(self.mode=='head', 'Unknown fine-tuning mode')
            if self.mode != 'head':
                for attr in ['norm','fc_norm']:
                    if isinstance(getattr(self.backbone,attr,None), nn.Module):
                        self.unfrozen_modules.append(getattr(self.backbone,attr))
            for module in self.unfrozen_modules:
                module.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.mode != 'all':
            self.backbone.eval()
            for module in self.unfrozen_modules:
                module.train(True)
        return self

    def embeddings(self, x):
        features = self.backbone(x)
        require(isinstance(features,torch.Tensor) and features.ndim==2, 'Expected pooled embeddings')
        return F.normalize(features,p=2,dim=1) if self.l2 else features

    def forward(self,x):
        logits = self.classifier(self.embeddings(x))
        return logits.squeeze(-1) if self.binary else logits
