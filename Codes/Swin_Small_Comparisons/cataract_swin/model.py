import torch
from torch import nn
import torch.nn.functional as F
import timm
from .config import DEFAULT_MODEL, MODEL_REGISTRY as REGISTRY
from .io_utils import require

TUNE_OVERRIDES = {}


class ImageClassifier(nn.Module):
    def __init__(self, name, classes, cfg, pretrained=True, input_size=448):
        super().__init__()
        # Swin builds its patch grid at construction. ResNet pools arbitrary spatial sizes.
        kwargs = {'img_size': input_size} if name == DEFAULT_MODEL else {}
        self.backbone = timm.create_model(REGISTRY[name], pretrained=pretrained, num_classes=0,
                                          **kwargs)
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
                stages = (list(self.backbone.layers) if name == DEFAULT_MODEL else
                          [self.backbone.layer1, self.backbone.layer2,
                           self.backbone.layer3, self.backbone.layer4])
                require(1<=self.depth<=len(stages), 'Invalid stage depth')
                self.unfrozen_modules = stages[-self.depth:]
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


# Keep the import used by older Swin analysis scripts.
TransformerClassifier = ImageClassifier
