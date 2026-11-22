"""Pinned timm 1.0.22 backbones and tensor layouts for poor_dilation experiments."""
MODEL_REGISTRY = {
    'ResNet18': 'resnet18.tv_in1k',
    'ResNet34': 'resnet34.tv_in1k',
    'ResNet50': 'resnet50.tv2_in1k',
    **{f'EfficientNet-B{i}': f'tf_efficientnet_b{i}.in1k' for i in range(6)},
    'DenseNet121': 'densenet121.tv_in1k',
    'DenseNet169': 'densenet169.tv_in1k',
    'VGG16': 'vgg16.tv_in1k',
    **{f'Swin-{size.title()}': f'swin_{size}_patch4_window7_224.ms_in1k'
       for size in ('tiny', 'small', 'base')},
    'ViT-B_16': 'vit_base_patch16_224.augreg_in1k',
    'ViT-B_32': 'vit_base_patch32_224.augreg_in1k',
    **{f'DeiT-{size.title()}': f'deit_{size}_patch16_224.fb_in1k'
       for size in ('tiny', 'small', 'base')},
    'DeiT-Small-Distilled': 'deit_small_distilled_patch16_224.fb_in1k',
}


def fixed_grid(name):
    return name.startswith(('ViT-', 'DeiT-', 'Swin-'))


def run_key(name):
    return name.lower().replace('-', '_')
