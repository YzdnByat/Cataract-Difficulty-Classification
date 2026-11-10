"""YAML configuration and notebook-default validation."""
from pathlib import Path
import json
import math
import re
import yaml
from .preprocessing import MODEL_NAME, preprocessing_metadata, validate_preprocessing

DEFAULT_MODEL = 'Swin-Small'
MODEL_REGISTRY = {
    DEFAULT_MODEL: MODEL_NAME,
    'ResNet50': 'resnet50.tv2_in1k',
}
SELECTION_METRICS = ('val_macro_f1', 'val_loss')

TRAINING_KEYS = {'seed','epochs','batch_size','workers','patience','backbone_lr','head_lr',
                 'weight_decay','tune_mode','tune_depth','l2_features'}
SAVE_KEYS = {'epoch_predictions','test_originals','test_inputs'}


def resolve_path(value, base):
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    return str((base/p).resolve() if not p.is_absolute() else p.resolve())


def model_preprocessing_metadata(config, model=DEFAULT_MODEL):
    # The pipeline is shared. Record the weights associated with this run.
    return {**preprocessing_metadata(config), 'pretrained_checkpoint': MODEL_REGISTRY[model]}


def load_config(path, *, images_zip=None, metadata=None, output_dir=None, cache_dir=None, run_name=None, gpu=None,
                model=None):
    path=Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as f:
        cfg=yaml.safe_load(f)
    if not isinstance(cfg,dict) or set(cfg)-{'model'}!={'task','run','data','training','preprocessing','save'}:
        raise ValueError('Expected task, run, data, training, preprocessing, save and an optional model.')
    cfg['model'] = model or cfg.get('model', DEFAULT_MODEL)
    if cfg['model'] not in MODEL_REGISTRY:
        raise ValueError(f'Choose a model from {", ".join(MODEL_REGISTRY)}.')
    cfg['preprocessing']=validate_preprocessing(cfg['preprocessing'])
    if cfg['task'] not in {'severity','poor_dilation'}:
        raise ValueError('task must be severity or poor_dilation.')
    if set(cfg['training'])-{'selection_metric'}!=TRAINING_KEYS:
        raise ValueError(f'training keys must be {sorted(TRAINING_KEYS)}')
    if set(cfg['save'])!=SAVE_KEYS or any(type(v) is not bool for v in cfg['save'].values()):
        raise ValueError('save entries must be the three documented boolean switches.')
    if set(cfg['run'])!={'name','output_dir','cache_dir','gpu'} or set(cfg['data'])!={'images_zip','metadata'}:
        raise ValueError('Unknown or missing run/data configuration keys.')
    training=cfg['training']
    if training.get('selection_metric', SELECTION_METRICS[0]) not in SELECTION_METRICS:
        raise ValueError('selection_metric must be val_macro_f1 or val_loss.')
    for key in ['seed','epochs','batch_size','workers','patience','tune_depth']:
        if type(training[key]) is not int:
            raise ValueError(f'{key} must be an integer.')
    if training['seed']<0 or training['workers']<0 or any(training[k]<1 for k in ['epochs','batch_size','patience','tune_depth']):
        raise ValueError('Invalid seed, workers, epoch count, batch size, patience or tune depth.')
    for key in ['backbone_lr','head_lr','weight_decay']:
        value=training[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError(f'{key} must be a finite nonnegative number.')
        training[key]=float(value)
    if type(training['l2_features']) is not bool or training['tune_mode'] not in {'all','head','last_stages'}:
        raise ValueError('Use a boolean l2_features and tune_mode all, head or last_stages.')
    cfg['run']['name']=run_name or cfg['run']['name']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',str(cfg['run']['name'])):
        raise ValueError('run.name must be a single directory name starting with a letter or digit.')
    cfg['run']['gpu']=str(gpu if gpu is not None else cfg['run']['gpu'])
    gpu_ids = cfg['run']['gpu'].split(',')
    if (not re.fullmatch(r'(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))?', cfg['run']['gpu'])
            or len(set(gpu_ids)) != len(gpu_ids)):
        raise ValueError('Choose one GPU or two distinct GPUs, for example --gpu 0 or --gpu 0,1.')
    cfg['run']['output_dir']=resolve_path(output_dir,Path.cwd()) if output_dir else resolve_path(cfg['run']['output_dir'],path.parent)
    cfg['run']['cache_dir']=resolve_path(cache_dir,Path.cwd()) if cache_dir else resolve_path(cfg['run']['cache_dir'],path.parent)
    for key,override in [('images_zip',images_zip),('metadata',metadata)]:
        cfg['data'][key]=resolve_path(override,Path.cwd()) if override else resolve_path(cfg['data'][key],path.parent)
    source_path=Path(__file__).with_name('sources.json')
    sources=json.loads(source_path.read_text(encoding='utf-8'))
    for spec in sources.values():
        spec['counts']={int(k):v for k,v in spec['counts'].items()}
    return cfg,sources,path
