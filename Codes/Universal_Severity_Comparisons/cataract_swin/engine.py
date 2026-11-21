from pathlib import Path
import gc, json, platform, random
import numpy as np
import pandas as pd
import torch
from torch import nn
import timm, sklearn, torchvision
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from .io_utils import require, file_hash, table_hash, save_json
from .config import DEFAULT_MODEL, model_preprocessing_metadata
from .registry import fixed_grid
from .model import ImageClassifier, REGISTRY
from .transforms import FrameDataset, make_loaders
from .preprocessing import validate_preprocessing

CONFIG = {}
TUNE_OVERRIDES = {}
SOURCES = {}
OUTPUT_ROOT = None
PREPROCESSING = validate_preprocessing()

def configure(*, training, sources, output_root, preprocessing=None):
    global CONFIG, TUNE_OVERRIDES, SOURCES, OUTPUT_ROOT, PREPROCESSING
    CONFIG = training
    TUNE_OVERRIDES = {}
    SOURCES = sources
    OUTPUT_ROOT = Path(output_root)
    PREPROCESSING = validate_preprocessing(preprocessing)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def epoch_pass(model, loader, device, optimizer=None, scaler=None):
    from itertools import islice

    training = optimizer is not None
    model.train(training)
    forward = nn.DataParallel(model) if device.type == 'cuda' and torch.cuda.device_count() == 2 else model
    accumulation = CONFIG['batch_size'] // CONFIG['micro_batch'] if training else 1
    weights = None
    if training and not model.binary:
        counts = loader.dataset.df.label.value_counts().sort_index().to_numpy()
        weights = torch.tensor(counts, device=device, dtype=torch.float32).pow(-CONFIG['loss_power'])
        weights /= weights.mean()
    total, mass = 0., 0.
    truth, probabilities = [], []
    iterator = iter(loader)
    with torch.set_grad_enabled(training):
        while batches := list(islice(iterator, accumulation)):
            # One denominator across this entire update, including a short final batch.
            labels_all = torch.cat([labels for _, labels in batches]).to(device)
            denominator = weights[labels_all].sum().item() if weights is not None else len(labels_all)
            if training:
                optimizer.zero_grad(set_to_none=True)
            for images, labels in batches:
                x = images.to(device, non_blocking=True)
                y = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=training and device.type == 'cuda'):
                    logits = forward(x)
                    if model.binary:
                        loss = nn.functional.binary_cross_entropy_with_logits(logits, y.float(), reduction='sum')
                    else:
                        loss = nn.functional.cross_entropy(logits, y, weight=weights, reduction='sum')
                require(torch.isfinite(loss).item(), 'Non-finite loss')
                if training:
                    scaler.scale(loss / denominator).backward()
                total += loss.item()
                scores = logits.detach().float()
                if model.binary:
                    p = scores.sigmoid()
                    p = torch.stack((1 - p, p), dim=1)
                else:
                    p = scores.softmax(dim=1)
                truth.append(labels.numpy())
                probabilities.append(p.cpu().numpy())
            mass += denominator
            if training:
                scaler.step(optimizer)
                scaler.update()
    return total / mass, np.concatenate(truth), np.concatenate(probabilities)

def metrics(y,p):
    pred=p.argmax(axis=1); classes=list(range(p.shape[1]))
    out={'accuracy':float(accuracy_score(y,pred)),
         'macro_f1':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0)),
         'weighted_f1':float(f1_score(y,pred,labels=classes,average='weighted',zero_division=0)),
         'confusion_matrix':confusion_matrix(y,pred,labels=classes).tolist(),'n_frames':len(y)}
    if p.shape[1]==2:
        out.update(poor_precision=float(precision_score(y,pred,zero_division=0)),
                   poor_recall=float(recall_score(y,pred,zero_division=0)),
                   specificity=float(recall_score(y,pred,pos_label=0,zero_division=0)),
                   roc_auc=float(roc_auc_score(y,p[:,1])))
    return out

def save_checkpoint(path,obj):
    temp=path.with_suffix('.tmp'); torch.save(obj,temp); temp.replace(path)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])

def record_epoch(state):
    """Replaced by the recorder in the worker; no model operations here."""
    pass

def train_run(task,name,df,*,data_seed=None):
    seed_all(CONFIG['seed'])
    require(torch.cuda.is_available(),'Enable a Kaggle GPU before training')
    device=torch.device('cuda:0')
    folder=OUTPUT_ROOT/task/name.replace('/','_'); folder.mkdir(parents=True,exist_ok=True)
    mode,depth=TUNE_OVERRIDES.get(name,(CONFIG['tune_mode'],CONFIG['tune_depth']))
    classes=SOURCES[task]['classes']
    selection=CONFIG.get('selection_metric','val_macro_f1')
    spec=model_preprocessing_metadata(PREPROCESSING,name)
    protocol=dict(task=task,model=name,timm_name=REGISTRY[name],classes=classes,config=CONFIG,
                  tuning=[mode,depth],split_hash=table_hash(df,['file_name','video_id','label','image_sha256','split']),
                  preprocessing=spec,cuda_device_count=torch.cuda.device_count(),data_sampling_seed=data_seed,
                  selection_metric=selection,selection_mode='min' if selection=='val_loss' else 'max',
                  checkpoint_policy='best_only_padded_v3',
                  early_stopping_metric='val_loss',
                  versions=dict(torch=torch.__version__,torchvision=torchvision.__version__,timm=timm.__version__,sklearn=sklearn.__version__,
                                numpy=np.__version__,python=platform.python_version()))
    protocol_path=folder/'protocol.json'
    if protocol_path.exists():
        require(json.loads(protocol_path.read_text())==protocol,'Run protocol changed. Use a new RUN_NAME.')
    else: save_json(protocol_path,protocol)
    required=['best.pth','history.csv','val_predictions.csv','test_predictions.csv','metrics.json']
    if (folder/'complete.json').exists():
        require(all((folder/p).is_file() and (folder/p).stat().st_size for p in required),'Incomplete completed run')
        print('Skipping completed',task,name); return
    best_path=folder/'best.pth'
    model=ImageClassifier(name,classes,CONFIG,pretrained=not best_path.exists(),
                          input_size=PREPROCESSING['input_size']).to(device)
    pretrained_cfg=model.backbone.pretrained_cfg
    if fixed_grid(name):
        require(tuple(model.backbone.patch_embed.img_size) == (PREPROCESSING['input_size'],) * 2,
                'Model patch grid disagrees with the input size')
    # The common paper transform deliberately overrides native recipe defaults.
    train_counts = df.loc[df.split.eq('train')].label.value_counts().sort_index()
    loss_weights = train_counts.astype(float).pow(-CONFIG['loss_power'])
    actual_weights = loss_weights / loss_weights.mean()
    save_json(folder / 'optimization.json', {
        'optimizer': 'AdamW', 'scheduler': 'CosineAnnealingLR', 'T_max': CONFIG['epochs'],
        'global_micro_batch': CONFIG['micro_batch'], 'effective_batch': CONFIG['batch_size'],
        'accumulation_steps': CONFIG['batch_size'] // CONFIG['micro_batch'],
        'micro_batch_per_gpu': CONFIG['micro_batch'] // torch.cuda.device_count(),
        'parallelism': 'DataParallel' if torch.cuda.device_count() == 2 else 'single GPU',
        'sampling': 'with replacement, one training-set length per epoch',
        'train_counts': train_counts.tolist(),
        'sampler_weight_by_class': train_counts.astype(float).pow(-CONFIG['sampler_power']).tolist(),
        'training_CE_weights': actual_weights.tolist(), 'validation_and_test_CE': 'unweighted',
        'fully_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
    })
    save_json(folder/'preprocessing.json',spec)
    loaders=make_loaders(df,CONFIG,PREPROCESSING,data_seed=data_seed)
    backbone=[p for p in model.backbone.parameters() if p.requires_grad]
    groups=[{'params':model.classifier.parameters(),'lr':CONFIG['head_lr'],'name':'head'}]
    if backbone: groups.append({'params':backbone,'lr':CONFIG['backbone_lr'],'name':'backbone'})
    optimizer=torch.optim.AdamW(groups, weight_decay=CONFIG['weight_decay'], betas=tuple(CONFIG['betas']),
                                eps=CONFIG['epsilon'])
    print('Initial learning rates:',{g['name']:g['lr'] for g in optimizer.param_groups})
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    scaler=torch.amp.GradScaler('cuda')
    (folder/'trainable_parameters.txt').write_text('\n'.join(n for n,p in model.named_parameters() if p.requires_grad))
    save_json(folder/'pretrained_config.json',pretrained_cfg)
    start,best_loss,best_f1,bad,history=0,float('inf'),float('-inf'),0,[]
    if best_path.exists():
        state=torch.load(best_path,map_location='cpu',weights_only=False)
        require(state['protocol']==protocol,'Checkpoint protocol mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        start,best_loss,best_f1,bad,history=(state['epoch_index']+1,state['best_loss'],state['best_macro_f1'],state['bad_epochs'],state['history'])
        restore_rng(state['rng']); del state
        print('Resuming from the saved best epoch; later unsaved epochs will be repeated:',start)
    for epoch in range(start,CONFIG['epochs']):
        if bad>=CONFIG['patience']: break
        if data_seed is not None:
            # Match the data stream across backbones and after resuming from best.pth.
            loaders['train'].generator.manual_seed(data_seed + epoch)
        epoch_lrs={g['name']+'_lr':float(g['lr']) for g in optimizer.param_groups}
        train_loss,yt,pt=epoch_pass(model,loaders['train'],device,optimizer,scaler)
        val_loss,yv,pv=epoch_pass(model,loaders['val'],device)
        val_f1=float(f1_score(yv,pv.argmax(1),labels=list(range(len(classes))),average='macro',zero_division=0))
        history.append(dict(epoch=epoch+1,train_loss=train_loss,val_loss=val_loss,
                            train_accuracy=float(accuracy_score(yt,pt.argmax(1))),
                            val_accuracy=float(accuracy_score(yv,pv.argmax(1))),val_macro_f1=val_f1,**epoch_lrs))
        # Preserve the original loss-based early-stopping rule.
        loss_improved=val_loss<best_loss
        if loss_improved: best_loss,bad=val_loss,0
        else: bad+=1
        scheduler.step()
        f1_improved=val_f1>best_f1
        if f1_improved: best_f1=val_f1
        # An exact tie retains the earlier checkpoint.
        if (loss_improved if selection=='val_loss' else f1_improved):
            save_checkpoint(best_path,dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                epoch=epoch+1,epoch_index=epoch,val_loss=val_loss,val_macro_f1=val_f1,best_macro_f1=best_f1,
                best_loss=best_loss,bad_epochs=bad,history=history.copy(),rng=rng_state(),protocol=protocol))
        record_epoch(dict(epoch=epoch,history=history,best_loss=best_loss,best_macro_f1=best_f1,bad_epochs=bad))
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print(task,name,history[-1],flush=True)
    checkpoint=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    result=dict(task=task,model=name,best_epoch=checkpoint['epoch'],selection_metric=selection,
                best_val_macro_f1=best_f1,selected_val_macro_f1=checkpoint['val_macro_f1'],
                selected_val_loss=checkpoint['val_loss'],
                minimum_val_loss_seen=best_loss)
    for split in ['val','test']:
        loss,y,p=epoch_pass(model,loaders[split],device)
        result[split]=dict(loss=loss,**metrics(y,p))
        predictions=loaders[split].dataset.df[['file_name','video_id','label']].copy()
        require(np.array_equal(predictions.label.to_numpy(),y),'Prediction order mismatch')
        predictions['prediction']=p.argmax(1)
        for c in range(len(classes)): predictions[f'prob_{c}']=p[:,c]
        predictions.to_csv(folder/f'{split}_predictions.csv',index=False)
    save_json(folder/'metrics.json',result)
    save_json(folder/'complete.json',{'best_epoch':checkpoint['epoch'],'selection_metric':selection,'split_hash':protocol['split_hash']})
    print(json.dumps(result,indent=2))
    del model,optimizer,scheduler,scaler,checkpoint,loaders
    gc.collect(); torch.cuda.empty_cache()

