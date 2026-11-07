"""Additional epoch records without model operations."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from .io_utils import save_json


def write_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temp,index=False);temp.replace(path)


class TrainingRecorder:
    """Observe pass returns and checkpoint payloads without modifying their values.

    Epoch numbers come from the loop's epoch record, not from pass counts.
    No additional model forward pass, RNG draw or data-loader iteration is made.
    """
    def __init__(self,engine,folder,classes,options):
        self.engine=engine;self.folder=Path(folder);self.classes=classes;self.options=options
        self.original_epoch_pass=engine.epoch_pass
        self.pending={};self.durations={}

    def install(self):
        self.engine.epoch_pass=self.epoch_pass
        self.engine.record_epoch=self.record_epoch

    def epoch_pass(self,model,loader,device,optimizer=None,scaler=None):
        started=time.perf_counter()
        if hasattr(model,'named_parameters') and not (self.folder/'model_architecture.txt').exists():
            self.folder.mkdir(parents=True,exist_ok=True)
            (self.folder/'model_architecture.txt').write_text(str(model),encoding='utf-8')
            parameters=[{'name':name,'shape':str(list(p.shape)),'parameters':p.numel(),'trainable':p.requires_grad}
                        for name,p in model.named_parameters()]
            write_csv(pd.DataFrame(parameters),self.folder/'parameter_inventory.csv')
        result=self.original_epoch_pass(model,loader,device,optimizer,scaler)
        loss,y,p=result
        if optimizer is not None:
            self.pending={'train':(loss,y,p,None)}
            self.durations={'train_seconds':time.perf_counter()-started}
        elif 'train' in self.pending:
            self.pending['val']=(loss,y,p,loader.dataset.df)
            self.durations['val_seconds']=time.perf_counter()-started
        return result

    def record_epoch(self,obj):
        epoch=int(obj['epoch'])+1
        extra=dict(obj['history'][-1]);extra.update(self.durations)
        for split,(loss,y,p,rows) in self.pending.items():
            pred=p.argmax(1)
            extra[split+'_macro_f1']=float(f1_score(y,pred,labels=range(len(self.classes)),average='macro',zero_division=0))
            extra[split+'_weighted_f1']=float(f1_score(y,pred,labels=range(len(self.classes)),average='weighted',zero_division=0))
            if self.options['epoch_predictions']:
                if rows is None:
                    # Weighted sampling changes ordering; never attach false file identities.
                    frame=pd.DataFrame({'draw_index':np.arange(len(y)), 'label':y})
                else:
                    if not np.array_equal(rows.label.to_numpy(),y):
                        raise ValueError('Validation prediction order changed.')
                    frame=rows[['file_name','video_id','label']].copy()
                frame['prediction']=pred
                for i in range(len(self.classes)):frame[f'prob_{i}']=p[:,i]
                frame['confidence']=p.max(1)
                write_csv(frame,self.folder/'epoch_predictions'/f'{split}_epoch_{epoch:03d}.csv')
        metrics_path=self.folder/'epoch_metrics.csv'
        previous=pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        if not previous.empty:previous=previous[previous.epoch<epoch]
        write_csv(pd.concat([previous,pd.DataFrame([extra])],ignore_index=True),metrics_path)
        save_json(self.folder/'training_progress.json',{
            'last_completed_epoch':epoch,'best_validation_loss':float(obj['best_loss']),
            'bad_epochs':int(obj['bad_epochs']), 'phase':'epoch_finished', 'best_validation_macro_f1':float(obj['best_macro_f1']),
            'note':'Training accuracy/F1 reflect the sampled augmented training stream, not a deterministic full-training-set evaluation.'})
        self.pending={};self.durations={}
