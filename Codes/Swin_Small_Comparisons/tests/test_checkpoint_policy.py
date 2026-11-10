"""Execute the actual selection loop with simulated compute and checkpoint I/O."""
from pathlib import Path
import ast,copy,gc,json,platform,tempfile,types,unittest
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,roc_auc_score
from cataract_swin.config import DEFAULT_MODEL, model_preprocessing_metadata
from cataract_swin.io_utils import require,table_hash,save_json
from cataract_swin.preprocessing import (validate_preprocessing, preprocessing_metadata,
    validate_pretrained_contract, IMAGENET_MEAN, IMAGENET_STD, MODEL_NAME)


class Weight:
    requires_grad=True
    def __init__(self,value=0):self.value=value
    def detach(self):return self
    def cpu(self):return self
    def clone(self):return copy.deepcopy(self)
    def numel(self):return 1


class Model:
    def __init__(self,*args,**kwargs):
        self.weight=Weight();self.backbone=types.SimpleNamespace(parameters=lambda:[self.weight],pretrained_cfg={
            'input_size':(3,224,224),'mean':IMAGENET_MEAN,'std':IMAGENET_STD,'interpolation':'bicubic'},patch_embed=types.SimpleNamespace(img_size=(kwargs.get('input_size',448),)*2))
        self.classifier=types.SimpleNamespace(parameters=lambda:[self.weight])
    def to(self,device):return self
    def named_parameters(self):return [('weight',self.weight)]
    def state_dict(self):return {'weight':self.weight}
    def load_state_dict(self,state):self.weight=copy.deepcopy(state['weight'])


class Optimizer:
    def __init__(self,groups,**kwargs):self.param_groups=groups
    def state_dict(self):return {}


class Scheduler:
    def __init__(self,*args,**kwargs):self.steps=0
    def step(self):self.steps+=1
    def state_dict(self):return {'steps':self.steps}


class CheckpointPolicyTests(unittest.TestCase):
    def run_case(self,tie=False,selection='val_macro_f1'):
        path=Path(__file__).resolve().parents[1]/'cataract_swin/engine.py'
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'metrics','train_run'}]
        checkpoint={};written=[]
        def save(path,state):
            written.append(Path(path).name);checkpoint['state']=copy.deepcopy(state);Path(path).write_bytes(b'simulated checkpoint for test only')
        def load(*args,**kwargs):return copy.deepcopy(checkpoint['state'])
        fake_torch=types.SimpleNamespace(__version__='simulated',device=lambda s:s,
            cuda=types.SimpleNamespace(is_available=lambda:True,empty_cache=lambda:None,device_count=lambda:1),
            optim=types.SimpleNamespace(AdamW=Optimizer,lr_scheduler=types.SimpleNamespace(CosineAnnealingLR=Scheduler)),
            amp=types.SimpleNamespace(GradScaler=lambda *a:types.SimpleNamespace(state_dict=lambda:{})),load=load)
        y=np.array([0,0,1,1])
        def pass_epoch(model,loader,device,optimizer=None,scaler=None):
            if optimizer is not None:model.weight.value+=1
            epoch=model.weight.value
            pred={1:np.array([0,0,0,0]),2:y,3:y if tie else np.array([0,1,0,1])}[epoch]
            p=np.full((4,2),.1);p[np.arange(4),pred]=.9
            return {1:.4,2:.5,3:.3}[epoch],y,p
        rows=[dict(file_name=f'{s}_{i}.jpg',video_id=f'{s}_{i}',label=int(v),split=s,image_sha256=f'{s}_{i}')
              for s in ['train','val','test'] for i,v in enumerate(y)]
        df=pd.DataFrame(rows)
        loaders={s:types.SimpleNamespace(dataset=types.SimpleNamespace(df=df[df.split==s].reset_index(drop=True))) for s in ['train','val','test']}
        with tempfile.TemporaryDirectory() as tmp:
            namespace=dict(Path=Path,gc=gc,json=json,platform=platform,np=np,pd=pd,torch=fake_torch,
                timm=types.SimpleNamespace(__version__='1.0.22'),sklearn=sklearn,
                torchvision=types.SimpleNamespace(__version__='simulated'),PREPROCESSING=validate_preprocessing(),
                model_preprocessing_metadata=model_preprocessing_metadata,validate_pretrained_contract=validate_pretrained_contract,
                accuracy_score=accuracy_score,f1_score=f1_score,precision_score=precision_score,recall_score=recall_score,
                confusion_matrix=confusion_matrix,roc_auc_score=roc_auc_score,require=require,table_hash=table_hash,save_json=save_json,
                CONFIG=dict(seed=42,epochs=3,batch_size=32,workers=2,patience=5,backbone_lr=1e-5,head_lr=1e-4,
                    weight_decay=.01,tune_mode='all',tune_depth=4,l2_features=True,selection_metric=selection),TUNE_OVERRIDES={},
                SOURCES={'poor_dilation':{'classes':['normal','poor dilation']}},OUTPUT_ROOT=Path(tmp),
                REGISTRY={'Swin-Small':MODEL_NAME},ImageClassifier=Model,DEFAULT_MODEL=DEFAULT_MODEL,
                seed_all=lambda seed:None,make_loaders=lambda *a,**kw:loaders,epoch_pass=pass_epoch,
                save_checkpoint=save,rng_state=lambda:{},record_epoch=lambda s:None)
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-selection-loop>','exec'),namespace)
            namespace['train_run']('poor_dilation','Swin-Small',df)
            result=json.loads((Path(tmp)/'poor_dilation/Swin-Small/metrics.json').read_text())
            expected_epoch=3 if selection=='val_loss' else 2
            self.assertEqual(result['selection_metric'],selection)
            self.assertEqual(result['best_epoch'],expected_epoch)
            self.assertEqual(result['best_val_macro_f1'],1.)
            self.assertEqual(result['selected_val_macro_f1'],.5 if selection=='val_loss' else 1.)
            self.assertEqual(result['selected_val_loss'],.3 if selection=='val_loss' else .5)
            self.assertEqual(result['minimum_val_loss_seen'],.3)
            self.assertEqual(checkpoint['state']['scheduler']['steps'],expected_epoch)
            self.assertEqual(checkpoint['state']['model']['weight'].value,expected_epoch)
            self.assertEqual(set(written),{'best.pth'})
            self.assertEqual(len(list(Path(tmp).rglob('*.pth'))),1)
            self.assertIn('optimizer',checkpoint['state']);self.assertIn('rng',checkpoint['state'])
            self.assertEqual(checkpoint['state']['protocol']['preprocessing'],preprocessing_metadata())
    def test_macro_f1_overrules_lower_validation_loss(self):self.run_case()
    def test_equal_macro_f1_keeps_earlier_epoch(self):self.run_case(tie=True)
    def test_validation_loss_can_select_a_lower_f1_checkpoint(self):self.run_case(selection='val_loss')
