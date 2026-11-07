"""Run with: python -m unittest discover -s tests -v"""
from pathlib import Path
import ast
import hashlib
import json
import os
import random
import tempfile
import types
import unittest
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from cataract_swin.config import load_config
from cataract_swin.data_io import assign_splits
from cataract_swin.reporting import export_analysis,load_predictions
from cataract_swin.tracking import TrainingRecorder
from cataract_swin.run_lock import RunLock,LockBusy
from cataract_swin.preprocessing import DEFAULT_PREPROCESSING

ROOT=Path(__file__).resolve().parents[1]


class PreservationTests(unittest.TestCase):
    def test_run_directory_lock_is_exclusive_and_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run.lock'
            with RunLock(path):
                with self.assertRaises(LockBusy):
                    with RunLock(path):pass
            with RunLock(path):pass

    def test_all_original_bodies_match_both_notebooks(self):
        reference=json.loads((ROOT/'provenance/notebook_equivalence.json').read_text())
        for name,entry in reference['definitions'].items():
            if name in reference.get('authorized_modified_definitions',[]):continue
            module=ROOT/'cataract_swin'/entry['module']
            node=next(n for n in ast.parse(module.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
            digest=hashlib.sha256(ast.dump(ast.Module(body=[node],type_ignores=[]),include_attributes=False).encode()).hexdigest()
            self.assertEqual(digest,entry['ast_sha256'],name)

    def test_yaml_defaults_and_path_resolution(self):
        for task in ['severity','poor_dilation']:
            cfg,sources,_=load_config(ROOT/'configs'/f'{task}.yaml')
            self.assertEqual(cfg['training'],dict(seed=42,epochs=30,batch_size=32,workers=2,patience=5,
                backbone_lr=1e-5,head_lr=1e-4,weight_decay=.01,tune_mode='all',tune_depth=4,l2_features=True))
            self.assertTrue(Path(cfg['run']['output_dir']).is_absolute())
            self.assertEqual(cfg['preprocessing'],DEFAULT_PREPROCESSING)
            self.assertEqual(sources[task]['counts'].keys(),set(range(len(sources[task]['classes']))))

    def test_notebook_split_is_grouped_stratified_and_repeatable(self):
        rows=[]
        for label in range(4):
            for video in range(24):
                for frame in range(2):
                    rows.append({'video_id':f'v{label}_{video:02d}','file_name':f'{label}_{video}_{frame}.jpg',
                                 'label':label,'image_sha256':f'{label}_{video}_{frame}'})
        df=pd.DataFrame(rows)
        a=assign_splits(df,42);b=assign_splits(df,42)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.groupby('video_id').split.nunique().max(),1)
        for split,g in a.groupby('split'):self.assertEqual(set(g.label),set(range(4)))

    def test_duplicate_bytes_across_splits_are_rejected(self):
        rows=[{'video_id':f'v{label}_{i:02d}','file_name':f'{label}_{i}.jpg','label':label,'image_sha256':f'{label}_{i}'}
              for label in range(2) for i in range(24)]
        df=pd.DataFrame(rows);split=assign_splits(df,42)
        train=split[split.split=='train'].index[0];test=split[split.split=='test'].index[0]
        df.loc[[train,test],'image_sha256']='same bytes'
        with self.assertRaisesRegex(ValueError,'Identical image bytes'):assign_splits(df,42)


class ArtifactTests(unittest.TestCase):
    def test_archive_includes_weights_and_preserves_full_run_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'experiment.v1';run.mkdir()
            (run/'best.pth').write_bytes(b'test checkpoint placeholder')
            (run/'metrics.json').write_text('{}')
            (run/'run.lock').write_text('')
            subprocess.run([sys.executable,str(ROOT/'archive_results.py'),'--run-dir',str(run)],check=True,capture_output=True,text=True)
            archive=Path(tmp)/'experiment.v1.zip'
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()),{'experiment.v1/best.pth','experiment.v1/metrics.json'})
                self.assertIsNone(z.testzip())

    def fixture(self,root,classes):
        rows=[];predictions=[]
        for split in ['train','val','test']:
            for label in range(len(classes)):
                for j in range(2):
                    fname=f'{split}_{label}_{j}.jpg';video=f'video_{split}_{label}'
                    path=root/'input'/video/fname;path.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(72,48),(40+label*30,80+j*15,130)).save(path)
                    rows.append({'file_name':fname,'video_id':video,'label':label,'image_path':str(path),
                                 'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'split':split})
                    predicted=label if j==0 else (label+1)%len(classes)
                    p=np.full(len(classes),.1/(len(classes)-1));p[predicted]=.9
                    predictions.append({'file_name':fname,'video_id':video,'label':label,'prediction':predicted,
                                        **{f'prob_{i}':float(v) for i,v in enumerate(p)}})
        manifest=pd.DataFrame(rows);all_predictions=pd.DataFrame(predictions)
        folder=root/'run'/('severity' if len(classes)==4 else 'poor_dilation')/'Swin-Small';folder.mkdir(parents=True)
        for split in ['val','test']:
            names=set(manifest.loc[manifest.split==split,'file_name'])
            all_predictions[all_predictions.file_name.isin(names)].to_csv(folder/f'{split}_predictions.csv',index=False)
        pd.DataFrame([{'epoch':1,'train_loss':1.,'val_loss':1.1,'train_accuracy':.5,'val_accuracy':.5,'head_lr':.0001,'backbone_lr':.00001},
                      {'epoch':2,'train_loss':.8,'val_loss':.9,'train_accuracy':.6,'val_accuracy':.5,'head_lr':.00008,'backbone_lr':.000008}]).to_csv(folder/'history.csv',index=False)
        (folder/'metrics.json').write_text(json.dumps({'task':'severity' if len(classes)==4 else 'poor_dilation',
            'model':'Swin-Small','best_epoch':2,'best_val_loss':.9}))
        return manifest,folder

    def test_reports_images_and_manual_review_survive_regeneration(self):
        for classes in [['low nuclear density','dense','mature','brunescent'],['normal','poor dilation']]:
            with tempfile.TemporaryDirectory() as tmp:
                manifest,folder=self.fixture(Path(tmp),classes)
                original=(folder/'metrics.json').read_bytes()
                options=dict(epoch_weights=True,epoch_predictions=True,test_originals=True,test_inputs=True)
                export_analysis(folder,manifest,classes,options)
                complete=json.loads((folder/'analysis_complete.json').read_text())
                self.assertEqual(complete['test_frames'],len(classes)*2)
                self.assertEqual(complete['test_errors'],len(classes))
                self.assertEqual((folder/'metrics.json').read_bytes(),original)
                index=pd.read_csv(folder/'test_images/index.csv',keep_default_na=False)
                for r in index.itertuples():
                    self.assertEqual(hashlib.sha256((folder/r.copied_original).read_bytes()).hexdigest(),r.image_sha256)
                    with Image.open(folder/r.model_input_rgb) as im:self.assertEqual(im.size,(448,448))
                index.loc[0,'reviewer_notes']='Clinical review retained'
                index.to_csv(folder/'test_images/index.csv',index=False)
                export_analysis(folder,manifest,classes,options)
                self.assertEqual(pd.read_csv(folder/'test_images/index.csv').reviewer_notes.iloc[0],'Clinical review retained')
                for p in folder.rglob('*.png'):
                    with Image.open(p) as im:im.verify()
                self.assertTrue((folder/'test_gallery.html').is_file())
                cm=pd.read_csv(folder/'analysis/test/confusion_counts.csv',index_col=0)
                self.assertEqual(int(cm.to_numpy().sum()),len(classes)*2)
                roc=json.loads((folder/'analysis/test/roc_auc.json').read_text())
                self.assertIsNotNone(roc['macro_ovr_auc'])

    def test_missing_predictions_fail_instead_of_silent_row_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest,folder=self.fixture(Path(tmp),['normal','poor dilation'])
            p=folder/'test_predictions.csv';d=pd.read_csv(p);d.iloc[:-1].to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'cover their manifest exactly'):load_predictions(p,manifest,['normal','poor dilation'])

    def test_epoch_observer_keeps_returns_and_never_labels_sampler_draws_as_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            calls=[]
            result=(.5,np.array([1,0,1,0]),np.array([[.1,.9],[.8,.2],[.3,.7],[.6,.4]]))
            def original_pass(*args):return result
            def original_save(path,obj):
                calls.append((Path(path).name,obj));Path(path).write_bytes(b'test checkpoint placeholder')
            fake=types.SimpleNamespace(epoch_pass=original_pass,save_checkpoint=original_save)
            observer=TrainingRecorder(fake,root,['normal','poor dilation'],dict(epoch_weights=True,epoch_predictions=True))
            observer.install()
            loader=types.SimpleNamespace(dataset=types.SimpleNamespace(df=pd.DataFrame({'file_name':['a','b','c','d'],'video_id':['x']*4,'label':result[1]})))
            self.assertIs(fake.epoch_pass(None,loader,None,object(),None),result)
            self.assertIs(fake.epoch_pass(None,loader,None,None,None),result)
            payload={'epoch':0,'model':{},'protocol':{},'best_loss':.5,'bad_epochs':0,
                     'history':[dict(epoch=1,train_loss=.5,val_loss=.5,train_accuracy=1.,val_accuracy=1.,head_lr=.0001,backbone_lr=.00001)]}
            payload['best_macro_f1']=1.0
            fake.record_epoch(payload)
            self.assertEqual(calls,[])
            train=pd.read_csv(root/'epoch_predictions/train_epoch_001.csv')
            self.assertNotIn('file_name',train)
            val=pd.read_csv(root/'epoch_predictions/val_epoch_001.csv')
            self.assertEqual(val.file_name.tolist(),['a','b','c','d'])
            self.assertFalse(list(root.rglob('*.pth')))


if __name__=='__main__':unittest.main()
