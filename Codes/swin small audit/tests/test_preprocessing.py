"""Geometry and export checks, including odd sizes and very thin padding bands."""
from copy import deepcopy
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
from cataract_swin.preprocessing import (DEFAULT_PREPROCESSING, FitPad, fit_pad, geometry,
    mean_rgb, preprocessing_metadata, validate_preprocessing, validate_pretrained_contract,
    IMAGENET_MEAN, IMAGENET_STD)
from cataract_swin.preprocessing_audit import save_preprocessing_audit
from cataract_swin.reporting import copy_test_frames, export_analysis

ROOT = Path(__file__).resolve().parents[1]


def example(width, height):
    x, y = np.arange(width)[None, :], np.arange(height)[:, None]
    return Image.fromarray(np.stack(np.broadcast_arrays(
        30+x % 180, 35+y % 170, 40+(x+y) % 160), axis=-1).astype(np.uint8))


class PaddingTests(unittest.TestCase):
    def test_full_resized_frame_preserved_for_all_aspect_ratios(self):
        for width, height in [(640,360),(360,640),(317,317),(1919,1081),(224,223),(223,224),(1,300),(300,1)]:
            with self.subTest(size=(width,height)):
                image = example(width,height)
                result = np.asarray(fit_pad(image))
                g = geometry(width,height)
                w,h,left,top = (g[k] for k in ['resized_width','resized_height','pad_left','pad_top'])
                self.assertEqual(result.shape,(224,224,3))
                np.testing.assert_array_equal(result[top:top+h,left:left+w],
                    np.asarray(image.resize((w,h),Image.Resampling.BICUBIC)))
                self.assertLessEqual(abs(g['pad_left']-g['pad_right']),1)
                self.assertLessEqual(abs(g['pad_top']-g['pad_bottom']),1)
                self.assertEqual(max(w,h),224)
                self.assertAlmostEqual(g['padding_fraction'],1-w*h/224**2)

    def test_blend_boundary_matches_edge_and_far_padding_reaches_mean(self):
        for width,height in [(640,320),(320,640)]:
            image = example(width,height)
            padded = np.asarray(fit_pad(image))
            g = geometry(width,height)
            if width>height:
                top,bottom = g['pad_top'],g['pad_top']+g['resized_height']
                np.testing.assert_array_equal(padded[top-1],padded[top])
                np.testing.assert_array_equal(padded[bottom],padded[bottom-1])
                np.testing.assert_array_equal(padded[0],np.tile(mean_rgb(image),(224,1)))
                np.testing.assert_array_equal(padded[-1],np.tile(mean_rgb(image),(224,1)))
            else:
                left,right = g['pad_left'],g['pad_left']+g['resized_width']
                np.testing.assert_array_equal(padded[:,left-1],padded[:,left])
                np.testing.assert_array_equal(padded[:,right],padded[:,right-1])
                np.testing.assert_array_equal(padded[:,0],np.tile(mean_rgb(image),(224,1)))
                np.testing.assert_array_equal(padded[:,-1],np.tile(mean_rgb(image),(224,1)))
            self.assertGreater(padded.min(),0)

    def test_constant_images_stay_constant_and_square_images_have_no_padding(self):
        for dims in [(700,400),(400,700),(399,399)]:
            result = np.asarray(fit_pad(Image.new('RGB',dims,(67,124,89))))
            np.testing.assert_array_equal(result,np.broadcast_to([67,124,89],result.shape))
        self.assertEqual(geometry(399,399)['padding_fraction'],0)

    def test_mean_mode_uses_original_image_mean_and_rgb_conversion_is_shared(self):
        cfg = deepcopy(DEFAULT_PREPROCESSING);cfg['padding']['method']='mean'
        image = example(320,640)
        actual = np.asarray(fit_pad(image,cfg))
        np.testing.assert_array_equal(actual[:,0],np.tile(mean_rgb(image),(224,1)))
        self.assertEqual(fit_pad(Image.new('L',(90,60),120)).mode,'RGB')
        np.testing.assert_array_equal(np.asarray(FitPad()(image)),np.asarray(fit_pad(image)))

    def test_invalid_config_is_rejected_and_fingerprint_tracks_padding(self):
        for key,value in [('input_size',384),('input_size',True),('normalization','per_image'),('interpolation','bilinear')]:
            cfg = deepcopy(DEFAULT_PREPROCESSING);cfg[key]=value
            with self.assertRaises(ValueError):validate_preprocessing(cfg)
        for key,value in [('transition_fraction',0),('transition_fraction',float('nan')),('blur_radius',-1),('method','reflect')]:
            cfg = deepcopy(DEFAULT_PREPROCESSING);cfg['padding'][key]=value
            with self.assertRaises(ValueError):validate_preprocessing(cfg)
        cfg = deepcopy(DEFAULT_PREPROCESSING);cfg['padding']['method']='mean'
        self.assertNotEqual(preprocessing_metadata(cfg)['fingerprint'],preprocessing_metadata()['fingerprint'])
        with self.assertRaises(ValueError):geometry(0,10)

    def test_pretrained_normalization_resolution_contract(self):
        contract={'input_size':(3,224,224),'mean':IMAGENET_MEAN,'std':IMAGENET_STD,'interpolation':'bicubic'}
        validate_pretrained_contract(contract)
        for key,value in [('input_size',(3,384,384)),('mean',(0.5,0.5,0.5)),('std',(1,1,1))]:
            wrong={**contract,key:value}
            with self.assertRaises(ValueError):validate_pretrained_contract(wrong)

    def test_non_blocking_transfer_is_only_epoch_pass_algorithm_change(self):
        tree=ast.parse((ROOT/'cataract_swin/engine.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='epoch_pass')
        for call in ast.walk(node):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='to':
                call.keywords=[kw for kw in call.keywords if kw.arg!='non_blocking']
        actual=hashlib.sha256(ast.dump(ast.Module(body=[node],type_ignores=[]),include_attributes=False).encode()).hexdigest()
        reference=json.loads((ROOT/'provenance/notebook_equivalence.json').read_text())
        self.assertEqual(actual,reference['definitions']['epoch_pass']['ast_sha256'])


class PaddingExportTests(unittest.TestCase):
    def test_export_pixels_match_loader_callable_and_replace_stale_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'frame.png';image=example(360,640);image.save(path)
            frame=pd.DataFrame([dict(file_name=path.name,video_id='v1',label=0,prediction=1,
                true_class='low',predicted_class='dense',confidence=.9,correct=False,
                image_path=str(path),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
            options=dict(test_originals=True,test_inputs=True)
            folder=root/'results'
            index=copy_test_frames(frame,['low','dense'],folder,options)
            view_path=folder/index.model_input_rgb.iloc[0]
            expected=np.asarray(FitPad()(image))
            with Image.open(view_path) as view:np.testing.assert_array_equal(np.asarray(view),expected)
            view_path.write_bytes(b'corrupt old export')
            copy_test_frames(frame,['low','dense'],folder,options)
            with Image.open(view_path) as view:np.testing.assert_array_equal(np.asarray(view),expected)
            Image.new('RGB',(20,20),'red').save(path)
            copy_test_frames(frame,['low','dense'],folder,options)
            with Image.open(view_path) as view:np.testing.assert_array_equal(np.asarray(view),expected)
            with self.assertRaisesRegex(ValueError,'source checksum mismatch'):
                copy_test_frames(frame,['low','dense'],folder,dict(test_originals=False,test_inputs=True))
            path.unlink()  # Re-export can use the retained original if source was moved.
            copy_test_frames(frame,['low','dense'],folder,options)
            self.assertEqual(json.loads((folder/'test_images/preprocessing.json').read_text()),preprocessing_metadata())

    def test_report_refuses_padding_different_from_training_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            cfg=deepcopy(DEFAULT_PREPROCESSING);cfg['padding']['method']='mean'
            (folder/'protocol.json').write_text(json.dumps({'preprocessing':preprocessing_metadata(cfg)}))
            with self.assertRaisesRegex(ValueError,'differs from the trained model'):
                export_analysis(folder,pd.DataFrame(),['normal','poor'],{},DEFAULT_PREPROCESSING)

    def test_audit_has_every_frame_and_previews_use_training_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for label in range(2):
                for split in ['train','val','test']:
                    path=root/f'{label}_{split}.png';example(480,640).save(path)
                    rows.append(dict(file_name=path.name,video_id=f'v{label}_{split}',label=label,split=split,
                        image_path=str(path),original_width=480,original_height=640))
            manifest=pd.DataFrame(rows)
            save_preprocessing_audit(manifest,['low','dense'],root,DEFAULT_PREPROCESSING)
            audit=pd.read_csv(root/'preprocessing_frames.csv')
            self.assertEqual(len(audit),len(manifest))
            self.assertTrue(audit.pad_left.eq(28).all())
            previews=pd.read_csv(root/'preprocessing_previews/index.csv')
            self.assertEqual(len(previews),2);self.assertTrue(previews.split.eq('train').all())
            for name in previews.preview:
                with Image.open(root/'preprocessing_previews'/name) as im:im.verify()


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ['torch','torchvision']),
                     'PyTorch and torchvision are required for tensor/augmentation checks.')
class TensorPipelineTests(unittest.TestCase):
    def test_actual_swin_output_heads_and_pretrained_contract_without_downloads(self):
        if importlib.util.find_spec('timm') is None:
            self.skipTest('timm is required for the actual Swin forward check.')
        import torch
        from timm.models import get_pretrained_cfg
        from cataract_swin.model import TransformerClassifier,REGISTRY
        self.assertEqual(get_pretrained_cfg('swin_small_patch4_window7_224').tag,
                         get_pretrained_cfg(REGISTRY['Swin-Small']).tag)
        before_threads=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]),torch.inference_mode():
                for classes,shape in [(['low','dense','mature','brunescent'],(1,4)),(['normal','poor'],(1,))]:
                    model=TransformerClassifier('Swin-Small',classes,
                        dict(l2_features=True,tune_mode='all',tune_depth=4),pretrained=False).eval()
                    validate_pretrained_contract(model.backbone.pretrained_cfg)
                    output=model(torch.zeros(1,3,224,224))
                    self.assertEqual(tuple(output.shape),shape)
                    self.assertTrue(torch.isfinite(output).all().item())
                    del model
        finally:
            torch.set_num_threads(before_threads)

    def test_dataset_tensor_matches_exact_padded_rgb_and_imagenet_formula(self):
        import torch
        from torchvision.transforms import functional as TF
        from cataract_swin.transforms import FrameDataset, make_loaders
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frame.png';image=example(480,640);image.save(path)
            df=pd.DataFrame([{'image_path':str(path),'label':v,'split':s}
                for s in ['train','val','test'] for v in [0,0,0,1]])
            dataset=FrameDataset(df[df.split=='test'],False)
            actual,label=dataset[0]
            expected=TF.normalize(TF.to_tensor(FitPad()(image)),IMAGENET_MEAN,IMAGENET_STD)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertEqual(label,0)
            loaders=make_loaders(df,{'batch_size':2,'workers':0})
            sampler=loaders['train'].sampler
            self.assertEqual(sampler.num_samples,4);self.assertTrue(sampler.replacement)
            torch.testing.assert_close(sampler.weights,torch.tensor([1/3,1/3,1/3,1.],dtype=torch.double))

    def test_mean_fill_uses_original_random_draws_and_preserves_constant_colour(self):
        import torch
        from torchvision import transforms
        from cataract_swin.transforms import MeanFillRotation, MeanFillTranslation
        image=Image.new('RGB',(224,224),(57,91,123))
        torch.manual_seed(917)
        before=torch.get_rng_state().clone()
        original=transforms.Compose([transforms.RandomHorizontalFlip(.5),transforms.RandomVerticalFlip(.3),
            transforms.RandomRotation(15),transforms.RandomAffine(0,translate=(.1,.1))])
        original(image);expected_rng=torch.get_rng_state().clone()
        torch.set_rng_state(before)
        revised=transforms.Compose([transforms.RandomHorizontalFlip(.5),transforms.RandomVerticalFlip(.3),
            MeanFillRotation(),MeanFillTranslation()])
        actual=revised(image)
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
        np.testing.assert_array_equal(np.asarray(actual),np.asarray(image))


if __name__=='__main__':unittest.main()
