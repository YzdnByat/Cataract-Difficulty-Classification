import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cataract_swin import engine
from cataract_swin.cli import save_video_assignments
from cataract_swin.config import load_config
from cataract_swin.experiments import DEFAULT_CONFIG
from cataract_swin.registry import MODEL_REGISTRY
from cataract_swin.transforms import make_loaders


class UniversalTests(unittest.TestCase):
    def test_weighted_accumulation_including_partial_window(self):
        torch.manual_seed(42)
        x = torch.randn(13, 3)
        y = torch.tensor([0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1])
        data = TensorDataset(x, y)
        data.df = pd.DataFrame({'label': y.numpy()})
        a = nn.Linear(3, 2)
        b = copy.deepcopy(a)
        weights = y.bincount().float().rsqrt()
        direct = torch.optim.SGD(a.parameters(), lr=0.01)
        for xb, yb in DataLoader(data, batch_size=8):
            direct.zero_grad()
            nn.functional.cross_entropy(a(xb), yb, weight=weights).backward()
            direct.step()
        accumulated = torch.optim.SGD(b.parameters(), lr=0.01)
        with patch.object(engine, 'CONFIG', {'micro_batch': 2, 'batch_size': 8, 'loss_power': 0.5}):
            engine.epoch_pass(b, DataLoader(data,batch_size=2), torch.device('cpu'),
                              accumulated, torch.amp.GradScaler('cpu',enabled=False))
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q)

    def test_sampler_uses_training_counts_and_common_seed(self):
        cfg, _, _ = load_config(DEFAULT_CONFIG)
        df = pd.DataFrame({'split': ['train']*4+['val','test'], 'label':[0,0,0,1,1,1],
                           'image_path':['unused']*6})
        a = make_loaders(df,cfg['training'],cfg['preprocessing'],data_seed=42)
        b = make_loaders(df,cfg['training'],cfg['preprocessing'],data_seed=42)
        torch.testing.assert_close(a['train'].sampler.weights,
                                   torch.tensor([3**-0.5]*3+[1.0],dtype=torch.float64))
        self.assertEqual(list(a['train'].sampler),list(b['train'].sampler))
        self.assertEqual(a['test'].batch_size,16)
        self.assertEqual(len(MODEL_REGISTRY),21)

    def test_excel_split_is_lossless(self):
        df=pd.DataFrame({'video_id':['A','B','C','C'], 'file_name':['a','b','c','d'], 'group_id':['A','B','C','C'],
                         'label':[0,1,0,0], 'source':['Farabi']*4,'split':['train','val','test','test']})
        with tempfile.TemporaryDirectory() as d:
            save_video_assignments(df,['normal','poor dilation'],Path(d))
            with pd.ExcelFile(Path(d)/'split_assignments.xlsx') as book:
                rows=pd.concat([book.parse(s+'_videos') for s in ('train','val','test')])
                self.assertEqual(rows.video_id.nunique(),3)
                self.assertEqual(rows.frames.sum(),4)
                self.assertEqual(book.parse('test_frames').file_name.tolist(),['c','d'])


if __name__ == '__main__':
    unittest.main()
