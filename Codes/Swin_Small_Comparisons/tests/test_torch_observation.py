"""Numeric observation equivalence when the actual PyTorch stack is installed."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

AVAILABLE=all(importlib.util.find_spec(m) is not None for m in ['torch','torchvision','timm'])


@unittest.skipUnless(AVAILABLE,'PyTorch, torchvision and timm are required for the numeric equivalence test.')
class NumericEquivalenceTests(unittest.TestCase):
    def test_observer_preserves_cpu_epoch_result_weights_and_torch_rng(self):
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset,DataLoader
        from cataract_swin import engine
        from cataract_swin.tracking import TrainingRecorder
        class TinyClassifier(nn.Module):
            binary=False
            def __init__(self):
                super().__init__();self.layer=nn.Linear(4,2)
            def forward(self,x):return self.layer(x)
        torch.manual_seed(31);reference=TinyClassifier();observed=copy.deepcopy(reference)
        data=TensorDataset(torch.arange(32,dtype=torch.float32).reshape(8,4)/32,torch.tensor([0,1]*4))
        baseline_optimizer=torch.optim.SGD(reference.parameters(),lr=.01)
        observed_optimizer=torch.optim.SGD(observed.parameters(),lr=.01)
        with tempfile.TemporaryDirectory() as tmp:
            original=engine.epoch_pass;original_save=engine.save_checkpoint;original_record=engine.record_epoch
            try:
                torch.manual_seed(101)
                a=original(reference,DataLoader(data,batch_size=4),torch.device('cpu'),baseline_optimizer,
                           torch.amp.GradScaler('cuda',enabled=False))
                reference_rng=torch.get_rng_state().clone()
                torch.manual_seed(101)
                observer=TrainingRecorder(engine,Path(tmp),['normal','poor dilation'],dict(epoch_weights=False,epoch_predictions=False))
                observer.install()
                b=engine.epoch_pass(observed,DataLoader(data,batch_size=4),torch.device('cpu'),observed_optimizer,
                                    torch.amp.GradScaler('cuda',enabled=False))
                self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
                for key,value in reference.state_dict().items():torch.testing.assert_close(value,observed.state_dict()[key],atol=0,rtol=0)
                torch.testing.assert_close(reference_rng,torch.get_rng_state(),atol=0,rtol=0)
            finally:engine.epoch_pass=original;engine.save_checkpoint=original_save;engine.record_epoch=original_record
