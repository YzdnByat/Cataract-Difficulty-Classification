"""Fresh worker: notebook thread settings and one visible CUDA device."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import platform
import time
import pandas as pd
import torch
import torchvision
import timm
from . import engine
from .tracking import TrainingRecorder
from .reporting import export_analysis
from .io_utils import save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--settings',required=True)
    args=parser.parse_args()
    settings=json.loads(Path(args.settings).read_text())
    if timm.__version__!='1.0.22':
        raise RuntimeError('The source notebooks use timm==1.0.22. Install requirements.txt.')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Training needs one visible CUDA GPU. Select a GPU accelerator and pass --gpu 0 (or --gpu 1).')
    task=settings['task'];folder=Path(settings['output_root'])/task/'Swin-Small'
    folder.mkdir(parents=True,exist_ok=True)
    engine.configure(training=settings['config'],sources=settings['sources'],output_root=settings['output_root'],
                     preprocessing=settings['preprocessing'])
    df=pd.read_csv(settings['manifest'])
    environment={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
                 'timm':timm.__version__,'cuda_runtime':torch.version.cuda,
                 'gpu':torch.cuda.get_device_name(0),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','Pillow','PyYAML']}}
    save_json(folder/'environment.json',environment)
    observer=TrainingRecorder(engine,folder,settings['sources'][task]['classes'],settings['save'])
    observer.install()
    print('Task:',task,'Model: Swin-Small','CUDA:',environment['gpu'],flush=True)
    started=time.perf_counter()
    # Original optimization steps; user-authorized best-only macro-F1 checkpoint policy.
    engine.train_run(task,'Swin-Small',df)
    save_json(folder/'execution.json',{'train_invocation_seconds':time.perf_counter()-started})
    export_analysis(folder,df,settings['sources'][task]['classes'],settings['save'],settings['preprocessing'])
    print('Complete:',folder,flush=True)


if __name__=='__main__':main()
