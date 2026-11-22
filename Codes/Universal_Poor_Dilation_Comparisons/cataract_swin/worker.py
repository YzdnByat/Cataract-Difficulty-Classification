"""Train a model in a fresh process on the selected CUDA devices."""
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
from .config import DEFAULT_MODEL
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
    expected_gpus=len(settings.get('gpu','0').split(','))
    if not torch.cuda.is_available() or torch.cuda.device_count()!=expected_gpus:
        raise RuntimeError(f'Training needs {expected_gpus} visible CUDA GPU(s). Check the accelerator and --gpu.')
    task=settings['task']
    model=settings.get('model',DEFAULT_MODEL)
    folder=Path(settings['output_root'])/task/model
    folder.mkdir(parents=True,exist_ok=True)
    engine.configure(training=settings['config'],sources=settings['sources'],output_root=settings['output_root'],
                     preprocessing=settings['preprocessing'])
    df=pd.read_csv(settings['manifest'])
    environment={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
                 'timm':timm.__version__,'cuda_runtime':torch.version.cuda,
                 'gpu':[torch.cuda.get_device_name(i) for i in range(expected_gpus)],
                 'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'packages':{p:importlib.metadata.version(p) for p in
                             ['numpy','pandas','scikit-learn','Pillow','PyYAML','albumentations','opencv-python-headless']}}
    save_json(folder/'environment.json',environment)
    observer=TrainingRecorder(engine,folder,settings['sources'][task]['classes'],settings['save'])
    observer.install()
    print('Task:',task,'Model:',model,'CUDA:',environment['gpu'],flush=True)
    started=time.perf_counter()
    data_seed=settings['config']['seed'] if settings.get('preset') is not None else None
    engine.train_run(task,model,df,data_seed=data_seed)
    save_json(folder/'execution.json',{'train_invocation_seconds':time.perf_counter()-started})
    export_analysis(folder,df,settings['sources'][task]['classes'],settings['save'],settings['preprocessing'])
    print('Complete:',folder,flush=True)


if __name__=='__main__':main()
