"""One task and one isolated GPU worker per invocation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime,timezone
import pandas as pd
import yaml
from .run_lock import RunLock,LockBusy
from .config import load_config
from . import data_io
from .experiments import DEFAULT_CONFIG, PRESETS, prepare_preset
from .io_utils import require, save_json

PROJECT_ROOT=Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:stream.flush()


def implementation_snapshot(run_dir):
    protected=['data_io.py','preprocessing.py','transforms.py','model.py','engine.py',
               'color_constancy.py','experiments.py']
    hashes={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in protected}
    path=run_dir/'implementation.json'
    if path.exists() and json.loads(path.read_text())['protected_module_sha256']!=hashes:
        raise ValueError('Training/data implementation changed. Choose a new run.name.')
    save_json(path,{'protected_module_sha256':hashes})
    source=run_dir/'source';source.mkdir(exist_ok=True)
    shutil.copytree(PROJECT_ROOT/'cataract_swin',source/'cataract_swin',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(PROJECT_ROOT/'provenance',source/'provenance',dirs_exist_ok=True)
    for name in ['train.py','experiments.py','requirements.txt','README.md']:
        if (PROJECT_ROOT/name).is_file():shutil.copy2(PROJECT_ROOT/name,source/name)


def save_video_assignments(df,classes,task_dir):
    rows=[]
    for vid,g in df.groupby('video_id',sort=True):
        counts=g.label.value_counts()
        entry={'video_id':vid,'split':g.split.iloc[0],'frames':len(g),
               'labels_present':'; '.join(classes[i] for i in sorted(g.label.unique())),
               'identity_unit':'original video_id from notebook metadata'}
        for i,c in enumerate(classes):entry[f'class_{i}_frames']=int(counts.get(i,0))
        rows.append(entry)
    table=pd.DataFrame(rows)
    table.to_csv(task_dir/'video_split_assignments.csv',index=False)
    df.groupby(['split','label']).agg(frames=('file_name','size'),videos=('video_id','nunique')).reset_index().to_csv(task_dir/'split_class_counts.csv',index=False)
    for split in ['train','val','test']:
        df[df.split==split].to_csv(task_dir/f'{split}_frames.csv',index=False)
        table[table.split==split].to_csv(task_dir/f'{split}_videos.csv',index=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Swin-Small: best-only validation macro-F1 checkpoint with full artifact export.')
    parser.add_argument('--config',help='YAML config. The severity comparison defaults apply with --preset.')
    parser.add_argument('--preset',choices=PRESETS,help='Severity comparison experiment.')
    parser.add_argument('--data-dir',help='Shared severity data prepared by experiments.py prepare.')
    parser.add_argument('--images-zip');parser.add_argument('--metadata')
    parser.add_argument('--output-dir');parser.add_argument('--cache-dir');parser.add_argument('--run-name');parser.add_argument('--gpu')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true',help='Validate images and save the exact notebook split without importing torch.')
    mode.add_argument('--analyze-only',action='store_true',help='Regenerate reports from saved predictions; does not train or load weights.')
    args=parser.parse_args(argv)
    if not args.config and not args.preset:
        parser.error('Pass --config for a standalone run or --preset for a comparison.')
    if args.data_dir and not args.preset:
        parser.error('--data-dir requires --preset.')
    if args.preset and not args.analyze_only and not args.data_dir:
        parser.error('--preset requires --data-dir from experiments.py prepare.')
    if args.preset and (args.images_zip or args.metadata):
        parser.error('Supply --images-zip and --metadata to experiments.py prepare for a preset.')
    cfg,sources,config_path=load_config(args.config or DEFAULT_CONFIG,images_zip=args.images_zip,metadata=args.metadata,
        output_dir=args.output_dir,cache_dir=args.cache_dir,run_name=args.run_name or args.preset,gpu=args.gpu)
    if args.preset:
        require(cfg['task']=='severity' and cfg['preprocessing']['input_size']==448,
                'Comparison presets require the 448-pixel severity configuration.')
        preset=PRESETS[args.preset]
        sources['severity']['classes']=[sources['severity']['classes'][label] for label in preset.labels]
        cfg['preprocessing'].pop('color_constancy',None)
        if preset.color_constancy!='none':cfg['preprocessing']['color_constancy']=preset.color_constancy
    elif cfg['preprocessing'].get('color_constancy','none')!='none':
        parser.error('Select a color-constancy preset to use source-aware preprocessing.')
    run_dir=Path(cfg['run']['output_dir'])/cfg['run']['name'];run_dir.mkdir(parents=True,exist_ok=True)
    task=cfg['task'];task_dir=run_dir/task;task_dir.mkdir(exist_ok=True)
    log_dir=task_dir/'logs';log_dir.mkdir(exist_ok=True)
    try:
        with RunLock(run_dir/'run.lock'):
            experiment_path=task_dir/'experiment.json'
            if experiment_path.exists():
                require(json.loads(experiment_path.read_text())['preset']==args.preset,
                        'Use the original --preset for this run or choose a new --run-name.')
            if args.analyze_only:
                if args.preset:
                    require(experiment_path.is_file(), 'No saved preset run at this output path.')
                from .reporting import export_analysis
                previous=run_dir/'effective_config.yaml'
                if not previous.is_file():
                    raise ValueError('No saved run configuration. Run preparation/training first.')
                old=yaml.safe_load(previous.read_text())
                if old.get('preprocessing')!=cfg['preprocessing'] or old['task']!=task:
                    raise ValueError('Report preprocessing/task differs from the saved run. Use its original configuration.')
                df=pd.read_csv(task_dir/'split_manifest.csv')
                export_analysis(task_dir/'Swin-Small',df,sources[task]['classes'],cfg['save'],cfg['preprocessing'])
                return 0
            implementation_snapshot(run_dir)
            previous=run_dir/'effective_config.yaml'
            if previous.exists():
                old=yaml.safe_load(previous.read_text())
                if old['training']!=cfg['training'] or old['task']!=task or old.get('preprocessing')!=cfg['preprocessing']:
                    raise ValueError('Training configuration/task/preprocessing changed. Choose a new run.name.')
            previous.write_text(yaml.safe_dump(cfg,sort_keys=False))
            shutil.copy2(config_path,run_dir/'requested_config.yaml')
            save_json(task_dir/'run_status.json',{'status':'preparing','task':task})
            with (log_dir/'data_preparation.log').open('a',encoding='utf-8',buffering=1) as log:
                with redirect_stdout(Tee(sys.stdout,log)):
                    if args.preset:
                        df=prepare_preset(args.data_dir,args.preset,task_dir)
                    else:
                        data_io.configure(cache_dir=cfg['run']['cache_dir'],output_root=run_dir,sources=sources,
                            local_files={t:{'zip':cfg['data']['images_zip'] if t==task else None,
                                            'metadata':cfg['data']['metadata'] if t==task else None} for t in sources},
                            training=cfg['training'])
                        df=data_io.prepare_task(task)
            save_video_assignments(df,sources[task]['classes'],task_dir)
            from .preprocessing_audit import save_preprocessing_audit
            save_preprocessing_audit(df,sources[task]['classes'],task_dir,cfg['preprocessing'])
            save_json(task_dir/'label_map.json',{str(i):c for i,c in enumerate(sources[task]['classes'])})
            settings={'config':cfg['training'],'sources':sources,'task':task,'output_root':str(run_dir),
                      'manifest':str(task_dir/'split_manifest.csv'),'save':cfg['save'],'preprocessing':cfg['preprocessing']}
            save_json(task_dir/'worker_settings.json',settings)
            if args.prepare_only:
                save_json(task_dir/'run_status.json',{'status':'prepared','task':task,'frames':len(df)})
                print('Prepared:',task_dir,flush=True);return 0
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=cfg['run']['gpu'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            env['PYTHONPATH']=str(PROJECT_ROOT)+os.pathsep+env.get('PYTHONPATH','')
            command=[sys.executable,'-m','cataract_swin.worker','--settings',str(task_dir/'worker_settings.json')]
            save_json(task_dir/'run_status.json',{'status':'training','task':task,'gpu':cfg['run']['gpu']})
            print('Output:',task_dir,'\nWorker log:',log_dir/'training.log',flush=True)
            with (log_dir/'training.log').open('a',encoding='utf-8',buffering=1) as log:
                log.write('\nSESSION '+datetime.now(timezone.utc).isoformat()+'\n')
                process=subprocess.Popen(command,cwd=PROJECT_ROOT,env=env,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,
                    start_new_session=(os.name=='posix'))
                try:
                    for line in process.stdout:
                        print(line,end='',flush=True);log.write(line)
                    code=process.wait()
                except BaseException:
                    if process.poll() is None:
                        if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
                        else:process.terminate()
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                            else:process.kill()
                            process.wait()
                    raise
            save_json(task_dir/'run_status.json',{'status':'complete' if code==0 else 'failed','task':task,'exit_code':code})
            return code
    except LockBusy:
        raise RuntimeError('Another process is already using this run directory.')
    except BaseException as exc:
        with (log_dir/'launcher_errors.log').open('a',encoding='utf-8') as f:traceback.print_exc(file=f)
        save_json(task_dir/'run_status.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                                             'task':task,'error':str(exc)})
        raise
