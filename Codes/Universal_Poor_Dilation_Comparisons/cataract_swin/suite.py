"""Train one model on both GPUs, keeping one complete ZIP after each successful job."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from .config import load_config
from .experiments import DEFAULT_CONFIG, ROOT, read_prepared
from .io_utils import file_hash, save_json
from .registry import run_key
from .run_lock import RunLock


def signature(cfg, shared_hash):
    files = [*sorted((ROOT/'cataract_swin').glob('*.py')), ROOT/'requirements.txt']
    payload = [cfg['training'], cfg['preprocessing'], cfg['save'], cfg['run']['gpu'], shared_hash,
               [(p.name, file_hash(p)) for p in files]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def comparison_inputs(archive, destination, model, preset, recipe):
    """Validate completion and read only small comparison tables from the archived run."""
    name = f'{run_key(model)}__{preset}'
    files = ['poor_dilation/run_status.json', 'poor_dilation/experiment.json', 'poor_dilation/shared_split_manifest.csv',
             'poor_dilation/data_sources.json', *(f'poor_dilation/{model}/{f}' for f in
             ('metrics.json', 'protocol.json', 'test_predictions.csv'))]
    with ZipFile(archive) as z:
        saved = json.loads(z.read(f'{name}/suite_recipe.json'))
        if saved != dict(model=model, preset=preset, signature=recipe):
            raise ValueError(f'{archive.name} belongs to another recipe. Use a new output directory.')
        if json.loads(z.read(f'{name}/poor_dilation/run_status.json'))['status'] != 'complete':
            raise ValueError(f'Incomplete archive: {archive}')
        for required in ('best.pth','analysis_complete.json','test_images/index.csv','model_report.xlsx'):
            if z.getinfo(f'{name}/poor_dilation/{model}/{required}').file_size == 0:
                raise ValueError(f'Empty {required} in {archive}')
        for f in files:
            target = destination/name/f
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(f'{name}/{f}'))


def run_suite(data_dir, output_dir, models, presets, gpu):
    cfg, _, _ = load_config(DEFAULT_CONFIG, gpu=gpu)
    _, source = read_prepared(data_dir)
    recipe = signature(cfg, source['shared_split_hash'])
    output = Path(output_dir).resolve()
    zips = output/'zips'
    zips.mkdir(parents=True, exist_ok=True)
    working = output/'in_progress'
    status_path = output/'action_status.json'

    def run(*args):
        subprocess.run([sys.executable, *map(str,args)], cwd=ROOT, check=True)

    with RunLock(output/'suite.lock'), TemporaryDirectory(prefix='poor_dilation-tables-') as tmp:
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        completed, failures = [], []
        for preset in dict.fromkeys(presets):
            for model in dict.fromkeys(models):
                name = f'{run_key(model)}__{preset}'
                archive = zips/f'{name}.zip'
                directory = working/name
                marker = dict(model=model, preset=preset, signature=recipe)
                status[name] = {**marker, 'state':'running'}
                save_json(status_path,status)
                try:
                    if not archive.exists():
                        mark_path = directory/'suite_recipe.json'
                        if directory.exists() and (not mark_path.exists() or json.loads(mark_path.read_text()) != marker):
                            raise ValueError(f'Unrelated or changed work in {directory}. Use a new output directory.')
                        save_json(mark_path, marker)
                        run('train.py', '--preset',preset,'--model',model,'--data-dir',data_dir,
                            '--output-dir',working,'--run-name',name,'--gpu',gpu)
                        run('archive_results.py','--run-dir',directory,'--output',archive)
                    comparison_inputs(archive, Path(tmp), model, preset, recipe)
                    # Only remove this suite's generated working copy after a complete archive exists.
                    if directory.exists() and json.loads((directory/'suite_recipe.json').read_text()) == marker:
                        shutil.rmtree(directory)
                    completed.append(Path(tmp)/name)
                    status[name].update(state='complete', archive=str(archive.relative_to(output)))
                except Exception as exc:
                    failures.append(name)
                    status[name].update(state='failed', error=str(exc))
                    print(f'{name} failed: {exc}',flush=True)
                save_json(status_path,status)
        if completed:
            command = ['experiments.py','compare','--output-dir',output/'comparison']
            for folder in completed:
                command.extend(['--runs-dir',folder])
            run(*command)
    if failures:
        print('Failed runs:', ', '.join(failures))
    return int(bool(failures))
