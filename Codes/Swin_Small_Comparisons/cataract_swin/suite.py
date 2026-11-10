"""Train each requested model/preset pair and archive outputs as they finish."""
from pathlib import Path
import json
import subprocess
import sys

from .config import load_config
from .experiments import DEFAULT_CONFIG, ROOT
from .io_utils import save_json
from .run_lock import RunLock


def run_suite(data_dir, output_dir, models, presets, gpu):
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    runs_dir, zip_dir = output_dir / 'runs', output_dir / 'zips'
    status_path = output_dir / 'action_status.json'
    load_config(DEFAULT_CONFIG, gpu=gpu)  # Check the device argument before launching.
    output_dir.mkdir(parents=True, exist_ok=True)

    def run(*args):
        return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT).returncode

    with RunLock(output_dir / 'suite.lock'):
        status = json.loads(status_path.read_text()) if status_path.exists() else {}

        def action(name, command, *, model=None, preset=None):
            directory = runs_dir / name
            print(f'\nRunning {name}', flush=True)
            entry = {'state': 'running'}
            if model is not None:
                entry.update(model=model, preset=preset)
            status[name] = entry
            save_json(status_path, status)
            code = run(*command)
            entry.update(exit_code=code, state='failed' if code else 'complete')
            if directory.is_dir():
                archive = zip_dir / (f'{name}_failed.zip' if code else f'{name}.zip')
                archive_code = run('archive_results.py', '--run-dir', directory, '--output', archive)
                entry['archive_exit_code'] = archive_code
                if archive_code == 0:
                    entry['archive'] = str(archive.relative_to(output_dir))
                else:
                    entry['state'] = 'archive_failed'
            elif code == 0:
                entry.update(state='archive_failed', error='The run directory is missing.')
            save_json(status_path, status)
            return entry

        preview = action('preview', ('experiments.py', 'preview', '--data-dir', data_dir,
                                     '--output-dir', runs_dir / 'preview'))
        entries = [preview]
        completed = []
        for preset in dict.fromkeys(presets):
            for model in dict.fromkeys(models):
                name = f"{model.lower().replace('-', '_')}__{preset}"
                entry = action(name, (
                    'train.py', '--preset', preset, '--model', model, '--data-dir', data_dir,
                    '--output-dir', runs_dir, '--run-name', name, '--gpu', gpu,
                ), model=model, preset=preset)
                entries.append(entry)
                if entry['exit_code'] == 0:
                    completed.append(runs_dir / name)

        if completed:
            command = ['experiments.py', 'compare', '--output-dir', runs_dir / 'compare']
            for directory in completed:
                command.extend(['--runs-dir', directory])
            entries.append(action('compare', command))
        else:
            status['compare'] = {'state': 'skipped', 'reason': 'No completed training runs'}
            save_json(status_path, status)

    print(f'Archives: {zip_dir}\nStatus: {status_path}', flush=True)
    return int(any(entry['state'] != 'complete' for entry in entries))
