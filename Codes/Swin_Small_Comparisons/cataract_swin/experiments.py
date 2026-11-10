"""Severity presets built from one audited, video-level split."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import pandas as pd

from . import data_io
from .config import load_config
from .io_utils import file_hash, require, save_json, table_hash
from .run_lock import RunLock
from .tracking import write_csv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs' / 'severity_comparisons.yaml'
SPLIT_SEED = 42
SPLIT_COLUMNS = ['file_name', 'video_id', 'label', 'image_sha256', 'split']
SHARED_COLUMNS = [*SPLIT_COLUMNS, 'source']


@dataclass(frozen=True)
class Preset:
    labels: tuple[int, ...] = (0, 1, 2, 3)
    train_source: str | None = None
    evaluation_source: str | None = None
    color_constancy: str = 'none'
    allowed_sources: tuple[str, ...] | None = None
    selection_metric: str = 'val_macro_f1'


PRESETS = {
    'baseline': Preset(),
    'farabi3': Preset((0, 1, 2), train_source='Farabi', evaluation_source='Farabi'),
    'pooled3': Preset((0, 1, 2), evaluation_source='Farabi'),
    'gray_world': Preset(color_constancy='gray_world'),
    'white_patch': Preset(color_constancy='white_patch'),
    'gray_world_val_loss': Preset(color_constancy='gray_world', selection_metric='val_loss'),
    'farabi_youtube': Preset(allowed_sources=('Farabi', 'YouTube_external')),
}


def source_of(video_id):
    """Use the same video-ID rule as the preprocessing audit."""
    match = re.search(r'(?:^|_)(S[12])(?:_|$)', str(video_id), re.I)
    if match:
        return {'S1': 'Farabi', 'S2': 'Noor'}[match.group(1).upper()]
    return 'YouTube_external' if str(video_id).lower().startswith('brunescent_') else 'Unmapped'


def write_source_counts(frame, destination):
    counts = frame.groupby(['split', 'source', 'label']).agg(
        frames=('file_name', 'size'), videos=('video_id', 'nunique'),
    ).reset_index()
    write_csv(counts, destination)


def prepare_data(data_dir, *, images_zip=None, metadata=None, images_drive_id=None, metadata_drive_id=None):
    """Verify the full severity dataset and publish the shared split once."""
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    directory = data_dir / 'severity'
    directory.mkdir(exist_ok=True)
    with RunLock(data_dir / 'run.lock'):
        cfg, sources, _ = load_config(DEFAULT_CONFIG, images_zip=images_zip, metadata=metadata)
        source = sources['severity']
        for key, value in [('zip_id', images_drive_id), ('metadata_id', metadata_drive_id)]:
            if value is not None:
                require(re.fullmatch(r'[A-Za-z0-9_-]+', value), f'Expected a Drive file ID for {key}.')
                source[key] = value
        # A different remote input must not silently reuse a previous download.
        download_key = hashlib.sha256(f'{source["zip_id"]}/{source["metadata_id"]}'.encode()).hexdigest()[:16]
        data_io.configure(
            cache_dir=data_dir / 'cache' / download_key, output_root=data_dir, sources=sources,
            local_files={'severity': {'zip': cfg['data']['images_zip'], 'metadata': cfg['data']['metadata']}},
            training=cfg['training'],
        )
        labels_path = data_io.fetch('severity', 'metadata')
        zip_path = data_io.fetch('severity', 'zip')
        frame = data_io.canonical_metadata(labels_path, 'severity')
        frame = data_io.attach_images(frame, data_io.extract_archive(zip_path, 'severity'))
        frame = data_io.assign_splits(frame, SPLIT_SEED)
        frame['source'] = frame.video_id.map(source_of)

        digest = table_hash(frame, SHARED_COLUMNS)
        manifest = directory / 'split_manifest.csv'
        if manifest.exists():
            require(table_hash(pd.read_csv(manifest), SHARED_COLUMNS) == digest,
                    'The shared dataset or split changed. Prepare it in a new --data-dir.')
        write_csv(frame, manifest)
        write_source_counts(frame, directory / 'split_source_class_counts.csv')
        save_json(directory / 'data_sources.json', {
            'source_specification': sources['severity'],
            'archive_input': 'local' if images_zip is not None else 'drive',
            'metadata_input': 'local' if metadata is not None else 'drive',
            'metadata_sha256': file_hash(labels_path),
            'archive_sha256': file_hash(zip_path),
            'split_seed': SPLIT_SEED,
            'shared_split_hash': digest,
        })
    print(f'Prepared {len(frame)} frames from {frame.video_id.nunique()} videos in {directory}.')
    print(frame.groupby(['split', 'source']).size().unstack(fill_value=0).to_string())
    return directory


def read_prepared(data_dir):
    data_dir = Path(data_dir).expanduser().resolve()
    directory = data_dir / 'severity'
    manifest = directory / 'split_manifest.csv'
    reference_path = directory / 'data_sources.json'
    require(manifest.is_file() and reference_path.is_file(),
            f'No prepared severity data at {data_dir}. Run experiments.py prepare first.')
    frame = pd.read_csv(manifest)
    reference = json.loads(reference_path.read_text())
    require(reference['shared_split_hash'] == table_hash(frame, SHARED_COLUMNS),
            'The shared split manifest changed. Prepare a fresh data directory.')
    missing = next((path for path in frame.image_path if not Path(path).is_file()), None)
    require(missing is None, f'Missing extracted frame {missing}. Prepare the data in this session.')
    return frame, reference


def select_preset(master, preset_name):
    """Filter after splitting, so each video's assignment is shared by every preset."""
    preset = PRESETS[preset_name]
    mask = master.label.isin(preset.labels)
    if preset.allowed_sources is not None:
        mask &= master.source.isin(preset.allowed_sources)
    if preset.train_source is not None:
        mask &= master.split.ne('train') | master.source.eq(preset.train_source)
    if preset.evaluation_source is not None:
        mask &= master.split.eq('train') | master.source.eq(preset.evaluation_source)
    selected = master.loc[mask].copy()
    for split in ('train', 'val', 'test'):
        require(set(selected.loc[selected.split.eq(split), 'label']) == set(preset.labels),
                f'{preset_name} is missing a requested class in {split}. Check the shared split and sources.')
    return selected


def prepare_preset(data_dir, preset_name, task_dir):
    master, reference = read_prepared(data_dir)
    frame = select_preset(master, preset_name)
    preset = PRESETS[preset_name]
    specification = {
        'preset': preset_name,
        'label_ids': list(preset.labels),
        'color_constancy': preset.color_constancy,
        'train_source': preset.train_source,
        'evaluation_source': preset.evaluation_source,
        'split_seed': reference['split_seed'],
        'shared_split_hash': reference['shared_split_hash'],
        'run_split_hash': table_hash(frame, SHARED_COLUMNS),
    }
    if preset.allowed_sources is not None:
        specification['allowed_sources'] = list(preset.allowed_sources)
    if preset.selection_metric != 'val_macro_f1':
        specification['selection_metric'] = preset.selection_metric
    protocol = task_dir / 'experiment.json'
    manifest = task_dir / 'split_manifest.csv'
    if protocol.exists():
        require(json.loads(protocol.read_text()) == specification,
                'The experiment or shared split changed. Use a new --run-name.')
    if manifest.exists():
        previous = pd.read_csv(manifest)
        require(set(SHARED_COLUMNS).issubset(previous.columns) and
                table_hash(previous, SHARED_COLUMNS) == specification['run_split_hash'],
                'This run directory contains a different split. Use a new --run-name.')
    write_csv(master[SHARED_COLUMNS], task_dir / 'shared_split_manifest.csv')
    write_csv(frame, manifest)
    write_source_counts(frame, task_dir / 'split_source_class_counts.csv')
    save_json(task_dir / 'data_sources.json', reference)
    save_json(protocol, specification)
    print(preset_name, frame.groupby(['split', 'source']).size().unstack(fill_value=0).to_string())
    return frame
