"""Compare validation-selected checkpoints on explicitly matched test cohorts."""
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import MODEL_REGISTRY
from .experiments import PRESETS, SHARED_COLUMNS, SPLIT_COLUMNS, select_preset
from .io_utils import require, table_hash
from .reporting import load_predictions
from .tracking import write_csv

FARABI = 'shared_farabi_0_2'
FULL = 'full_4_class'
FARABI_YOUTUBE = 'shared_farabi_youtube_0_3'
NOOR = 'noor_class_0'
YOUTUBE = 'youtube_class_3'


def find_experiments(roots):
    paths = set()
    for value in roots:
        root = Path(value).expanduser().resolve()
        require(root.is_dir(), f'Run path does not exist: {root}')
        direct = root / 'severity' / 'experiment.json'
        paths.update([direct] if direct.is_file() else root.glob('*/severity/experiment.json'))
    return sorted(paths)


def metric_row(frame, preset, cohort, class_count, best_epoch):
    evaluation = frame.prediction
    labels = list(range(class_count))
    if cohort == FARABI:
        labels = [0, 1, 2]
        evaluation = frame[['prob_0', 'prob_1', 'prob_2']].to_numpy().argmax(axis=1)
    return {
        'preset': preset, 'cohort': cohort, 'head_classes': class_count,
        'frames': len(frame), 'videos': frame.video_id.nunique(), 'best_epoch': best_epoch,
        'accuracy': accuracy_score(frame.label, evaluation),
        'macro_f1': f1_score(frame.label, evaluation, labels=labels, average='macro', zero_division=0)
                    if cohort not in (NOOR, YOUTUBE) else None,
        'native_accuracy': accuracy_score(frame.label, frame.prediction),
        'native_class_3_calls': int(frame.prediction.eq(3).sum()),
    }


def cohort_deltas(left, right):
    for cohort in left.index.intersection(right.index):
        a, b = left.loc[cohort], right.loc[cohort]
        require(a.frames == b.frames and a.videos == b.videos, f'Unmatched cohort {cohort}.')
        yield {
            'cohort': cohort, 'frames': int(a.frames), 'videos': int(a.videos),
            'accuracy_delta_pp': 100 * (b.accuracy - a.accuracy),
            'macro_f1_delta_pp': 100 * (b.macro_f1 - a.macro_f1),
            'native_accuracy_delta_pp': 100 * (b.native_accuracy - a.native_accuracy),
        }


def compare_runs(roots, output_dir):
    summaries, matched = [], []
    common = None
    seen = set()
    for path in find_experiments(roots):
        task_dir = path.parent
        status_path = task_dir / 'run_status.json'
        if not status_path.exists() or json.loads(status_path.read_text())['status'] != 'complete':
            print(f'Skipping incomplete run {task_dir.parent.name}.')
            continue
        spec = json.loads(path.read_text())
        name = spec['preset']
        require(name in PRESETS, f'Unknown preset {name!r}.')
        preset = PRESETS[name]
        master = pd.read_csv(task_dir / 'shared_split_manifest.csv')
        split_hash = table_hash(master, SHARED_COLUMNS)
        require(split_hash == spec['shared_split_hash'], f'Shared split hash mismatch in {name}.')
        frame = select_preset(master, name)
        require(table_hash(frame, SHARED_COLUMNS) == spec['run_split_hash'],
                f'Preset membership mismatch in {name}.')
        model_folders = [task_dir / model for model in MODEL_REGISTRY
                         if (task_dir / model / 'protocol.json').is_file()]
        require(len(model_folders) == 1, f'Expected one model in {task_dir}.')
        folder = model_folders[0]
        model = folder.name
        require((model, name) not in seen,
                f'Expected one completed run for {model} / {name}. Choose the specific run folders.')
        protocol = json.loads((folder / 'protocol.json').read_text())
        require(protocol['model'] == model and protocol['timm_name'] == MODEL_REGISTRY[model],
                f'Backbone identity mismatch in {folder}.')
        require(protocol['split_hash'] == table_hash(frame, SPLIT_COLUMNS),
                f'Trained split hash mismatch in {name}.')
        classes = protocol['classes']
        source_record = json.loads((task_dir / 'data_sources.json').read_text())
        class_names = source_record['source_specification']['classes']
        require(classes == [class_names[i] for i in preset.labels],
                f'Class names or ordering differ from the preset in {name}.')
        preprocessing = protocol['preprocessing']['configuration'].copy()
        require(preprocessing.pop('color_constancy', 'none') == preset.color_constancy,
                f'Color method mismatch in {name}.')
        config = protocol['config'].copy()
        selection = protocol.get('selection_metric', 'val_macro_f1')
        require(selection == preset.selection_metric and config.pop('selection_metric', selection) == selection,
                f'Checkpoint selection rule mismatch in {name}.')
        settings = (split_hash, config, preprocessing, protocol['preprocessing']['training_augmentation'],
                    protocol.get('data_sampling_seed'), protocol.get('cuda_device_count'))
        if common is None:
            common = settings
        require(settings == common, 'Comparison requires the same shared split, recipe, data seed and GPU count.')
        best_epoch = json.loads((folder / 'metrics.json').read_text())['best_epoch']
        predictions = load_predictions(folder / 'test_predictions.csv', frame, classes)
        farabi = predictions.loc[predictions.source.eq('Farabi') & predictions.label.lt(3)].copy()
        require(len(farabi) > 0, f'Missing Farabi test cohort in {name}.')
        cohorts = {FARABI: farabi}
        farabi['predicted_closed_3'] = farabi[['prob_0', 'prob_1', 'prob_2']].to_numpy().argmax(axis=1)
        farabi['preset'] = name
        farabi['model'] = model
        matched.append(farabi[['model', 'preset', 'video_id', 'file_name', 'label', 'prediction', 'predicted_closed_3']])
        if len(classes) == 4:
            if len(predictions) == master.split.eq('test').sum():
                cohorts[FULL] = predictions
            cohorts[FARABI_YOUTUBE] = predictions.loc[predictions.source.isin(('Farabi', 'YouTube_external'))]
            youtube = predictions.loc[predictions.source.eq('YouTube_external')]
            if len(youtube):
                require(youtube.label.eq(3).all(), 'The YouTube class-3 cohort contains another class.')
                cohorts[YOUTUBE] = youtube
            noor = predictions.loc[predictions.source.eq('Noor')]
            if len(noor):
                require(noor.label.eq(0).all(), 'The Noor class-0 cohort contains another class.')
                cohorts[NOOR] = noor
        for cohort, cohort_frames in cohorts.items():
            summaries.append({
                'model': model, 'selection_metric': selection,
                **metric_row(cohort_frames, name, cohort, len(classes), best_epoch),
            })
        seen.add((model, name))

    require(summaries, 'No completed severity preset runs were found.')
    summary = pd.DataFrame(summaries).sort_values(['cohort', 'preset', 'model']).reset_index(drop=True)
    deltas = []
    preset_pairs = [('baseline', 'farabi3'), ('farabi3', 'pooled3'), ('baseline', 'gray_world'),
                    ('baseline', 'white_patch'), ('gray_world', 'gray_world_val_loss'),
                    ('baseline', 'farabi_youtube')]
    for model, group in summary.groupby('model'):
        for reference, candidate in preset_pairs:
            left = group.loc[group.preset.eq(reference)].set_index('cohort')
            right = group.loc[group.preset.eq(candidate)].set_index('cohort')
            deltas.extend(dict(model=model, reference=reference, candidate=candidate, **row)
                          for row in cohort_deltas(left, right))
    backbone_deltas = []
    for preset, group in summary.groupby('preset'):
        left = group.loc[group.model.eq('Swin-Small')].set_index('cohort')
        right = group.loc[group.model.eq('ResNet50')].set_index('cohort')
        backbone_deltas.extend(dict(preset=preset, reference_model='Swin-Small', candidate_model='ResNet50', **row)
                               for row in cohort_deltas(left, right))
    output_dir = Path(output_dir)
    write_csv(summary, output_dir / 'comparison_metrics.csv')
    write_csv(pd.concat(matched, ignore_index=True), output_dir / 'shared_farabi_predictions.csv')
    delta_columns = ['cohort', 'frames', 'videos', 'accuracy_delta_pp', 'macro_f1_delta_pp', 'native_accuracy_delta_pp']
    write_csv(pd.DataFrame(deltas, columns=['model', 'reference', 'candidate', *delta_columns]),
              output_dir / 'comparison_deltas.csv')
    write_csv(pd.DataFrame(backbone_deltas, columns=['preset', 'reference_model', 'candidate_model', *delta_columns]),
              output_dir / 'comparison_backbone_deltas.csv')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'Comparison reports: {output_dir}')
    return summary
