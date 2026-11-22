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

FULL = 'full_binary'
FARABI = 'farabi_binary'
FARABI_YOUTUBE = 'farabi_youtube_binary'

def find_experiments(roots):
    paths = set()
    for value in roots:
        root = Path(value).expanduser().resolve()
        require(root.is_dir(), f'Run path does not exist: {root}')
        direct = root / 'poor_dilation' / 'experiment.json'
        paths.update([direct] if direct.is_file() else root.glob('*/poor_dilation/experiment.json'))
    return sorted(paths)


def metric_row(frame, preset, cohort, class_count, best_epoch):
    from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
    y, pred = frame.label, frame.prediction
    return {
        'preset': preset, 'cohort': cohort, 'head_classes': class_count,
        'frames': len(frame), 'videos': frame.video_id.nunique(), 'groups': frame.group_id.nunique(),
        'best_epoch': best_epoch, 'accuracy': accuracy_score(y, pred),
        'macro_f1': f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0),
        'poor_precision': precision_score(y, pred, zero_division=0),
        'poor_recall': recall_score(y, pred, zero_division=0),
        'specificity': recall_score(y, pred, pos_label=0, zero_division=0),
        'roc_auc': roc_auc_score(y, frame.prob_1) if y.nunique() == 2 else None,
        'average_precision': average_precision_score(y, frame.prob_1) if y.nunique() == 2 else None,
    }


def cohort_deltas(left, right):
    for cohort in left.index.intersection(right.index):
        a, b = left.loc[cohort], right.loc[cohort]
        require(a.frames == b.frames and a.videos == b.videos, f'Unmatched cohort {cohort}.')
        yield {
            'cohort': cohort, 'frames': int(a.frames), 'videos': int(a.videos),
            'accuracy_delta_pp': 100 * (b.accuracy - a.accuracy),
            'macro_f1_delta_pp': 100 * (b.macro_f1 - a.macro_f1),
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
        master_test = master.loc[master.split.eq('test')]
        cohorts = {}
        for cohort, eligible in [(FULL, master_test),
                                 (FARABI, master_test.loc[master_test.source.eq('Farabi')]),
                                 (FARABI_YOUTUBE, master_test.loc[master_test.source.isin(('Farabi', 'YouTube_external'))])]:
            rows = predictions.merge(eligible[['video_id', 'file_name', 'label']],
                                     on=['video_id', 'file_name', 'label'], validate='one_to_one')
            if len(rows) and len(rows) == len(eligible):
                cohorts[cohort] = rows
        matched.append(predictions[['video_id', 'group_id', 'file_name', 'label', 'prediction', 'prob_0', 'prob_1']]
                       .assign(model=model, preset=name))
        for cohort, cohort_frames in cohorts.items():
            summaries.append({
                'model': model, 'selection_metric': selection,
                **metric_row(cohort_frames, name, cohort, len(classes), best_epoch),
            })
        seen.add((model, name))

    require(summaries, 'No completed poor_dilation preset runs were found.')
    summary = pd.DataFrame(summaries).sort_values(['cohort', 'preset', 'model']).reset_index(drop=True)
    deltas = []
    preset_pairs = [('baseline', 'farabi_only'), ('farabi_only', 'pooled_farabi_eval'), ('baseline', 'gray_world'),
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
        reference = 'ResNet50' if 'ResNet50' in set(group.model) else sorted(group.model.unique())[0]
        left = group.loc[group.model.eq(reference)].set_index('cohort')
        for candidate in sorted(set(group.model) - {reference}):
            right = group.loc[group.model.eq(candidate)].set_index('cohort')
            backbone_deltas.extend(dict(preset=preset, reference_model=reference, candidate_model=candidate, **row)
                                   for row in cohort_deltas(left, right))
    output_dir = Path(output_dir)
    write_csv(summary, output_dir / 'comparison_metrics.csv')
    write_csv(pd.concat(matched, ignore_index=True), output_dir / 'test_predictions_all_models.csv')
    delta_columns = ['cohort', 'frames', 'videos', 'accuracy_delta_pp', 'macro_f1_delta_pp']
    write_csv(pd.DataFrame(deltas, columns=['model', 'reference', 'candidate', *delta_columns]),
              output_dir / 'comparison_deltas.csv')
    write_csv(pd.DataFrame(backbone_deltas, columns=['preset', 'reference_model', 'candidate_model', *delta_columns]),
              output_dir / 'comparison_backbone_deltas.csv')
    summary.to_excel(output_dir / 'comparison.xlsx', sheet_name='test_cohorts', index=False)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'Comparison reports: {output_dir}')
    return summary
