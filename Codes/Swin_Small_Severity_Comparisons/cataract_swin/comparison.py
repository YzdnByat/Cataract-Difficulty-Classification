"""Compare validation-selected checkpoints on explicitly matched test cohorts."""
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .experiments import PRESETS, SHARED_COLUMNS, SPLIT_COLUMNS, select_preset
from .io_utils import require, table_hash
from .reporting import load_predictions
from .tracking import write_csv

FARABI = 'shared_farabi_0_2'
FULL = 'full_4_class'
NOOR = 'noor_class_0'


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
                    if cohort != NOOR else None,
        'native_accuracy': accuracy_score(frame.label, frame.prediction),
        'native_class_3_calls': int(frame.prediction.eq(3).sum()),
    }


def compare_runs(roots, output_dir):
    summaries, matched = [], []
    common = None
    seen = set()
    for path in find_experiments(roots):
        task_dir = path.parent
        status = json.loads((task_dir / 'run_status.json').read_text())
        if status['status'] != 'complete':
            print(f'Skipping {task_dir.parent.name} ({status["status"]}).')
            continue
        spec = json.loads(path.read_text())
        name = spec['preset']
        require(name in PRESETS and name not in seen,
                f'Expected one completed run per preset. Select the specific run folders for {name!r}.')
        preset = PRESETS[name]
        master = pd.read_csv(task_dir / 'shared_split_manifest.csv')
        split_hash = table_hash(master, SHARED_COLUMNS)
        require(split_hash == spec['shared_split_hash'], f'Shared split hash mismatch in {name}.')
        frame = select_preset(master, name)
        require(table_hash(frame, SHARED_COLUMNS) == spec['run_split_hash'],
                f'Preset membership mismatch in {name}.')
        folder = task_dir / 'Swin-Small'
        protocol = json.loads((folder / 'protocol.json').read_text())
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
        # The three-class and color variants have a different head or color rule by design.
        settings = (split_hash, protocol['timm_name'], protocol['config'], preprocessing,
                    protocol['preprocessing']['training_augmentation'])
        if common is None:
            common = settings
        require(settings == common, 'Comparison requires the same shared split, model, training settings and base transforms.')
        best_epoch = json.loads((folder / 'metrics.json').read_text())['best_epoch']
        predictions = load_predictions(folder / 'test_predictions.csv', frame, classes)
        farabi = predictions.loc[predictions.source.eq('Farabi') & predictions.label.lt(3)].copy()
        require(len(farabi) > 0, f'Missing Farabi test cohort in {name}.')
        summaries.append(metric_row(farabi, name, FARABI, len(classes), best_epoch))
        farabi['predicted_closed_3'] = farabi[['prob_0', 'prob_1', 'prob_2']].to_numpy().argmax(axis=1)
        farabi['preset'] = name
        matched.append(farabi[['preset', 'video_id', 'file_name', 'label', 'prediction', 'predicted_closed_3']])
        if len(classes) == 4:
            summaries.append(metric_row(predictions, name, FULL, len(classes), best_epoch))
            noor = predictions.loc[predictions.source.eq('Noor')]
            if len(noor):
                require(noor.label.eq(0).all(), 'The Noor class-0 cohort contains another class.')
                summaries.append(metric_row(noor, name, NOOR, len(classes), best_epoch))
        seen.add(name)

    require(summaries, 'No completed severity preset runs were found.')
    summary = pd.DataFrame(summaries).sort_values(['cohort', 'preset']).reset_index(drop=True)
    deltas = []
    for reference, candidate in [('baseline', 'farabi3'), ('farabi3', 'pooled3'),
                                 ('baseline', 'gray_world'), ('baseline', 'white_patch')]:
        left = summary.loc[summary.preset.eq(reference)].set_index('cohort')
        right = summary.loc[summary.preset.eq(candidate)].set_index('cohort')
        for cohort in left.index.intersection(right.index):
            a, b = left.loc[cohort], right.loc[cohort]
            deltas.append({
                'cohort': cohort, 'reference': reference, 'candidate': candidate,
                'frames': int(a.frames), 'videos': int(a.videos),
                'accuracy_delta_pp': 100 * (b.accuracy - a.accuracy),
                'macro_f1_delta_pp': 100 * (b.macro_f1 - a.macro_f1),
            })
    output_dir = Path(output_dir)
    write_csv(summary, output_dir / 'comparison_metrics.csv')
    write_csv(pd.concat(matched, ignore_index=True), output_dir / 'shared_farabi_predictions.csv')
    delta_columns = ['cohort', 'reference', 'candidate', 'frames', 'videos', 'accuracy_delta_pp', 'macro_f1_delta_pp']
    write_csv(pd.DataFrame(deltas, columns=delta_columns), output_dir / 'comparison_deltas.csv')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'Comparison reports: {output_dir}')
    return summary
