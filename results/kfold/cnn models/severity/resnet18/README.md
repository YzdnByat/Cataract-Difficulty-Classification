# ResNet-18 — Cataract severity

Five completed video-grouped validation folds, seed **42**. This folder retains the original run outputs and adds reports derived from their saved predictions and histories.

## Results

| Metric | Fold mean ± sample SD | Pooled OOF |
| --- | --- | --- |
| accuracy | 0.8587 ± 0.0263 | 0.8587 |
| macro_f1 | 0.8351 ± 0.0055 | 0.8411 |
| weighted_f1 | 0.8583 ± 0.0254 | 0.8583 |
| roc_auc | 0.9700 ± 0.0099 | 0.9701 |

Scores and SDs above are fractions, not percentages. OOF contains **2,541** frames, each predicted once while its video group was held out. The held-out validation fold also selects the checkpoint; this is validation cross-validation, not an independent outer test.

## Files

| File | Origin and purpose |
| --- | --- |
| [resolved_config.json](resolved_config.json) | Original task/model hyperparameters and execution settings |
| [fold_metrics.csv](fold_metrics.csv) | Original per-fold scores, selected epochs and losses |
| [summary_metrics.csv](summary_metrics.csv), [summary.json](summary.json) | Original fold means/sample SDs, pooled scores and completion |
| [out_of_fold_predictions.csv](out_of_fold_predictions.csv) | Original row/video/fold identifiers, labels, predictions and class probabilities |
| [oof_confusion_matrix.png](oof_confusion_matrix.png) | Original count and row-normalized matrices side by side |
| [oof_confusion_matrix_counts.csv](oof_confusion_matrix_counts.csv) | Derived machine-readable count matrix |
| [oof_confusion_matrix_normalized.csv](oof_confusion_matrix_normalized.csv) | Derived row-normalized matrix |
| [per_class_metrics.csv](per_class_metrics.csv) | Derived pooled precision, recall, F1 and support |
| [per_class_fold_summary.csv](per_class_fold_summary.csv) | Derived mean/sample SD of per-class fold scores |
| [training_curves.png](training_curves.png) | Derived plot of the original five histories, marking minimum-loss epochs |

Each of `fold_0/` through `fold_4/` contains original `history.csv`, `metrics.json`, `validation_predictions.csv`, and `confusion_matrix.png`, plus derived `per_class_metrics.csv`, `confusion_matrix_counts.csv`, and `confusion_matrix_normalized.csv`.

## Interpretation and provenance

Class order: **low, dense, mature, brunescent**. Matrix rows are true labels; columns are predictions. All scalar fold metrics match the saved predictions within the declared verification tolerance. Configuration and split records belong to this archived run. The source archive does not contain trained checkpoints or an authenticated training-code commit identifier.

The original `resolved_config.json` can include historical single-split `reference_run` fields. These are configuration provenance, not this model's K-fold results. Use `fold_metrics.csv` and `summary_metrics.csv` for the K-fold scores.

[Original launcher log](../provenance/launcher_logs/resnet18_all_folds.log) · [Task comparison](../README.md) · [Verification report](../../verification_report.json)
