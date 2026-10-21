# Cataract severity — Transformer K-fold results

[Back to Transformer results](../) · [All results](../../../README.md)

4 models evaluated on 2,541 frames from 645 video-ID groups. Each model has five completed folds; the recorded split seed is **42**.

## Model comparison

Values are **mean ± sample standard deviation across five folds, in percent**. Rows are ordered by mean macro-F1; the order describes these runs and does not establish statistical significance.

| Model | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | ROC-AUC (%) | Details |
| --- | ---: | ---: | ---: | ---: | --- |
| [Swin-Small](Swin-Small/) | 88.59 ± 1.85 | 89.54 ± 2.61 | 88.60 ± 1.83 | 98.08 ± 0.36 | [Folds](Swin-Small/fold_metrics.csv) · [Confusion matrix](Swin-Small/oof_confusion_matrix.png) · [Curves](Swin-Small/training_curves.png) |
| [Swin-Tiny](Swin-Tiny/) | 87.93 ± 0.46 | 88.64 ± 1.16 | 87.93 ± 0.42 | 98.09 ± 0.28 | [Folds](Swin-Tiny/fold_metrics.csv) · [Confusion matrix](Swin-Tiny/oof_confusion_matrix.png) · [Curves](Swin-Tiny/training_curves.png) |
| [ViT-B/16](ViT-B_16/) | 88.15 ± 1.24 | 88.34 ± 2.99 | 88.14 ± 1.25 | 98.05 ± 0.55 | [Folds](ViT-B_16/fold_metrics.csv) · [Confusion matrix](ViT-B_16/oof_confusion_matrix.png) · [Curves](ViT-B_16/training_curves.png) |
| [DeiT-Small](DeiT-Small/) | 88.35 ± 1.31 | 87.82 ± 2.58 | 88.33 ± 1.29 | 97.85 ± 0.55 | [Folds](DeiT-Small/fold_metrics.csv) · [Confusion matrix](DeiT-Small/oof_confusion_matrix.png) · [Curves](DeiT-Small/training_curves.png) |

[Download comparison data](model_comparison.csv) · [Exact fold assignments](fold_assignments.csv) · [Evaluation and recorded configurations](protocol.json)

## Reading the results

The held-out fold is used to select the checkpoint with minimum validation loss and to report its scores. These are frame-level validation cross-validation results. Splits separate video IDs; the supplied records do not establish distinct patients across video IDs.

CSV scores remain fractions from 0 to 1. For example, `0.90` means 90%. `*_mean` and `*_std` summarize the five folds; `oof_*` combines every held-out prediction once. Pooled scores can differ from the fold mean. SD describes variation across folds and is not a confidence interval.

`low` and `low nuclear density` refer to the same class.

## Classes

| Label index | Class | Frames |
| --- | --- | ---: |
| 0 | low nuclear density | 795 |
| 1 | dense | 1,047 |
| 2 | mature | 614 |
| 3 | brunescent | 85 |

## Files in each model folder

| File | Contents |
| --- | --- |
| `fold_metrics.csv` | Five rows: selected epoch, epochs trained, minimum validation loss, and scores. |
| `per_class_metrics.csv` | Pooled held-out precision, recall, F1, and support for each class. |
| `oof_confusion_matrix.png` | Confusion matrix combining held-out predictions from all five folds. |
| `oof_confusion_matrix_counts.csv` | Exact integer counts behind that matrix; rows are truth, columns are predictions. |
| `training_curves.png` | Training and validation histories for all five folds. |

The figures describe the archived runs. Training accuracy reflects the training sampler and augmentation. Curves may continue after the selected epoch because checkpoint selection uses the lowest validation loss.

## Example: Swin-Small

This model has the highest mean macro-F1 in this group. The matrix combines all five held-out folds.

![Pooled confusion matrix for Swin-Small](Swin-Small/oof_confusion_matrix.png)

## Complete experiment records

The original `Transformer-KFold-Results(1).zip` contains per-fold histories, predictions, logs, and verification records. This folder provides the compact presentation of those results. No training was rerun.
