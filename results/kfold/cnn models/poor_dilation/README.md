# Poor pupil dilation — CNN K-fold results

**10 models × five completed folds = 50 completed runs.** The archived protocol uses seed **42**, `StratifiedGroupKFold`, and `video_id` grouping. Its canonical manifest contains **3,081 frame records** in **700 video-ID groups**.

## Comparison

| Model | Folds | Accuracy, % (mean ± SD) | Macro-F1, % (mean ± SD) | Pooled OOF macro-F1, % |
| --- | --- | --- | --- | --- |
| [ResNet-50](resnet50/) | 5/5 | 93.87 ± 2.68 | 88.87 ± 3.70 | 88.95 |
| [ResNet-34](resnet34/) | 5/5 | 93.54 ± 3.85 | 88.40 ± 4.59 | 87.70 |
| [EfficientNet-B2](efficientnet_b2/) | 5/5 | 93.16 ± 3.62 | 87.96 ± 4.51 | 87.65 |
| [ResNet-18](resnet18/) | 5/5 | 93.05 ± 3.05 | 87.43 ± 3.06 | 87.23 |
| [DenseNet-121](densenet121/) | 5/5 | 91.85 ± 3.85 | 86.11 ± 3.19 | 85.79 |
| [EfficientNet-B1](efficientnet_b1/) | 5/5 | 92.00 ± 3.12 | 86.09 ± 3.23 | 86.03 |
| [DenseNet-169](densenet169/) | 5/5 | 92.06 ± 2.93 | 85.68 ± 3.34 | 85.76 |
| [EfficientNet-B3](efficientnet_b3/) | 5/5 | 89.32 ± 3.14 | 82.19 ± 2.62 | 82.19 |
| [EfficientNet-B4](efficientnet_b4/) | 5/5 | 86.14 ± 2.35 | 75.81 ± 1.67 | 76.21 |
| [EfficientNet-B5](efficientnet_b5/) | 5/5 | 81.82 ± 1.83 | 72.66 ± 3.94 | 73.15 |

The table is ordered descriptively by observed fold-mean macro-F1. SD is the sample standard deviation across folds (`ddof=1`), expressed in percentage points above. Pooled OOF scores are recomputed from all held-out predictions and need not equal fold means. The held-out validation fold selects the minimum-loss checkpoint and supplies its reported scores; these are validation CV results.

## Shared result files

| File | Contents |
| --- | --- |
| [model_comparison.csv](model_comparison.csv) | Full-precision model summary, fold means/SDs, pooled OOF scores and verification status |
| [fold_metrics_all_models.csv](fold_metrics_all_models.csv) | Original per-fold values combined across models |
| [per_class_metrics_all_folds.csv](per_class_metrics_all_folds.csv) | Derived per-class precision, recall, F1 and support for every model/fold |
| [fold_assignments.csv](fold_assignments.csv) | Original, exact sample/video/label/fold assignments |
| [fold_audit.csv](fold_audit.csv) | Original per-fold class counts and video-group counts |
| [protocol.json](protocol.json) | Original splitter, seed and fold-assignment fingerprint |
| [label_mapping.json](label_mapping.json) | Explicit zero-based class order |
| [dataset_summary.json](dataset_summary.json) | Counts derived from this run's manifest |

The duplicated GPU-0 and GPU-1 shared manifests/protocols were checked byte-for-byte and stored once. Their original locations remain recorded in the [file inventory](../files_manifest.csv). Per-model logs are retained under `provenance/launcher_logs/`; other model outputs are organized by task/model rather than execution GPU.

Class order: **normal, poor_dilation**. These are counts of the identifiers actually stored in the archived run. Grouping by video ID does not itself establish independent patient identities across videos. For poor dilation, the unique-record total can differ from the row count of an earlier raw metadata export.

## Figures

Each model folder retains its original pooled and per-fold confusion-matrix PNGs and adds a five-fold training-history figure. Count and row-normalized confusion-matrix CSVs and per-class tables are derived directly from the saved predictions. None of these outputs required retraining.

[All CNN result files](../README.md) · [Verification details](../verification_report.json)
