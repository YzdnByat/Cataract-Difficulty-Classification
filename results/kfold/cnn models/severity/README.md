# Cataract severity — CNN K-fold results

**11 models × five completed folds = 55 completed runs.** The archived protocol uses seed **42**, `StratifiedGroupKFold`, and `video_id` grouping. Its canonical manifest contains **2,541 frame records** in **645 video-ID groups**.

## Comparison

| Model | Folds | Accuracy, % (mean ± SD) | Macro-F1, % (mean ± SD) | Pooled OOF macro-F1, % |
| --- | --- | --- | --- | --- |
| [ResNet-50](resnet50/) | 5/5 | 86.42 ± 2.78 | 86.05 ± 2.07 | 86.07 |
| [EfficientNet-B1](efficientnet_b1/) | 5/5 | 85.47 ± 1.66 | 85.12 ± 1.84 | 85.33 |
| [EfficientNet-B5](efficientnet_b5/) | 5/5 | 85.73 ± 1.03 | 84.97 ± 2.28 | 85.60 |
| [ResNet-34](resnet34/) | 5/5 | 85.24 ± 2.32 | 83.66 ± 2.99 | 84.34 |
| [EfficientNet-B2](efficientnet_b2/) | 5/5 | 84.82 ± 2.36 | 83.53 ± 5.01 | 83.44 |
| [ResNet-18](resnet18/) | 5/5 | 85.87 ± 2.63 | 83.51 ± 0.55 | 84.11 |
| [EfficientNet-B3](efficientnet_b3/) | 5/5 | 83.40 ± 3.38 | 83.00 ± 4.17 | 83.18 |
| [VGG-16](vgg16/) | 5/5 | 82.81 ± 2.26 | 81.60 ± 1.92 | 82.11 |
| [EfficientNet-B4](efficientnet_b4/) | 5/5 | 81.49 ± 2.81 | 81.04 ± 1.48 | 81.26 |
| [DenseNet-169](densenet169/) | 5/5 | 81.56 ± 0.67 | 80.74 ± 1.21 | 80.92 |
| [DenseNet-121](densenet121/) | 5/5 | 80.87 ± 2.93 | 77.89 ± 3.68 | 78.46 |

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

Class order: **low, dense, mature, brunescent**. These are counts of the identifiers actually stored in the archived run. Grouping by video ID does not itself establish independent patient identities across videos. For poor dilation, the unique-record total can differ from the row count of an earlier raw metadata export.

## Figures

Each model folder retains its original pooled and per-fold confusion-matrix PNGs and adds a five-fold training-history figure. Count and row-normalized confusion-matrix CSVs and per-class tables are derived directly from the saved predictions. None of these outputs required retraining.

[All CNN result files](../README.md) · [Verification details](../verification_report.json)
