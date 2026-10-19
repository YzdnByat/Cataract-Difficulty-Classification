# Transformer single-split benchmarks

This folder contains two self-contained Kaggle notebooks. Each trains and evaluates the same nine transformer architectures for one task on a video-grouped train/validation/test split.

| Folder | Notebook | Task |
| --- | --- | --- |
| [Cataract severity](cataract%20severity%20task/) | `transformer-models-severity.ipynb` | Four-class severity |
| [Poor pupil dilation](poor%20pupil%20dialtion%20task/) | `transformer-models-poor-dilation.ipynb` | Binary poor dilation |

The second folder's existing name is `poor pupil dialtion task`; links intentionally retain that spelling.

## Architectures

| Display name | `timm` identifier |
| --- | --- |
| ViT-B/16 | `vit_base_patch16_224` |
| ViT-B/32 | `vit_base_patch32_224` |
| DeiT-Tiny | `deit_tiny_patch16_224` |
| DeiT-Small | `deit_small_patch16_224` |
| DeiT-Base | `deit_base_patch16_224` |
| DeiT-Small-Distilled | `deit_small_distilled_patch16_224` |
| Swin-Tiny | `swin_tiny_patch4_window7_224` |
| Swin-Small | `swin_small_patch4_window7_224` |
| Swin-Base | `swin_base_patch4_window7_224` |

A pretrained backbone produces pooled features, optional L2 normalization is applied (enabled by default), and a new linear classifier predicts the task labels. Distilled DeiT uses its pretrained distilled representation without training a new teacher.

## Shared training settings

| Setting | Default |
| --- | --- |
| Split | Approximately 70/15/15 by video groups, stratified by group label set, seed 42 |
| Input | 224 × 224 RGB, ImageNet normalization |
| Fine-tuning | Entire backbone and classifier; configurable scope overrides |
| Backbone / head LR | 1e-5 / 1e-4 |
| Optimizer | AdamW, weight decay 0.01 |
| Scheduler | Cosine annealing |
| Batch size / workers | 32 / 2 per model |
| Maximum epochs / early-stopping patience | 30 / 5 |
| Training sampler | Inverse-class-frequency weighted sampling with replacement |
| Severity loss | Cross-entropy |
| Dilation loss | Single-logit `BCEWithLogitsLoss` |
| Best epoch | Minimum validation loss |
| Architecture ranking | Validation macro-F1 among completed runs |

These are explicit benchmark settings, not Optuna-selected winning configurations. The train/validation/test partitions also differ from the original CNN notebook partitions. The held-out test metrics do not select the architecture winner.

## Setup and execution

Use Kaggle with Internet enabled and **GPU T4 x2**. The supplied single-split launcher requires two visible GPUs and starts one model per device. It uses fresh Python worker processes, not data parallelism across both devices for a single model.

Run the notebook's numbered sections in order. The dependency cell installs `timm==1.0.22`, `gdown==5.2.0`, and `openpyxl`, while retaining Kaggle's torch/torchvision pair. Other imported scientific packages must also be available in the runtime. The root dependency file alone does not supply this full environment.

In Section 2, review `RUN_NAME`, `RUN_TRAINING`, `MODEL_NAMES`, `CONFIG`, `TUNE_OVERRIDES`, and `LOCAL_FILES`. `RUN_TRAINING` is currently `True`; set it to `False` for data checking without training. Local overrides accept the original image ZIP and original-format metadata instead of downloading from the source URLs.

The loader validates original identifiers, one-hot labels, expected counts and metadata checksum, resolves image paths, checks image readability, hashes images, and saves the group split. Mixed-label dilation groups remain intact. Video-ID grouping alone does not verify patient identity.

## Saved artifacts and resume

| Scope | Artifacts |
| --- | --- |
| Task | `split_manifest.csv`, `data_sources.json`, `summary.csv`, `best_model.json`, `run_status.json`, `README.txt`, logs, generated worker source/settings |
| Model | `protocol.json`, `pretrained_config.json`, `trainable_parameters.txt`, `history.csv`, `metrics.json`, `val_predictions.csv`, `test_predictions.csv`, `complete.json` |
| Weights / resume | `best.pth` contains the selected model under the `model` key; `last.pth` adds optimizer, scheduler, scaler, RNG, epoch, and history state |

The summary ranks completed models by validation macro-F1. `best_model.json` is provisional until all requested models complete. Scores are fractions in [0, 1]. Confusion matrices are stored as numeric arrays in metrics; this notebook does not automatically export Grad-CAM, t-SNE, or a full set of plot PNGs.

To continue in a new session, restore the complete prior `RUN_NAME` output folder at the same `OUTPUT_ROOT` before execution. Keep the configuration and data unchanged; protocol checks reject incompatible reuse. Completed models are skipped and unfinished models resume from the last saved epoch. The optional final ZIP cell includes partial runs and resume checkpoints; `EXPORT_ZIP` defaults to `False`.

The launcher writes and executes an embedded `WORKER_SOURCE`. Section 2 settings are passed to it, but changes to training/model function definitions also need to be reflected in that source. Reading only the visible function cells is insufficient when modifying the implementation.

[Transformer K-fold notebooks](../../K%20Fold/transformer%20models/) provide repeated-fold evaluation. [Notebook overview](../README.md)
