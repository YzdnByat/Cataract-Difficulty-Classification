# Final training and single-split benchmarks

This folder contains the project's notebook-based training and evaluation experiments. It is the entry point for inspecting a complete model-development run without using the K-fold Python runner.

| Folder | Contents | Configuration provenance |
| --- | --- | --- |
| [cnn](cnn/) | 21 individual notebooks: 11 severity models and 10 poor-dilation models | Retained final CNN configurations, one architecture/task per notebook |
| [transformer](transformer/) | Two notebooks, one per task, each comparing nine architectures | Explicit shared starting settings with optional fine-tuning overrides; no Optuna search |

The CNN notebooks are the selected-configuration runs, rather than hyperparameter-search notebooks. Each records the architecture-specific choices used for its final experiment. The transformer notebooks perform a multi-model benchmark; their settings must not be described as Optuna-selected winners.

## What these notebooks evaluate

Each experiment uses a single video-grouped train/validation/test partition. The CNN notebooks use two `GroupShuffleSplit` stages. The transformer benchmarks instead stratify groups by their label sets. Both target approximately 70/15/15 group proportions with seed 42, but they do not necessarily assign the same video IDs to each partition.

A checkpoint is selected using validation loss and then evaluated on the test partition. The transformer summary additionally ranks completed architectures by validation macro-F1. Compare scores only with their task, metadata version, partition, and checkpoint rule identified.

## Inputs and outputs

CNN notebooks expect Google Drive image ZIPs and task-specific one-hot CSV metadata. They save best weights and display reports and figures in notebook output. They do not automatically export all metrics and split assignments as a structured experiment directory.

Transformer notebooks accept their original data-source URLs or local ZIP/metadata paths, verify canonical records and image bytes, save split manifests, and export per-model metrics, predictions, checkpoints, and resume state. They are set up to train two independent models concurrently on Kaggle T4 x2.

Run cells in order after configuring paths. Preserve the original executed outputs when archiving a historical run. The folder READMEs describe filenames, settings, and persistence limitations in detail.

For repeated-fold evaluation, use [K Fold](../K%20Fold/ReadMe.md). For CNN parameter searches, use [Optuna](../Optuna/ReadMe.md).

[Code overview](../ReadMe.md)
