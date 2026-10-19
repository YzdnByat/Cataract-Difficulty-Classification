# Experiment code

This folder contains the implementations used at different stages of the cataract severity and poor-pupil-dilation study. It preserves the original notebook-based workflow alongside the modular Python tools used for CNN hyperparameter search and cross-validation.

## Folder guide

| Folder | Purpose | How experiments are organized |
| --- | --- | --- |
| [notebooks](notebooks/) | Final CNN runs and single-split transformer benchmarks | Individual CNN notebooks by task; one multi-model transformer notebook per task |
| [Optuna](Optuna/ReadMe.md) | CNN hyperparameter optimization followed by final training | Python modules, with a historical Kaggle launcher |
| [K Fold](K%20Fold/ReadMe.md) | Grouped cross-validation of fixed settings | CNN Python runner and transformer notebooks in separate folders |

## How the workflows relate

1. The individual CNN notebooks preserve the final model/task settings selected during model development, including fine-tuning scopes, optimizer groups, input size, class sampling, and loss settings.
2. The Optuna workflow searches selected CNN hyperparameters. Its current implementation saves search-selected values and starts a fresh final training run. It does not edit the original notebooks.
3. CNN K-fold reads its retained notebook configurations from [winning_configs.json](K%20Fold/cnn%20models/configs/winning_configs.json), then trains a fresh model for each fold.
4. The two transformer benchmark notebooks compare nine architectures with an explicit starting configuration. They are not Optuna studies.
5. Transformer K-fold evaluates a subset using fixed settings. It can use a prior validation summary to choose candidates and can reuse saved CNN fold assignments.

A configuration called `winning` here means a retained selected configuration in the project artifacts. It does not establish a global optimum, and there is no automatic chain that transfers the current Optuna study output into all other experiments.

## Tasks

| Task argument | Prediction | Zero-based label order |
| --- | --- | --- |
| `severity` | Cataract severity, four classes | Low, dense, mature, brunescent |
| `poor_dilation` | Poor pupil dilation, binary | Normal, poor dilation |

The command-line runners accept `--task severity` or `--task poor_dilation`. Metadata column formats vary by workflow; prepare the inputs described by the destination folder rather than renaming columns without checking the loader.

## Execution and persistence

| Workflow | Intended environment | GPU behavior | Restart behavior |
| --- | --- | --- | --- |
| Original CNN notebooks | Google Colab with Drive | One model on one selected device | Best weights only; no full interrupted-epoch training resume |
| Transformer benchmark notebooks | Kaggle T4 x2 | Two isolated model workers | Full state saved after each epoch; completed models skipped |
| Optuna | Kaggle or configured Python environment | One selected device per process, sequential models | No persistent-study or interrupted-training resume |
| CNN K-fold | Kaggle or configured Python environment | One selected device per process, sequential model/fold jobs | `--skip-existing` skips completed folds; incomplete folds restart |
| Transformer K-fold | Kaggle GPU environment | Up to two independent model/fold workers; one GPU supported | Last completed epoch resumes; completed folds skipped |

Two GPU workers train independent jobs. Their memory is not combined into a larger device for one model. Changing physical batch size changes the experiment and must be recorded.

## Before training

Read the workflow README, install its dependencies, obtain the required original-ID inputs, and verify all labels and group assignments. Grouping uses video IDs. Keep run settings and split manifests with the outputs; a seed or a filename by itself is insufficient to identify a historical experiment.

Folder names and capitalization are retained for compatibility. Paths containing spaces must be quoted in terminal commands. The transformer benchmark path `poor pupil dialtion task` retains its existing spelling.

[Project overview](../README.md)
