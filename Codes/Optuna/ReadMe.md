# CNN hyperparameter optimization with Optuna

This folder contains the Python search-and-training pipeline for cataract severity and poor-pupil-dilation CNN experiments. It searches learning rates, weight decay, and label smoothing, then builds a fresh pretrained model for final training and held-out test evaluation.

It is separate from the retained [final CNN notebooks](../notebooks/cnn/) and [CNN K-fold runner](../K%20Fold/cnn%20models/ReadMe.md). Its selected settings are saved as run artifacts; they do not automatically replace those other workflows' configurations.

## Models and tasks

| Item | Available values |
| --- | --- |
| `--task` | `severity`, `poor_dilation`, or `all` |
| `--model` | `resnet18`, `resnet34`, `resnet50`, `densenet121`, `densenet169`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5` |
| Model selections | One architecture, `all`, `run1`, or `run2` |

There is no VGG or transformer search in this implementation. `run1` contains ResNet-18/34/50, DenseNet-121/169, and EfficientNet-B1; `run2` contains EfficientNet-B2/B3/B4/B5.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Orchestrate data loading, HPO, fresh final training, reports, and summary |
| `config.py` | Paths, task encodings, class quotas, model sizes, and search/training budgets |
| `metadata_utils.py` | Canonical/original column conversion, label validation, duplicate checks, image resolution |
| `dataset.py` | Datasets, video-grouped splits, class-quota sampling, and loaders |
| `models.py` | Pretrained architectures, fixed unfreezing scopes, optimizer groups |
| `optimize.py` | Optuna objective, TPE sampler, median pruning |
| `train.py` | Epoch loops, early stopping, best weights, diagnostics, generated report |
| `losses.py`, `metrics.py`, `transforms.py` | Task losses, evaluation, and preprocessing |
| `Kaggle_Optuna.ipynb` | Historical launcher for a separately supplied package and metadata snapshots |
| [START_HERE.md](START_HERE.md) | Setup routes and launcher prerequisites |
| `requirements-kaggle.txt` | Additional dependencies for a runtime with working torch/torchvision |
| `analyze_datset.py` | Auxiliary dataset audit; uses an old local Windows path that must be edited before use |

The misspelling `analyze_datset.py` is the actual filename. This auxiliary script is not part of the training launch path.

## Input contract

CSV metadata can use `file_name` / `video_id`, or the original `filename` / `videoname` aliases. Supply either a zero-based numeric `label` column or all one-hot columns for the task:

| Task | Label order / one-hot columns |
| --- | --- |
| Severity | 0 low, 1 dense, 2 mature, 3 brunescent; columns `low nuclear density`, `dense`, `mature`, `brunescent` |
| Poor dilation | 0 normal, 1 poor; columns `normal`, `poor dilation` |

If both numeric and one-hot labels exist, they must agree. Exact repeated image/video records with identical labels are collapsed with a warning; conflicting records are rejected. Identifiers are read as text. This loader reads CSV, not XLSX.

The image resolver accepts `ROOT/video_id/file_name`, `ROOT/file_name`, and the same layouts below `ROOT/images/`. It rejects missing or ambiguous matches. Set the image root after extracting your archive and inspect the video-grouped split/class coverage before search.

## Direct Python route

Run from the repository root in a Bash-compatible terminal, replacing the input/output paths with your own:

```bash
cd "Codes/Optuna"
python -m pip install -r requirements-kaggle.txt
export SEVERITY_CSV_PATH="/path/to/severity_metadata.csv"
export SEVERITY_IMG_DIR="/path/to/severity/image_root"
export OPTUNA_OUTPUT_DIR="/path/to/outputs/optuna-severity-run1"
python main.py --task severity --model resnet18
```

For dilation, set `DILATION_CSV_PATH`, `DILATION_IMG_DIR`, and a separate `OPTUNA_OUTPUT_DIR`, then run `python main.py --task poor_dilation --model resnet18`. These variables are read when `config.py` is imported, so configure them before starting the Python process.

The additional requirements file does not install torch/torchvision. Provide a compatible pair first; Kaggle normally already has one. The unmodified historical notebook additionally uses `hashlib.file_digest`, requiring Python 3.11 or newer. [START_HERE.md](START_HERE.md) shows a Kaggle route that uses the checked-in Python files without relying on the missing package.

`--skip-optuna` uses hard-coded baseline values from `main.py`. It does not load a previous search winner and does not resume an interrupted trial.

## Search definition

| Hyperparameter | Search range | Sampling |
| --- | --- | --- |
| Classifier LR (`lr_head`) | 1e-4 to 1e-2 | Log scale |
| Backbone LR (`lr_backbone`) | 5e-6 to 1e-4 | Log scale |
| Weight decay | 1e-5 to 1e-3 | Log scale |
| Label smoothing | 0.00 to 0.10 | Linear scale |

The search uses `TPESampler(seed=42)`, a maximizing study, and `MedianPruner(n_startup_trials=3, n_warmup_steps=2)`. Defaults are **12 trials per task/model**, at most **8 epochs per trial**, with validation macro-F1 reported each epoch. A completed trial returns macro-F1 from its final trial epoch, rather than its best intermediate epoch.

Model resolution, physical batch size, fine-tuning depth, sampler quotas, and augmentation are fixed by the code; Optuna does not search them. Model input sizes and batches are:

| Model | Input side | Batch |
| --- | --- | --- |
| ResNet-18 / ResNet-34 | 224 | 64 |
| ResNet-50 | 224 | 32 |
| DenseNet-121 | 224 | 32 |
| DenseNet-169 | 224 | 24 |
| EfficientNet-B1 | 240 | 48 |
| EfficientNet-B2 | 288 | 32 |
| EfficientNet-B3 | 300 | 24 |
| EfficientNet-B4 | 380 | 12 |
| EfficientNet-B5 | 456 | 8 |

These sizes/batches are specific to this Optuna implementation and differ from several final-notebook settings.

## Data split and final training

Two `GroupShuffleSplit` stages target approximately 70/15/15 by video ID with seed 42. This is video isolation, not proof of independent patients across different video IDs. The splitter is not a stratified K-fold splitter.

Only the training loader applies class quotas and random augmentation. Quotas are severity `[550, 700, 450, 150]` and dilation `[1000, 1000]`; sampling uses replacement only when a requested quota exceeds available training frames.

After search, a fresh pretrained model is trained with the chosen values for up to **50 epochs**. AdamW uses separate backbone/head groups; `ReduceLROnPlateau` follows validation loss. The best checkpoint and early stopping follow validation macro-F1, with patience 10 and minimum improvement 1e-4. The selected checkpoint is evaluated on the test partition.

The split and TPE sampler have fixed seeds, but this code does not comprehensively seed Python, NumPy, and torch training randomness. Search reruns are not guaranteed to reproduce identical scores or winners.

## Outputs and limitations

Each `<model>_<task>/` directory saves `best_hyperparameters.json`, `best_model.pth`, `training_log.txt`, `training_curves.png`, `confusion_matrix.png`, `per_class_metrics.png`, and a generated `README.md` with settings and test/class-wise results. The output root also receives `benchmark_summary.csv`.

The original Kaggle launcher additionally saves `metadata_used.csv`, train/validation/test split CSVs, `environment.txt`, and `input_manifest.json`. Direct `main.py` runs do not automatically export all of those provenance files. They also do not export a persistent Optuna database, the full trial table, machine-readable per-frame test predictions, or a complete final-training resume checkpoint.

Studies are in memory and model/task jobs are sequential on one selected device. The launcher masks to GPU 0; two-GPU scheduling is not implemented here. The notebook refuses existing model output directories. Direct `main.py` has no equivalent completed-run guard, so use a fresh output directory to avoid overwriting artifacts or appending duplicate summary rows.

`main.py` catches per-experiment failures and continues, so a process exit alone is insufficient evidence of success. Confirm the generated model report and corresponding summary row. Interrupted models need a new output/run name and a fresh search/training run.

[Code overview](../ReadMe.md)
