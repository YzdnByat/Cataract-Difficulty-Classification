# Cataract difficulty assessment: grouped K-fold training

This project runs **one task at a time** while supporting both datasets from the same codebase:

- `severity`: 4 classes, 11 CNNs
- `poor_dilation`: 2 classes, 10 CNNs

Each task/model pair loads the configuration executed in its final Colab notebook. Configurations are not shared across tasks. Poor-dilation VGG16 is intentionally unavailable because no final poor-dilation VGG16 notebook was supplied.

## Cross-validation design

The default is **5-fold `StratifiedGroupKFold`** with `video_id` as the patient/video group. All frames from one video remain in exactly one validation fold. The program aborts if a video crosses folds or if any training/validation fold lacks a class.

Five folds are preferred because both metadata files have enough videos (632 severity and 685 poor-dilation videos). Three folds would reduce computation, but gives a coarser and generally less stable estimate. The rare severity class has only a small number of videos in some validation folds, so always inspect `fold_audit.csv` and report fold variability.

This is grouped cross-validation of configurations selected previously. It is not nested cross-validation, and it does not turn the earlier test set into untouched external validation.

## Notebook-exact behavior retained

- Model-specific input resolution, batch size, unfrozen layers and learning rates
- Model/task-specific optimizer parameter groups and weight decay
- Task-specific class weights and label smoothing
- Fixed per-class sampling quotas, with replacement, on training folds only
- Notebook augmentation on training folds only
- Deterministic preprocessing on validation folds
- 30 epochs, no early stopping
- Best checkpoint selected by minimum validation loss
- `ReduceLROnPlateau` settings from each notebook

The implementation adds deterministic seeds and unique task/model/fold checkpoint paths. These fix reproducibility and overwrite problems without changing the intended training configuration.

## Repository files

```text
.
├── configs/winning_configs.json   # 21 task/model configurations
├── config_loader.py
├── dataset.py
├── folds.py
├── transforms.py
├── models.py
├── losses.py
├── metrics.py
├── kfold_engine.py
├── run_kfold.py
├── validate_project.py
├── requirements.txt
└── tests/
```

## Expected metadata

The CSV must contain:

```text
file_name,video_id,label
```

Older notebook headings `filename` and `videoname` are accepted. Images may be either flat or organized by video. The loader checks these layouts:

```text
IMAGE_DIR/file_name
IMAGE_DIR/video_id/file_name
IMAGE_DIR/images/file_name
IMAGE_DIR/images/video_id/file_name
```

## Kaggle setup

After cloning the repository and downloading/unzipping one or both datasets:

```bash
%cd /kaggle/working/cataract-kfold
!pip install -r requirements.txt
!python validate_project.py
```

Kaggle normally already contains PyTorch. Avoid upgrading it unnecessarily if its installed `torch` and `torchvision` versions are compatible.

## Always audit folds first

Severity:

```bash
!python run_kfold.py \
  --task severity \
  --n-splits 5 \
  --csv-path /kaggle/working/data/Dataset_Severity/SEV_metadata.csv \
  --image-dir /kaggle/working/data/Dataset_Severity/images \
  --output-dir /kaggle/working/outputs/kfold \
  --dry-run
```

Poor dilation:

```bash
!python run_kfold.py \
  --task poor_dilation \
  --n-splits 5 \
  --csv-path /kaggle/working/data/Dataset_Poordilation/PD_metadata.csv \
  --image-dir /kaggle/working/data/Dataset_Poordilation/images \
  --output-dir /kaggle/working/outputs/kfold \
  --dry-run
```

Review each task's `fold_assignments.csv` and `fold_audit.csv` before training.

## Run the tasks separately

### Kaggle version for severity

```bash
!python run_kfold.py \
  --task severity \
  --model all \
  --n-splits 5 \
  --csv-path /kaggle/working/data/Dataset_Severity/SEV_metadata.csv \
  --image-dir /kaggle/working/data/Dataset_Severity/images \
  --output-dir /kaggle/working/outputs/kfold \
  --skip-existing
```

### Separate Kaggle version for poor dilation

```bash
!python run_kfold.py \
  --task poor_dilation \
  --model all \
  --n-splits 5 \
  --csv-path /kaggle/working/data/Dataset_Poordilation/PD_metadata.csv \
  --image-dir /kaggle/working/data/Dataset_Poordilation/images \
  --output-dir /kaggle/working/outputs/kfold \
  --skip-existing
```

The task output directories are separate, so results cannot overwrite one another.

## Runtime warning and recommended scheduling

One complete severity task is 55 training runs; poor dilation is 50. Either task alone may exceed a 12-hour Kaggle commit, particularly EfficientNet-B4/B5. A safer schedule is one model at a time:

```bash
!python run_kfold.py --task severity --model resnet18 --n-splits 5 \
  --csv-path /path/to/SEV_metadata.csv --image-dir /path/to/severity/images \
  --output-dir /kaggle/working/outputs/kfold --skip-existing
```

Or one fold at a time:

```bash
!python run_kfold.py --task poor_dilation --model efficientnet_b5 --fold 0 \
  --n-splits 5 --csv-path /path/to/PD_metadata.csv \
  --image-dir /path/to/poor_dilation/images \
  --output-dir /kaggle/working/outputs/kfold --skip-existing
```

`--model` also accepts a comma-separated subset such as `resnet18,resnet34,resnet50`.

Kaggle does not automatically make `/kaggle/working` from one commit writable in the next. Save each completed output as a notebook version/dataset, attach it to the next run, and restore the existing task directory before using `--skip-existing`.

## Available models

```bash
!python run_kfold.py --task severity --list-models
!python run_kfold.py --task poor_dilation --list-models
```

## Emergency memory override

All notebooks used batch size 64. This is especially demanding for severity EfficientNet-B4 at 380px and B5 at 456px. If the exact run is impossible:

```bash
--batch-size-override 16
```

This changes the experimental protocol because BatchNorm sees a smaller physical batch. Record the override in the paper. The project stores it in `resolved_config.json` and prevents incompatible resumed runs from being mixed.

## Outputs

```text
outputs/kfold/<task>/<model>/
├── resolved_config.json
├── fold_0/
│   ├── best_model.pth
│   ├── history.csv
│   ├── validation_predictions.csv
│   ├── metrics.json
│   └── confusion_matrix.png
├── ...
├── fold_metrics.csv
├── out_of_fold_predictions.csv
├── oof_confusion_matrix.png
├── summary_metrics.csv
└── summary.json
```

Task-level files include `protocol.json`, `fold_assignments.csv`, and `fold_audit.csv`.

## Verification

```bash
python -m unittest discover -s tests -v
python validate_project.py --build-models
```

The second command instantiates each architecture without pretrained downloads and verifies that every trainable parameter belongs to exactly one configured optimizer group.
