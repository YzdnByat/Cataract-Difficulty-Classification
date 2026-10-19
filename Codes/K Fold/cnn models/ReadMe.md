# CNN K-fold evaluation: Python implementation

This is the modular Python implementation of grouped K-fold evaluation for the project's final CNN configurations. Use `run_kfold.py` to train one task at a time, choosing one model, a comma-separated subset, or every available model.

| Task | Configurations | Available architectures |
| --- | --- | --- |
| `severity` | 11 | ResNet-18/34/50, DenseNet-121/169, EfficientNet-B1/B2/B3/B4/B5, VGG-16 |
| `poor_dilation` | 10 | ResNet-18/34/50, DenseNet-121/169, EfficientNet-B1/B2/B3/B4/B5 |

There is no poor-dilation VGG-16 configuration because the corresponding final notebook is not included. Settings are loaded separately for each task/model from [configs/winning_configs.json](configs/winning_configs.json), not selected by a new Optuna run.

## Source files

| File or folder | Responsibility |
| --- | --- |
| `run_kfold.py` | CLI, task/model selection, protocol guards, fold scheduling, aggregation |
| `config_loader.py` | Load, normalize, and validate the retained configurations |
| [configs](configs/) | Model/task-specific final-notebook settings |
| `folds.py` | CSV normalization, group-fold creation, overlap and class checks, audits |
| `dataset.py` | Image resolution, datasets, class-quota sampling, data loaders |
| `transforms.py` | Configuration-driven training and evaluation preprocessing |
| `models.py` | Architectures, trainable scopes, and optimizer parameter groups |
| `losses.py` | Class-weighted / smoothed cross-entropy |
| `kfold_engine.py` | Training, checkpoint evaluation, prediction export, result aggregation |
| `metrics.py` | Accuracy, macro/weighted F1, ROC-AUC, confusion matrices |
| `validate_project.py` | Configuration and optional architecture/optimizer checks |
| [tests](tests/) | Existing configuration and group-fold unit tests |
| `requirements.txt` | Dependencies for this implementation |

## Setup

After cloning the repository, run these commands from its root:

```bash
cd "Codes/K Fold/cnn models"
python -m pip install -r requirements.txt
python validate_project.py
python run_kfold.py --task severity --list-models
python run_kfold.py --task poor_dilation --list-models
```

Use a compatible torch/torchvision build for the target machine. On Kaggle, retain a working preinstalled pair rather than replacing it without a compatibility reason. Commands here are ordinary terminal commands; in notebook code cells use `!` before shell commands and `%cd` for a persistent working-directory change.

## Metadata and images

Supply a CSV containing:

```csv
file_name,video_id,label
frame_001.jpg,video_001,0
```

This row illustrates the schema, not a complete training dataset. Legacy column names `filename` and `videoname` are accepted as aliases. A single `label` column is required; this loader does not derive labels from one-hot columns and does not read XLSX files.

| Task | Accepted numeric labels | Accepted text labels |
| --- | --- | --- |
| Severity | `0, 1, 2, 3` | `low`, `dense`, `mature`, `brunescent` |
| Poor dilation | `0, 1` | `normal`, `poor dilation`, `poor_dilation`, `poordilation` |

All classes must be present. Missing required values and duplicate `(video_id, file_name)` records are rejected. Convert one-hot metadata explicitly using the class order above while preserving row order and identifiers; do not pass one-based severity labels directly.

The dataset loader checks these candidate layouts in order, taking the first existing file:

```text
IMAGE_DIR/file_name
IMAGE_DIR/video_id/file_name
IMAGE_DIR/images/file_name
IMAGE_DIR/images/video_id/file_name
```

Set `--image-dir` to the matching root. Avoid multiple conflicting copies of the same referenced frame within these layouts.

## Audit folds before training

Examples below use illustrative absolute paths. Replace them with your actual CSV and extracted image locations.

```bash
python run_kfold.py --task severity --n-splits 5 --seed 42 \
  --csv-path /path/to/SEV_metadata.csv \
  --image-dir /path/to/severity/images \
  --output-dir /path/to/outputs/kfold --dry-run
```

```bash
python run_kfold.py --task poor_dilation --n-splits 5 --seed 42 \
  --csv-path /path/to/PD_metadata.csv \
  --image-dir /path/to/poor_dilation/images \
  --output-dir /path/to/outputs/kfold --dry-run
```

Review `<output-dir>/<task>/fold_assignments.csv` and `fold_audit.csv`. The runner verifies video isolation and full class coverage in training and validation. Each class must occur in at least `n_splits` distinct video IDs. The dry run validates metadata and folds; it does **not** open or verify every image.

The default seed is **42**. Use the seed and exact saved assignment file associated with the result you intend to reproduce. If historical outputs used seed 11, supply `--seed 11` and compare their manifest rather than silently relying on the default.

## Train selected models

Remove `--dry-run` and select the model set:

```bash
python run_kfold.py --task severity --model resnet18 --n-splits 5 --seed 42 \
  --csv-path /path/to/SEV_metadata.csv \
  --image-dir /path/to/severity/images \
  --output-dir /path/to/outputs/kfold --skip-existing
```

For dilation, replace the task and input paths. `--model all` selects every configuration available for the task; `--model resnet18,resnet34` selects a subset. Add `--fold 0` to run only the first, zero-based fold. The default five-fold budgets are 55 severity jobs and 50 dilation jobs.

The runner processes models and folds sequentially on `cuda` when available, otherwise CPU. It does not automatically use two GPUs. Independent processes can be assigned different devices and non-overlapping model jobs, but concurrent writes to the same model/fold directory are not coordinated by this runner.

## Fixed training settings

The retained configurations preserve task/model-specific input size, unfrozen layers, parameter-group learning rates, weight decay, class weights, augmentations, and fixed sampling quotas. All configurations currently use batch size 64, 30 epochs, cross-entropy with label smoothing 0.05, AdamW, and `ReduceLROnPlateau`. There is no early stopping. Best weights are selected by minimum validation loss.

The training sampler draws quotas with replacement; validation is neither resampled nor randomly augmented. Per-fold seeds add the fold number to the CLI seed. Model/fold checkpoint paths are unique.

`--batch-size-override 16` is available when the configured batch does not fit GPU memory. It changes the physical batch and therefore the experiment; it is recorded in `resolved_config.json`. It is not equivalent to the retained batch-64 run.

## Saved outputs

| Scope | Files |
| --- | --- |
| Task directory | `protocol.json`, `fold_assignments.csv`, `fold_audit.csv`; `failures.json` if jobs fail |
| Model directory | `resolved_config.json`, `fold_metrics.csv`, `out_of_fold_predictions.csv`, `oof_confusion_matrix.png`, `summary_metrics.csv`, `summary.json` |
| `fold_0/` through `fold_4/` | `best_model.pth`, `history.csv`, `validation_predictions.csv`, `metrics.json`, `confusion_matrix.png` |

Directories are `<output-dir>/<task>/<model>/fold_<index>/`. Prediction CSVs retain sample/video/fold identifiers, the true and predicted labels, and class probabilities. `summary_metrics.csv` includes completed-fold means, sample standard deviations (`ddof=1`), and pooled OOF metrics. `summary.json` records completed/expected folds and whether all planned folds finished.

The held-out fold is used for both checkpoint selection and metric reporting. Scores are validation CV results, not an independent outer test. Partial OOF files include only completed folds and cannot be presented as a complete five-fold result. Per-class scalar reports can be computed from exported predictions; this runner does not itself export the transformer's `per_class_metrics.csv` file.

## Restart behavior

`--skip-existing` skips a fold when its `metrics.json` exists. It is a completed-fold skip mechanism, not full epoch resume, and it does not comprehensively verify the accompanying checkpoint and prediction files. Inspect restored outputs before treating a fold as complete.

An interrupted fold restarts training because full optimizer/scheduler/epoch state is not saved. Restore previous outputs into the same directory before continuing. Task and model protocol guards reject changed seeds, fold-assignment hashes, retained configurations, or batch overrides in an existing run. Use a new output directory for a changed experiment.

## Existing verification tools

From this directory:

```bash
python -m unittest discover -s tests -v
python validate_project.py --build-models
```

The tests cover configuration loading, selected task-specific settings, and synthetic group-fold behavior. The optional model check instantiates architectures without pretrained downloads and checks optimizer coverage. These checks do not train the models or verify a clinical dataset.

[K-fold overview](../ReadMe.md)
