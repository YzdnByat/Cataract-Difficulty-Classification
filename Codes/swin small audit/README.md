# Swin-Small training project

This is a **configuration-driven Python training project**: Python modules contain the implementation, and YAML files contain the task, paths, training settings and saving options. It trains only Swin-Small for severity or poor pupil dilation. There are no notebooks in this package.

Version 1.2 adds the requested aspect-preserving padding and efficiency improvements. The original labels, split construction, model class, augmentation ranges, sampler, losses, optimization steps and LR schedule are retained. Only `best.pth` is saved, selected by validation macro-F1; early stopping still monitors validation loss. Fourteen original function/class bodies remain unchanged. Exact changes and original reference hashes are recorded under `provenance/`.

## Run

Use Python 3.10 or newer with a matched CUDA-enabled PyTorch/torchvision pair. Kaggle already supplies that pair. Keep it installed and install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Run from the extracted `swin_small_training` directory:

```bash
python train.py --config configs/severity.yaml
python train.py --config configs/poor_dilation.yaml
```

Each command trains one task on one visible GPU, matching how each original notebook worker trained an individual model. The original two-GPU launcher trained different models concurrently; this package does not introduce data parallel training. To run the two tasks simultaneously, use separate terminals with `--gpu 0` for severity and `--gpu 1` for poor dilation. Separate task run names are already configured.

In a Kaggle code cell, after extracting this project under `/kaggle/working`, use:

```python
%cd /kaggle/working/swin_small_training
%pip install -r requirements.txt
!python train.py --config configs/severity.yaml --output-dir /kaggle/working/outputs --gpu 0
```

For the other task:

```python
!python train.py --config configs/poor_dilation.yaml --output-dir /kaggle/working/outputs --gpu 0
```

Enable a Kaggle GPU accelerator and Internet for the original Drive data sources and pretrained weights. Files are saved under `/kaggle/working`; save a Kaggle version or download them before the session is discarded. Run the code from a writable directory, rather than directly from a read-only `/kaggle/input` mount.

## Data paths

By default the YAML data paths are `null`. This invokes the same Drive download logic and file IDs as the supplied notebooks. To use local files, set `data.images_zip` and `data.metadata` in the corresponding YAML, or pass overrides:

```bash
python train.py --config configs/severity.yaml --images-zip /path/Dataset_Severity_V2.zip --metadata /path/Dataset_Severity_V2.xlsx
python train.py --config configs/poor_dilation.yaml --images-zip /path/Dataset_poordilation.zip --metadata /path/dataset_poordialtion.csv
```

Use the original metadata expected by the notebooks. Severity expects the `Dataset_Severity_V2` XLSX sheet; poor dilation expects the original one-hot CSV. The smaller mapped/cleaned Excel files are not replacements for those inputs. The notebook's existing duplicate-row removal, expected counts, checksum checks, image-read checks and missing-file failures remain intact. This conversion does not repair metadata or silently filter unavailable images.

The project-source poor-dilation image ZIP checked during conversion is missing five frames from `PH_0025_S1_0012`, assigned to validation by the original split. Their filenames and the checked archive hash are recorded in `provenance/input_archive_check.json`. If that ZIP is used, the unchanged loader will stop on a missing file. Use the complete original image archive to run that task; the conversion does not remove these rows or change their split.

Relative paths written inside YAML resolve from the YAML file's directory. Relative paths passed on the command line resolve from the current terminal directory. Default run outputs are in this project's `outputs/` directory and downloaded/extracted data in `data_cache/`. `--output-dir` and `--cache-dir` can place these in a different writable location.

Validate and save the original split before using a GPU:

```bash
python train.py --config configs/severity.yaml --prepare-only
```

## Training settings

The training defaults are retained from the two notebooks, with the requested checkpoint policy and padding changes:

| Setting | Value |
|---|---|
| Model | `swin_small_patch4_window7_224.ms_in22k_ft_in1k`, timm 1.0.22; explicit tag pins the original default weights |
| Initialization | Pretrained backbone; new task-specific linear head |
| Head | Four outputs for severity; one logit for binary poor dilation |
| Feature L2 normalization | Enabled |
| Fine-tuning | All backbone layers |
| Split | Original stratified video-group 70/15/15 procedure, seed 42 |
| Maximum epochs / patience | 30 / 5 |
| Batch size / loader workers | 32 / 2 |
| Optimizer | AdamW, weight decay 0.01 |
| Head LR / backbone LR | 1e-4 / 1e-5 |
| Scheduler | CosineAnnealingLR, `T_max=epochs`, stepped after each epoch |
| Loss | Unweighted cross-entropy / BCEWithLogitsLoss |
| Sampling | Original inverse-class-frequency replacement sampler on training frames |
| AMP | Original CUDA autocast and GradScaler |
| Best checkpoint | Highest validation macro-F1; exact ties retain the earlier epoch |
| Early stopping | Original validation-loss rule, patience 5 |

These notebooks use one train/validation/test split, not five-fold cross-validation. Grouping is by the original `video_id` exactly as in the notebooks. No new alias grouping, source balancing, clinical relabelling or patient-identity verification is introduced.

## Preprocessing and padding

Training image operations, in order:

1. Convert to RGB.
2. Bicubic resize preserving aspect ratio, then center on a 224 x 224 square with colour-blended padding.
3. Horizontal flip with probability 0.5.
4. Vertical flip with probability 0.3.
5. Random rotation from -15 to +15 degrees, using the current image's mean RGB for newly exposed pixels.
6. Random affine translation of up to 10% per axis, with zero affine rotation; current image mean RGB fills newly exposed pixels.
7. ToTensor.
8. Normalize with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

Validation/test use RGB conversion, the identical resize-and-padding operation, ToTensor and normalization. No random augmentation is applied. The rotation and translation keep their original nearest-neighbour interpolation and random parameter draws; only their fill colour changes. No brightness/contrast jitter, whole-image blur, cropping, extra augmented samples or new loss weighting is introduced.

The previous direct resize retained the frame boundaries but stretched rectangular anatomy. The new operation fits the full frame inside the square, maintaining its proportions within integer-pixel rounding. For example, 1280 x 720 becomes 224 x 126 with 49 pixels of padding above and below. The actual resized frame is pasted unchanged into the result.

The default `padding.method: edge_blend` improves on constant mean-colour padding:

1. Compute the original frame's mean RGB colour (used only to colour padding).
2. Start each padding band at the adjacent image-edge colour, avoiding an abrupt boundary.
3. Fade smoothly through a blurred edge extension toward the global image mean. The transition spans at most 10% of the 224-pixel canvas, compressed for narrower bands.
4. Apply the blur and blending only outside the resized frame. No anatomy is reflected into the pads.

A one-pixel padding band simply matches its adjacent edge. Set `padding.method: mean` in YAML to use constant image-mean padding closer to the original ResNet idea; choose a new `run.name` when changing it. Padding is calculated at the final resolution to avoid a large full-resolution square allocation. This resize-then-pad implementation is not pixel-identical to the ResNet notebook's full-resolution pad-then-resize operation.

Padding preserves frame geometry; downsampling still removes fine detail, and the existing training rotation/translation can move some content outside the canvas. Padding also preserves differences in aspect ratio and any source-specific content already present, so it does not establish that the model is free of source bias.

### Why 224 and ImageNet normalization?

The pinned pretrained checkpoint specifies a fixed 224 x 224 input, bicubic interpolation, mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. These are checked against the loaded model's pretrained configuration before training. The checkpoint was pretrained on ImageNet-22k and fine-tuned on ImageNet-1k. Its [official config](https://huggingface.co/timm/swin_small_patch4_window7_224.ms_in22k_ft_in1k/blob/main/config.json) and [model card](https://huggingface.co/timm/swin_small_patch4_window7_224.ms_in22k_ft_in1k) describe those settings.

ImageNet normalization is the matching starting point for these weights. There is no validation evidence here that another normalization improves this task. A dataset-specific alternative would need an experiment with statistics calculated on training data only. Per-image RGB means used for padding do not replace tensor normalization or alter the clinical image's colours. Tensor normalization remains `(RGB / 255 - mean) / std`.

We deliberately use the full-frame padding transform instead of the model card's evaluation center crop. Increasing resolution is a separate experiment involving model settings and compute; a larger input is not automatically better, and the current configuration rejects incompatible sizes.

## Efficiency and reproducibility changes

- Cache path and integer-label tuples in the dataset, avoiding a pandas row allocation for every sample.
- Use the existing pinned-memory loaders with non-blocking device transfers. Numeric training operations and AMP settings are unchanged.
- Calculate padding on a 224-pixel canvas, using Pillow's channel statistics instead of a full-resolution floating-point RGB array.
- Import download dependencies only when a download is needed. Local-file preparation works without `gdown` or `requests` imports.
- Record image dimensions while checking readability, then reuse them for the padding audit.
- Share `FitPad` between loaders and test-image exports; regenerate PNG views atomically so stale stretched views cannot survive.
- Include preprocessing, its implementation hash, Pillow/NumPy versions, exact pretrained tag and torchvision version in the saved model protocol. Resume and analysis reject incompatible preprocessing. Default names end in `_padded_v3` so earlier outputs stay intact.

These changes reduce avoidable work; no GPU speedup or accuracy gain is claimed without benchmarking/training.

## Saved files

Each run has an `effective_config.yaml`, the requested YAML, a protected implementation fingerprint and a copy of the executing source. Each task directory contains:

| File | Contents |
|---|---|
| `split_manifest.csv` | Every frame, original video ID, label, path, image hash, dimensions and assigned split |
| `video_split_assignments.csv` | One row per original video ID and its train/val/test assignment |
| `train_frames.csv`, `val_frames.csv`, `test_frames.csv` | Complete frame lists for each split |
| `train_videos.csv`, `val_videos.csv`, `test_videos.csv` | Separate lists of video assignments |
| `split_class_counts.csv`, `label_map.json` | Class distributions and class-number meaning |
| `preprocessing.json` | Full padding/normalization specification, checkpoint identity and fingerprint |
| `preprocessing_frames.csv` | Per-frame original/resized dimensions, padding on each side and padding fraction |
| `preprocessing_summary.csv` | Mean/max padding fraction by split and class |
| `preprocessing_previews/` | Before/after padding examples and index, one training frame per class; largest padding fraction selected deterministically |
| `data_sources.json`, `worker_settings.json` | Original data/source hashes and runtime settings |
| `logs/data_preparation.log`, `logs/training.log` | Preparation output and full worker stdout/stderr, appended across resumes |
| `logs/launcher_errors.log` | Launcher exceptions, if any |
| `summary.csv`, `run_status.json` | Final metrics summary and run status |

The task's `Swin-Small/` directory contains:

| File/directory | Contents |
|---|---|
| `best.pth` | Single macro-F1-selected checkpoint with model, optimizer, scheduler, scaler, RNG and history at that selected epoch |
| `history.csv` | Original notebook history, including losses, accuracies and actual per-epoch head/backbone LRs |
| `epoch_metrics.csv` | Extra macro/weighted F1 and epoch-pass timings |
| `epoch_predictions/` | Every epoch's validation frame probabilities and sampled training prediction stream |
| `learning_curves.png` | Loss, accuracy, extra macro-F1 and LR curves |
| `learning_rates.csv`, `learning_rate_curve.png` | Both learning-rate series |
| `metrics.json` | Original validation/test metric definitions evaluated at the macro-F1-selected checkpoint, plus selection details |
| `val_predictions.csv`, `test_predictions.csv` | Original per-frame labels, predictions and probabilities |
| `analysis/val/`, `analysis/test/` | Extended predictions, classification reports, confusion matrices, confusion-pair summaries, ROC data/plots and per-video error summaries |
| `model_architecture.txt`, `parameter_inventory.csv` | Model structure and parameter names/shapes/trainability |
| `protocol.json`, `pretrained_config.json`, `preprocessing.json`, `trainable_parameters.txt` | Training protocol, exact pretrained configuration, transform specification and trainable parameters |
| `environment.json`, `execution.json` | Package/GPU/CUDA details and invocation timing |
| `complete.json`, `training_progress.json`, `analysis_complete.json` | Original completion marker and separate observation/export status |
| `artifacts.csv` | Relative paths and sizes of saved model/report/image artifacts |

The epoch observer receives predictions already produced by the training loop; it does not run extra model forwards or iterate loaders again. **Training sampler outputs are stored with `draw_index`, not false image names.** Training metrics reflect the sampled, augmented stream. Validation/test frame identities are preserved and checked against their manifest.

Only one .pth file is retained: best.pth. It is atomically replaced when validation macro-F1 strictly improves. No last.pth or per-epoch weight snapshots are written. It includes full resume state at the selected best epoch. On interruption, later unsaved epochs must be repeated. Per-epoch CSV metrics/predictions are still saved. The three save switches control prediction/image exports only.

## Inspect individual test frames

Every test frame is copied byte-for-byte into:

```text
Swin-Small/test_images/originals/true_1_dense/pred_2_mature/<video_id>/<original_filename>
```

The same layout includes correct predictions and every other reference/predicted class pair. Class numbers for severity are 0=low, 1=dense, 2=mature, 3=brunescent; poor dilation is 0=normal, 1=poor dilation. The copied originals are checked against their manifest hashes.

`test_images/model_input_rgb/` contains the exact deterministic padded 224 x 224 RGB view before tensor normalization. The PNG pixels are produced by the same `FitPad` callable as the dataset. The normalization specification/fingerprint is also saved in `test_images/preprocessing.json`. Normalized tensors have channel values outside 0..1 and cannot be represented faithfully as ordinary RGB PNGs; use the saved formula to reconstruct them. `test_images/index.csv` links originals and model views to the prediction, all class probabilities, confidence, entropy and review fields.

Open `test_gallery.html` in a browser **while retaining the accompanying `test_images` directory**. Filter reference=Dense and prediction=Mature, inspect errors/correct cases, search video names and sort by confidence. The gallery defaults to errors. It works offline using relative image links.

`analysis/test/misclassified_frames.csv` lists all mistakes. `confusion_pairs.csv` ranks problem pairs. `video_error_summary.csv` identifies videos contributing multiple wrong frames. No clinical reason is automatically invented. Use the blank `reviewed_reason` and `reviewer_notes` columns for evidence-based review. Those fields survive report regeneration when the same image identity remains present.

ROC plots use the saved class probabilities: positive poor dilation for binary classification, and one-vs-rest curves for severity. The severity macro ROC-AUC is an extra post-training statistic, saved separately from the notebook's original `metrics.json`.

## Resume, regenerate reports and archive everything

Repeat the same training command with the same run name and complete output directory. Resume restores best.pth at its selected epoch, including optimizer/scheduler/scaler/RNG/history. Later unsaved epochs are repeated. Completed training is skipped and reporting is rerun. A changed training config, preprocessing, data split, protected source implementation or worker protocol requires a new run name. Recorded software versions identify environment differences; this does not guarantee bitwise-identical runs across environments.

Regenerate reports after export interruption without training or model inference:

```bash
python train.py --config configs/severity.yaml --analyze-only
```

Keep the source images accessible at the manifest paths, or retain the already-copied test originals, when rebuilding image exports.

Archive the complete run, including **best.pth**, logs, source/config snapshots and copied test frames:

```bash
python archive_results.py --run-dir outputs/swin_small_severity_padded_v3
python archive_results.py --run-dir outputs/swin_small_poor_dilation_padded_v3
```

The default archive is written next to the run directory with no file exclusions apart from transient writes/lock files. For a new Kaggle session, extract the **complete run archive** back into the configured output directory and make the data/cache paths available again before resuming. A `.pth` file alone does not contain the manifests, completion markers or copied reports. Archive a running job only after stopping it cleanly, so files are not changing while copied.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests verify the unchanged original function/class ASTs, the sole non-blocking-transfer change in `epoch_pass`, best-only macro-F1 selection, YAML defaults, grouped splits, duplicate-byte rejection, prediction coverage, binary/four-class reports, exact original/test-view copies, review-field retention and observer behaviour. Padding tests cover landscape/portrait/square/odd/extreme sizes, exact preservation of the resized content, boundary continuity, mean-colour convergence, preprocessing protocol mismatches and stale-view regeneration. PyTorch tests additionally check normalized tensors, sampler weights, augmentation RNG draws, observer equivalence and real Swin-Small forwards with both task heads (random initialization, no weight downloads) when the required dependencies are installed.

See `provenance/verification.json` for the final test counts and environment. No GPU training or pretrained-weight loading was executed while creating this code package.

The severity preparation was also run on the supplied real image archive and metadata. Its frame/video split checksum matches the historical result exactly: 1,795/385/361 train/validation/test frames. Applying the preserved split procedure to the historical poor-dilation manifest likewise reproduces 2,116/495/470 frames and its original split checksum. See `provenance/verification.json` for the checks and limitations.

Use the new default `_padded_v3` run names. Earlier output directories and checkpoints are not modified or deleted. Checkpoint selection uses validation only; test metrics do not select an epoch. The field `selected_val_loss` is the loss at the best-F1 epoch; `minimum_val_loss_seen` separately records the lowest validation loss observed.
