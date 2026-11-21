# Cataract severity backbone comparison

This project extends the GitHub severity trainer from `Codes/Swin_Small_Comparisons`. It fine-tunes the complete backbone and the same four-output linear head for the paper baseline. Copy this directory into `Codes/Universal_Severity_Comparisons` in your repository. Import the two notebooks from the release's `notebooks` directory into Kaggle. The training notebook also accepts the complete release ZIP as a Kaggle input.

## Cohort and fixed split

The canonical severity CSV contains 2,540 labeled frames from 645 video IDs. The source IDs and canonical label fingerprint are in `cataract_swin/sources.json`. The original two-model trainer stratifies at video level with seed 42 and a 70/15/15 allocation. The paper trainer runs the same splitter and checks a frozen SHA-256 of the sorted (file, video, label, split) table. It checks that every video has one role, rejects identical image bytes crossing roles, and saves the full manifest. Patient identity across distinct video IDs would require a separate linkage table.

| Split | Videos | Low | Dense | Mature | Brunescent | Total frames |
|---|---:|---:|---:|---:|---:|---:|
| Train | 451 | 556 | 720 | 460 | 57 | 1,793 |
| Validation | 97 | 123 | 142 | 90 | 11 | 366 |
| Test | 97 | 116 | 162 | 86 | 17 | 381 |

The 17 test brunescent frames come from two YouTube video IDs. Class, source, and frame clustering matter when interpreting the class-3 results.

## Model inventory

| Family | Registered names |
|---|---|
| ResNet | ResNet18, ResNet34, ResNet50 |
| EfficientNet | EfficientNet-B0, EfficientNet-B1, EfficientNet-B2, EfficientNet-B3, EfficientNet-B4, EfficientNet-B5 |
| DenseNet and VGG | DenseNet121, DenseNet169, VGG16 |
| Swin | Swin-Tiny, Swin-Small, Swin-Base |
| ViT | ViT-B_16, ViT-B_32 |
| DeiT | DeiT-Tiny, DeiT-Small, DeiT-Base, DeiT-Small-Distilled |

Historical evidence for VGG16 comes from the severity CNN fold reports and the repository's Final_VGG_16 notebook. The archived severity transformer summary contains all nine transformer variants. EfficientNet-B0 is included at the author's request. Each model has an explicit ImageNet-1K checkpoint tag in `cataract_swin/registry.py`. The common downstream recipe controls training conditions. The upstream recipes differ, including the original DeiT distillation. The paper Swin-Small uses the ImageNet-1K initialization. Earlier two-model runs used an ImageNet-22K-pretrained Swin-Small fine-tuned on ImageNet-1K.

## Paper protocol

Edit `configs/severity_universal.yaml` or set common training overrides in the first notebook cell. Each model uses full-frame aspect-preserving bicubic fit and edge-blend padding at 448 × 448, then ImageNet normalization. Only training applies the shared crop, affine, horizontal/vertical flips, and color jitter. All models share the same sampler and worker seeds per epoch. There are no per-backbone learning-rate or augmentation overrides.

| Setting | Default |
|---|---|
| Fine-tuning | All layers, optional L2 at the common linear head enabled |
| Effective batch | 32 frames per full optimizer update |
| Physical batch | 16 across two T4s, 8 per replica; accumulate two forwards per full update |
| Optimizer | AdamW, backbone LR 0.00001, head LR 0.0001, weight decay 0.01 |
| Schedule | Epoch-wise cosine annealing to zero over 30 epochs |
| Selection | Best validation macro-F1, earliest exact tie |
| Early stopping | Five epochs without improved validation loss |
| Imbalance | Replacement sampler and weighted training cross-entropy, each with weight proportional to inverse square root of the training class frame count |
| Evaluation | Natural validation/test distributions, unweighted cross-entropy, one prediction per frame |

The sampler and loss share the frequency correction. With class count *n*, the expected number of sampled frames is proportional to √*n*, and the per-frame loss weight is proportional to 1/√*n*. Loss is normalized by the sum of selected weights over the entire optimizer update, including the short final update. For the older inverse-frequency sampler and ordinary cross-entropy use `sampler_power: 1.0` and `loss_power: 0.0`. This reproduces the old balancing rule. The new batch and initialization decisions still apply.

Both T4s process every full microbatch using PyTorch DataParallel. BatchNorm uses replica-local statistics. If GPU memory requires a smaller microbatch, configure it once for the entire comparison and use a new output directory. Keep the effective batch fixed. Peak memory and throughput depend on the backbone. Select model subsets across Kaggle sessions as necessary. Reuse archived results only under a matching protocol.

## Training and artifacts

The notebook selects `baseline` by default. Other presets are `farabi3`, `pooled3`, `gray_world`, `white_patch`, `gray_world_val_loss`, and `farabi_youtube`. Each preset filters the frozen master split and preserves video roles. The model-independent preview remains available through `experiments.py preview`.

The suite writes one complete training ZIP per model and preset under `zips/`. Completed runs retain all source snapshots, configs, optimizer/scheduler/RNG state at the selected epoch, weights, layer/parameter inventory, preprocessing audit, per-epoch logs and predictions, test originals and exact model-input PNGs. CSV, JSON, plots and Excel workbooks are included. The suite keeps incomplete training folders in `in_progress/`. At completion it keeps one archived copy of each run. `action_status.json` records outcomes. Completed, valid ZIPs in the output folder are reused on rerun. Training interrupted within a model restarts from its saved best epoch.

| Artifact inside a training ZIP | Contents |
|---|---|
| severity/split_assignments.xlsx | Counts by split, source and class, plus a video sheet and a frame sheet for each split |
| severity/MODEL/model_report.xlsx | Validation and test predictions, classification reports, confusion counts, video error summaries, epoch metrics and parameter inventory |
| severity/MODEL/protocol.json, optimization.json, pretrained_config.json, environment.json | Full settings, class order and weights, pretrained identity, split hashes and software/hardware |
| severity/MODEL/analysis/ | Validation and test metrics, ROC curves, confusion matrices, frame and video reports |
| severity/MODEL/best.pth, test_predictions.csv, test_images/ | Checkpoint, test probabilities, identity index, original frames and padded RGB exports |

The suite-level `comparison/comparison.xlsx` summarizes selected completed models and matched cohorts. To aggregate runs from different sessions, extract their per-run ZIPs into one parent folder and run `python experiments.py compare --runs-dir PARENT --output-dir COMPARISON`. The command checks common split, config, seed, GPU count and cohort membership.

## Interpretability notebook

Upload individual training ZIPs to a private Kaggle input, or place them inside one ZIP and paste its Google Drive file link in the notebook. Keep the generated per-model filenames when nesting archives. A single ZIP of unpacked run folders also works. The analysis notebook contains its own model and evaluation code and requires no GitHub checkout.

Select the models, presets, CAM target/layers, case filters, t-SNE dimensions, perplexities, and seeds in cell one. Prediction verification uses the saved frame identity, class order and class probabilities. The CAM filters apply only to case maps. Each model's t-SNE fits use all its held-out frames. The paired plots require the same frame IDs, labels, class order and source-image hashes. The notebook writes native CAM arrays, overlay panels, per-frame manifests, error tables, pooled embeddings, 2D and 3D coordinates, 300-DPI PNG/PDF figures and an interactive 3D HTML view. Heatmaps are scaled within each image. Differences in cluster spacing across separate t-SNE fits have no common numeric scale.
