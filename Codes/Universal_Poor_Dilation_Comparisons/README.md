# Poor pupil dilation backbone comparison

Fine-tune the same 21 backbones as the published universal severity trainer. A frame has one of two reference labels, 0 for normal and 1 for poor dilation. The linear head produces two class logits. Both scores have gradients for class-specific Grad-CAM and Grad-CAM++.

## Dataset and split

The source IDs in `cataract_swin/sources.json` point to the audited Drive image ZIP and CSV. The training notebook also accepts local image ZIP and CSV or XLSX paths. The Excel override reads the linked workbook’s original sheet, named `Dataset_Severity_V2`. The current CSV and XLSX agree after one exact duplicate is collapsed. The older mapped project workbook has different labels.

The 700 original video IDs map to 679 recording groups, including 21 linked pairs. The included `cataract_swin/recording_split.csv` freezes the preflight holdout, generated with seed 42 and 70/15/15 stratification by the label composition of each recording. Each frame keeps its own reference label, including frames from the 57 videos with both labels. The loader validates image bytes, frame identities, class labels and split assignment hashes. Each model uses exactly the same held-out frame identities under a given preset.

| Split | Recording groups | Video IDs | Normal frames | Poor-dilation frames | Total frames |
|---|---:|---:|---:|---:|---:|
| Train | 475 | 490 | 1,779 | 393 | 2,172 |
| Validation | 102 | 106 | 388 | 80 | 468 |
| Test | 102 | 104 | 372 | 69 | 441 |

## Common training settings

| Setting | Default |
|---|---|
| Input | 448 × 448 bicubic aspect fit with edge-blend padding and ImageNet normalization |
| Training augmentation | Random resized crop, shift/rotation, horizontal flip, vertical flip and color jitter. Exact parameters are in the YAML. |
| Fine-tuning | All layers, L2-normalized pooled features, linear head with two outputs |
| Class sampling | With replacement, one training-set length each epoch. Each frame has sampling weight proportional to its training class count raised to −0.5. |
| Loss | Weighted two-class softmax cross-entropy for training. Class weights use the same inverse-square-root exponent and mean normalization. Validation and test use unweighted cross-entropy on the natural distribution. |
| Batch | Global physical microbatch 16, eight per T4, two microbatches per full optimizer update. Loss uses the selected weight sum across the update. The last update can be shorter. |
| Optimization | AdamW, backbone LR 1e−5, head LR 1e−4, weight decay 0.01, betas 0.9/0.999, epsilon 1e−8 |
| Schedule | Epoch-wise cosine to zero over at most 30 epochs |
| Model selection | Highest validation macro-F1. An exact tie keeps the earlier checkpoint. Five epochs without improved validation loss stop training. |
| Execution | PyTorch DataParallel across both selected GPUs, training autocast and gradient scaling, seed 42 |

The baseline sampler uses per-frame weights proportional to 1/√1779 and 1/√393. Both sampling weights and cross-entropy weights are recomputed from training frames for every preset. Training accuracy uses the sampled and augmented stream. Decision labels use the higher of the two softmax probabilities.

## Backbones and presets

Registered names are ResNet18, ResNet34, ResNet50, EfficientNet-B0 through EfficientNet-B5, DenseNet121, DenseNet169, VGG16, Swin-Tiny, Swin-Small, Swin-Base, ViT-B_16, ViT-B_32, DeiT-Tiny, DeiT-Small, DeiT-Base and DeiT-Small-Distilled. The exact pretrained tags are in `cataract_swin/registry.py`.

The default `baseline` uses all three sources. The optional `gray_world`, `white_patch` and `gray_world_val_loss` presets apply source-aware correction to Noor images. The last selects by validation loss. `farabi_only` restricts every split to Farabi. `pooled_farabi_eval` trains on all sources and evaluates on Farabi. `farabi_youtube` uses only Farabi and external YouTube frames. Each retains the original recording assignments and uses both dilation labels.

## Entry points and output

The first training notebook cell controls model selection, presets and common training overrides. The shared configuration is `configs/poor_dilation_universal.yaml`. A command-line run from this directory uses the same entry points.

```bash
python experiments.py prepare --data-dir /tmp/pd-data
python experiments.py run --data-dir /tmp/pd-data --output-dir /kaggle/working/poor_dilation_paper_v1 --models ResNet50 Swin-Small --presets baseline --gpu 0,1
```

Each completed model ZIP contains the selected checkpoint, configuration, source snapshot, preprocessing audit, optimizer and scheduler settings, epoch-level predictions and learning curves. `poor_dilation/split_assignments.xlsx` has the frame and video lists for every split, including original video and recording group IDs. `poor_dilation/<model>/model_report.xlsx` contains the summary, confusion counts, predictions, per-class reports and video error summaries. The archive also includes unmodified test frames, exact padded model inputs and a test-image index.

The suite produces `comparison/comparison.xlsx` and detailed CSV files. The full holdout row is `full_binary`. The binary metrics include poor-dilation sensitivity, specificity, precision, macro-F1, ROC AUC and average precision. Source-restricted comparisons use the same frame identities. After saving a session, completed full ZIPs can be recovered into the same output layout. An interrupted model with retained working files resumes from its saved best checkpoint.

The interpretation notebook accepts full or compact model ZIPs. Set the results folder in its first cell. With `CLASS_IDS = [1]` only true poor-dilation frames receive CAMs. With `ERRORS_ONLY = True` the CAM selection is restricted to misclassified frames. t-SNE always uses the complete test set for the selected preset. Its 2D and 3D projections are fitted separately per model. Classification markers preserve the true class as fill and the predicted class as the outline for errors. Saved frame identities and probabilities allow each point to be traced to its source.
