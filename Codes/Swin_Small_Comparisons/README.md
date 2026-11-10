# Swin-Small and ResNet50 severity comparisons

Both models use the same severity experiment definitions, data pipeline and training loop. Only the backbone, its pretrained weights and the width of its pooled features differ within a preset. Both heads use L2-normalized pooled features followed by a linear classifier.

| Model | Pinned timm backbone and weights |
|---|---|
| Swin-Small | `swin_small_patch4_window7_224.ms_in22k_ft_in1k` |
| ResNet50 | `resnet50.tv2_in1k` (TorchVision ImageNet-1K V2 weights) |

These weights use different pretraining recipes. The experiment compares the two pretrained models under a matched fine-tuning protocol.

## Experiments

| Preset | Training cohort | Validation and test cohort | Color correction | Checkpoint |
|---|---|---|---|---|
| `baseline` | All sources, classes 0–3 | All sources, classes 0–3 | None | Max validation macro-F1 |
| `farabi3` | Farabi, classes 0–2 | Farabi, classes 0–2 | None | Max validation macro-F1 |
| `pooled3` | Farabi plus 181 Noor class-0 frames, classes 0–2 | Farabi, classes 0–2 | None | Max validation macro-F1 |
| `gray_world` | All sources, classes 0–3 | All sources, classes 0–3 | Gray World on Noor | Max validation macro-F1 |
| `white_patch` | All sources, classes 0–3 | All sources, classes 0–3 | White Patch on Noor | Max validation macro-F1 |
| `gray_world_val_loss` | All sources, classes 0–3 | All sources, classes 0–3 | Gray World on Noor | Min validation loss |
| `farabi_youtube` | Farabi and YouTube, classes 0–3 | Farabi and YouTube, classes 0–3 | None | Max validation macro-F1 |

The label order is 0 low nuclear density, 1 dense, 2 mature, and 3 brunescent. Source comes from the existing metadata rule: `S1` is Farabi, `S2` is Noor, and `Brunescent_*` is YouTube/external. Gray World and White Patch operate on decoded Noor RGB before resize and padding in every split that includes Noor. The original dataset's Noor frames already reach 255 per channel, so White Patch is a no-op on this archive.

Preparation validates the original severity XLSX and image ZIP, then creates one shared split at seed 42. It yields 1,795 training, 385 validation and 361 test frames. Every preset filters that manifest after splitting. The split uses the historical metadata `video_id` for comparability with prior runs. Some YouTube groups share a source recording, including across validation and test, so this split cannot establish independent-video generalization for Brunescent.

## Shared recipe

- Proportional bicubic resize and Edge-Blend padding to `448 × 448`, the existing Albumentations train transforms, and ImageNet normalization. Evaluation uses the same deterministic preprocessing in both models.
- One weighted draw per training frame per epoch, with replacement and weight `1 / training class count`. For paired runs, each epoch's sampler and augmentation seeds come from a data generator independent of model initialization and dropout. Both backbones receive the same draws and augmentation for the same preset, epoch and worker settings. This stream differs from the historical global RNG stream.
- Batch size 8, two data workers, up to 30 epochs, all backbone layers trainable, and mixed precision during CUDA training. Cross-entropy, AdamW with weight decay 0.01, learning rates `1e-5` for the backbone and `1e-4` for the head, and cosine annealing.
- Early stopping monitors validation loss with patience 5 for every preset. `best.pth` holds only the checkpoint selected by that preset. Resume starts from that checkpoint and repeats later unsaved epochs.

## Run on Kaggle

Upload the contents of this project to `Codes/Swin_Small_Comparisons/` in the GitHub repository. Import `Swin_ResNet50_Comparisons_Kaggle.ipynb`. Enable Internet and select GPU T4 ×2 for `GPU = "0,1"`. The notebook clones the project, prepares the severity dataset once, then runs each selected preset for both models in sequence. Each training job uses both GPUs with a total batch size of 8.

The default is all 14 model/preset pairs. Set `PRESETS = ["baseline"]` to run the baseline pair, or select a subset for one session. Download the archives from `/kaggle/working/swin-resnet50-comparisons/zips/`. The suite writes each archive as soon as that action completes. `action_status.json` records completed and failed actions.

## Run from the terminal

Install a matched CUDA-enabled PyTorch and torchvision pair, then run these commands from this folder.

```bash
python -m pip install -r requirements.txt

python experiments.py prepare --data-dir ./severity-data \
  --images-zip /path/to/Dataset_Severity_V2.zip \
  --metadata /path/to/Dataset_Severity_V2_1.xlsx

python experiments.py run --data-dir ./severity-data --output-dir ./comparison-output --gpu 0,1
```

The suite accepts `--models Swin-Small ResNet50` and `--presets baseline gray_world` to select runs. For a single model/preset pair:

```bash
python train.py --model ResNet50 --preset baseline --data-dir ./severity-data --output-dir ./runs --gpu 0
python train.py --model Swin-Small --preset baseline --data-dir ./severity-data --output-dir ./runs --gpu 0
```

Append `--prepare-only` to either command to validate its split and config on CPU. The original `--config` entry point still accepts the standalone severity and poor dilation configurations. Presets belong to the severity task.

For Drive inputs, pass `--images-drive-id` and `--metadata-drive-id` to `prepare` in place of the local paths. To compare runs from separate sessions, unzip the archives and pass each run folder to `experiments.py compare --runs-dir <folder> --runs-dir <folder> --output-dir <report>`.

## Outputs

The suite saves `runs/<model>__<preset>/` and `zips/<model>__<preset>.zip`. Each run includes its effective config, environment, shared and filtered split manifests, video assignments, training logs, learning curves, best checkpoint, prediction probabilities, ROC, confusion matrices and both original and model-input test images. The suite also archives the Noor color preview and comparison report.

- `comparison_metrics.csv` contains accuracy and macro-F1 on matched cohorts. `shared_farabi_0_2` uses the same 304 Farabi frames for all presets, with scores restricted to classes 0–2. It also reports native accuracy and native class-3 calls.
- `full_4_class` contains the common 361 test frames for the four-class presets that include Noor. `shared_farabi_youtube_0_3` uses the same 320 frames to compare four-class runs with and without Noor.
- `comparison_deltas.csv` compares presets within a model. `comparison_backbone_deltas.csv` compares ResNet50 minus Swin-Small for the same preset and cohort, in percentage points.
