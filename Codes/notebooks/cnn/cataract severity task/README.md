# Cataract severity: final CNN notebooks

One notebook trains one CNN using the retained final settings for the `severity` task. These are single-split final runs, not Optuna studies or K-fold notebooks.

## Notebook inventory

| Notebook | Architecture key | Active input size |
| --- | --- | --- |
| [Final_DenseNet_121.ipynb](Final_DenseNet_121.ipynb) | `densenet121` | 224 × 224 |
| [Final_DenseNet_169.ipynb](Final_DenseNet_169.ipynb) | `densenet169` | 224 × 224 |
| [Final_EFF_B1.ipynb](Final_EFF_B1.ipynb) | `efficientnet_b1` | 240 × 240 |
| [Final_EFF_B2.ipynb](Final_EFF_B2.ipynb) | `efficientnet_b2` | 260 × 260 |
| [Final_EFF_B4.ipynb](Final_EFF_B4.ipynb) | `efficientnet_b4` | 380 × 380 |
| [Final_EFF_B5.ipynb](Final_EFF_B5.ipynb) | `efficientnet_b5` | 456 × 456 |
| [Final_Eff_B3.ipynb](Final_Eff_B3.ipynb) | `efficientnet_b3` | 260 × 260 |
| [Final_ResNet_18.ipynb](Final_ResNet_18.ipynb) | `resnet18` | 224 × 224 |
| [Final_ResNet_34.ipynb](Final_ResNet_34.ipynb) | `resnet34` | 224 × 224 |
| [Final_ResNet_50.ipynb](Final_ResNet_50.ipynb) | `resnet50` | 224 × 224 |
| [Final_VGG_16.ipynb](Final_VGG_16.ipynb) | `vgg16` | 224 × 224 |

Input sizes reflect the active preprocessing cells, not generic ImageNet recommendations or debug preview code. Model-specific learning rates and unfrozen layers remain in each notebook; equivalent retained settings are listed in the [CNN K-fold configuration file](../../../K%20Fold/cnn%20models/configs/winning_configs.json).

## Labels and inputs

| Model index | Meaning | Original one-hot field |
| --- | --- | --- |
| 0 | Low nuclear density | `low nuclear density` → `low` |
| 1 | Dense | `dense` |
| 2 | Mature | `mature` |
| 3 | Brunescent | `brunescent` |

The original Colab defaults point to `Dataset_Severity_V2.zip` and `Dataset_Severity_V2.csv` under `/content/drive/MyDrive/`. Update `ZIP_PATH` and `CSV_PATH` for your files. The CSV must contain `filename`, `videoname`, `low nuclear density`, `dense`, `mature`, and `brunescent`. Class columns encode a single class per frame. An XLSX file is not read directly by these notebooks; export the appropriate label sheet to CSV while retaining identifiers and one-hot columns.

Images are located using `IMAGE_ROOT/videoname/filename`. Check the resolved root printed after extraction. The notebook's split groups use `videoname`, which is a video identifier; patient independence requires additional identity information.

## Training protocol

| Setting | Implementation |
| --- | --- |
| Partitioning | Approximately 70/15/15 by video groups, two `GroupShuffleSplit` stages, seed 42 |
| Model | Pretrained CNN; selected late layers and classifier fine-tuned |
| Physical batch size | 64 |
| Duration | 30 epochs; no early stopping in the final training loop |
| Class sampler | Fixed quotas with replacement: 1,200 low, 1,200 dense, 1,200 mature, and 400 brunescent samples per epoch (4,000 total) |
| Loss | Class-weighted cross-entropy, weights `[1, 1, 1, 2]`, label smoothing 0.05 |
| Optimizer | AdamW, with model-specific parameter groups |
| Scheduler | `ReduceLROnPlateau`, minimum validation loss, factor 0.5, patience 3; explicit minimum LR 1e-6 for ResNet/DenseNet/VGG, omitted for EfficientNet |
| Checkpoint | Lowest validation loss |
| Final evaluation | Reload selected weights and evaluate on the test partition |

Sampling quotas describe training draws, including repeated frames; they are not the size or class distribution of the original dataset. Training augmentation is stochastic, with mean-color square padding, resizing, geometric and photometric transforms, and ImageNet normalization. Validation and test preprocessing omit random augmentation.

## Run and preserve the result

Open one notebook in Colab, enable a GPU, mount Drive, set the input paths, and run the cells sequentially. Check the no-overlap assertions and printed distributions before training. The root dependency list is a starting list, not a locked historical Colab environment.

Each notebook currently uses `SAVE_PATH = "efficientnet_b3_best.pth"`, including architectures whose name differs from that filename. Set a unique checkpoint destination or isolate each run. This path controls both saving and reloading within the notebook; the filename alone does not identify the architecture.

The evaluation cell prints accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1/support, and displays count and normalized confusion matrices. Loss curves and augmentation previews are displayed in notebook output. The best weights are saved, but structured per-frame predictions, split CSVs, and full optimizer resume state are not automatically exported.

Preserve the executed notebook and checkpoint together. Split seed 42 does not seed every training operation in these original notebooks. For per-fold JSON/CSV outputs, use the [CNN K-fold runner](../../../K%20Fold/cnn%20models/ReadMe.md).

[CNN overview](../README.md)
