# Individual CNN final runs

These are the project's model-specific CNN training and evaluation notebooks. Each notebook retains the final configuration used for one architecture and one task, including its input resolution, fine-tuning scopes, optimizer parameter groups, loss, and class-sampling quotas.

| Task folder | Notebooks | Models |
| --- | --- | --- |
| [Cataract severity](cataract%20severity%20task/) | 11 | ResNet-18/34/50, DenseNet-121/169, EfficientNet-B1/B2/B3/B4/B5, VGG-16 |
| [Poor pupil dilation](poor%20pupil%20dilation%20task/) | 10 | ResNet-18/34/50, DenseNet-121/169, EfficientNet-B1/B2/B3/B4/B5 |

`Final` in a filename identifies a retained model-development notebook. These files do not run an Optuna search or prove that every setting matches a winner from the current separate Optuna implementation.

## Shared workflow

1. Mount Google Drive and extract the task's image archive.
2. Read the CSV and derive the task labels from its one-hot columns.
3. Keep frames from each `videoname` in one partition using two `GroupShuffleSplit` stages, seed 42, targeting approximately 70/15/15 by video groups.
4. Apply class-quota sampling and stochastic augmentation to training frames only; use deterministic preprocessing for validation and test frames.
5. Fine-tune a pretrained CNN with AdamW for 30 epochs, batch size 64, and `ReduceLROnPlateau` driven by validation loss.
6. Save the checkpoint with the lowest validation loss, reload it, and print held-out test metrics and confusion matrices.

The trainable layers and learning rates differ by model. Severity and dilation configurations also differ, even for the same backbone. The original notebooks use one selected device; they do not schedule two GPU workers or implement K-fold.

## Running and saving

Open one notebook in Google Colab, enable a GPU, update `ZIP_PATH`, `CSV_PATH`, and extraction paths, then run cells in order. The CSV and images must agree on original video and frame identifiers. The task-specific READMEs give the exact schemas.

The best checkpoint is a model `state_dict`; optimizer, scheduler, RNG, and epoch state are not saved for complete training resume. Metrics and plots are printed/displayed rather than comprehensively exported to CSV/JSON. Download the checkpoint and preserve the executed notebook before ending the runtime.

The notebooks reuse generic checkpoint names across architectures. Use a separate working directory or a unique `SAVE_PATH` for each model/task/run to avoid overwriting another model's weights. Some comments and debug augmentation previews are historical; the active training cells determine the experiment.

The source notebooks do not seed all stochastic operations, even though their group split uses a fixed seed. An exact rerun is therefore not guaranteed from seed 42 alone.

## Related implementations

[CNN K-fold](../../K%20Fold/cnn%20models/ReadMe.md) extracts these settings into a validated JSON configuration file and provides per-fold outputs. [Optuna](../../Optuna/ReadMe.md) is the separate CNN search workflow.

[Notebook overview](../README.md)
