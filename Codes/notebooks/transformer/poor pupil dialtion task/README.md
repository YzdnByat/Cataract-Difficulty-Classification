# Poor pupil dilation: transformer benchmark

[transformer-models-poor-dilation.ipynb](transformer-models-poor-dilation.ipynb) compares nine pretrained ViT, DeiT, and Swin variants for `poor_dilation` on one video-grouped train/validation/test partition. It is a single-split benchmark, not K-fold or an Optuna search.

## Input contract

| Item | Required value |
| --- | --- |
| Zero-based class order | Normal, poor dilation |
| Original metadata format | CSV |
| Required columns | `filename`, `videoname`, `normal`, `poor dilation` |
| Canonical frame records | 3,081 |
| Video-ID groups | 700 |
| Class counts in order | 2,541 / 540 |
| Image archive layout | Frames within folders named by the original video IDs |

The notebook contains original Drive/Sheets source settings and permits local overrides through `LOCAL_FILES['poor_dilation']['zip']` and `LOCAL_FILES['poor_dilation']['metadata']`. It enforces the expected canonical metadata checksum after validating labels and collapsing exact repeated records. Arbitrarily substituting a renamed or edited export can fail that check.

## Running

1. Open this notebook in Kaggle with Internet and GPU T4 x2 enabled.
2. Review Section 2 settings: `TASKS`, `MODEL_NAMES`, `RUN_NAME`, `CONFIG`, and local input overrides.
3. Run the dependency and data-preparation sections. Verify counts, image matches, and split distributions.
4. Continue through the model definitions and dual-GPU launcher. The default starting configuration uses 30 maximum epochs, batch size 32, early-stopping patience 5, full-backbone fine-tuning, backbone/head learning rates 1e-5/1e-4, AdamW, and cosine decay.
5. Preserve the complete output folder or enable `EXPORT_ZIP` in the final cell to package it.

Set `RUN_TRAINING=False` before execution for a data audit. The supplied training launcher requires two GPUs; each worker trains a separate model. The settings are not Optuna-selected winners.

## Split and selection

The split targets approximately 70/15/15 by video groups, stratifies by each group's label set, and uses seed 42. All frames from one video ID stay together. The best epoch is selected by validation loss; completed architectures are ordered by validation macro-F1. Test scores are reported after checkpoint selection, without selecting the winning architecture from test performance.

## Outputs

The default task directory is `/kaggle/working/outputs/transformers_poor_dilation_v2_two_lr/poor_dilation/`.

Each model folder saves `best.pth`, `last.pth`, `history.csv`, `metrics.json`, validation/test prediction CSVs, protocol and pretrained-model records, trainable parameter names, and a completion marker. The task directory contains its split manifest, data-source records, summaries, model selection, status, and worker logs.

Restore this whole run directory to the same output location to resume. The last completed epoch of unfinished models is recoverable; completed models are skipped. Choose a new `RUN_NAME` when changing data or settings. The committed notebook has no saved training outputs, so architecture availability does not imply completed results.

See the [transformer overview](../README.md) for the complete architecture registry, worker-source behavior, and resume details. The corresponding [K-fold implementation](../../../K%20Fold/transformer%20models/poor%20pupil%20dilation%20task/) evaluates repeated grouped folds.
