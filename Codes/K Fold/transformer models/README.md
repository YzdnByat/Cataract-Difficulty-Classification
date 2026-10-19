# Transformer K-fold notebooks

This folder contains two self-contained Kaggle notebooks for grouped K-fold evaluation of transformer models. Each notebook embeds its own Python engine; no separate checked-in transformer `.py` package is needed.

| Folder | Notebook | Task |
| --- | --- | --- |
| [Cataract severity](cataract%20severity%20task/) | `transformer-k-fold-severity.ipynb` | Four-class severity |
| [Poor pupil dilation](poor%20pupil%20dilation%20task/) | `transformer-k-fold-poor-dilation.ipynb` | Binary dilation |

## Candidate selection

Both notebooks default to **ViT-B/16, DeiT-Small, Swin-Tiny, and Swin-Small**. This is an editable candidate shortlist, not a verified ranking of the highest-performing models.

Set `PRIOR_RESULTS_CSV` to a previous summary for the same task to select up to `TOP_N=4` supported models by validation macro-F1. The file must contain `model` and `val_macro_f1` (or `validation_macro_f1`). Without it, `MODEL_NAMES` supplies the candidates. `model_selection.json` records how selection was made. Test metrics do not drive this shortlist.

The embedded registry supports the same nine ViT/DeiT/Swin variants as the [single-split transformer notebooks](../../notebooks/transformer/). Candidate shortlisting from prior experiments is separate from fold training; this is not nested model selection within each outer fold.

## Default protocol

| Setting | Default |
| --- | --- |
| Splitter | `StratifiedGroupKFold`, five folds, shuffle enabled, seed 42 |
| Group | Original `video_id` |
| Fold reuse | Optional `CNN_FOLDS_CSV`; saved/restored assignments take precedence |
| Input | 224 × 224, ImageNet normalization |
| Model | Pretrained backbone, L2-normalized pooled features, new linear head |
| Fine-tuning | All backbone layers plus classifier |
| Backbone / head LR | 1e-5 / 1e-4 |
| Optimizer / scheduler | AdamW, weight decay 0.01; cosine annealing |
| Physical batch / workers | 32 / 2 per job |
| Epoch limit / early-stopping patience | 20 / 4 |
| Sampling | Training-only inverse-frequency weighted sampling with replacement |
| Loss | Severity: cross-entropy; dilation: one-logit binary cross-entropy |
| Best checkpoint | Minimum held-out-fold validation loss |
| GPU schedule | At most two independent model/fold jobs; one GPU is also supported |

The default budget is four models × five folds = 20 jobs per task. Two GPUs run separate jobs, not one model split across two devices.

The held-out fold determines early stopping/checkpoint selection and supplies the final fold scores. Interpret the outputs as validation CV results, not independent test results. The sampling, loss, and training budget differ from CNN K-fold.

## Setup and numbered sections

Open the appropriate notebook in Kaggle with Internet and a GPU accelerator, preferably T4 x2. Keep Kaggle's working torch/torchvision pair; the dependency cell installs `timm==1.0.22`, `gdown==5.2.0`, and `openpyxl`.

1. Install dependencies.
2. Review `RUN_NAME`, `RUN_TRAINING`, `N_SPLITS`, `SEED`, `TRAINING`, `MODEL_NAMES`, local data overrides, prior results, and fold reuse.
3. Check the GPU/environment.
4. Write and import the embedded engine.
5. Restore a previous output archive if continuing a run.
6. Verify original metadata/images and create or reuse the fold manifest.
7. Select the candidate subset and save its provenance.
8. Launch or resume workers.
9. Inspect fold completion and summaries.
10. Optionally export the complete or partial output directory.

`RUN_TRAINING` currently defaults to `True`. Set it to `False` for data/fold preparation and summary inspection without launching training. `LOCAL_FILES` accepts original image-ZIP and metadata paths; it does not bypass canonical metadata checks.

## Exact fold reuse

Provide the matching CNN `fold_assignments.csv` through `CNN_FOLDS_CSV` when comparing families on identical partitions. The importer validates original identifiers, labels, fold numbers, row coverage, video isolation, and class coverage. If a restored transformer manifest already exists, it is reused and must agree with any supplied CNN manifest.

Matching the seed alone is insufficient because record order, metadata, and software versions can change assignments. A historical seed-11 manifest can be reused explicitly; keep the settings consistent with its recorded provenance. The current notebook defaults remain seed 42.

## Outputs

The default task root is `/kaggle/working/outputs/<RUN_NAME>/<task>/`.

| Scope | Key artifacts |
| --- | --- |
| Task | `fold_assignments.csv`, `fold_audit.csv`, `data_protocol.json`, `data_sources.json`, `model_selection.json`, `notebook_settings.json`, `engine_source.py`, `worker_settings.json`, `run_status.csv`, `model_summary.csv`, worker logs |
| Model | `fold_metrics.csv`, `out_of_fold_predictions.csv`, `oof_metrics.json`, `oof_confusion_matrix.csv`, `oof_confusion_matrix.png`, `per_class_metrics.csv` |
| Fold | `protocol.json`, `best.pth`, `history.csv`, `validation_predictions.csv`, `metrics.json`, `complete.json`, confusion matrix and training curves |
| Unfinished fold | `last.pth`, containing model, optimizer, scheduler, scaler, RNG, and epoch state |

Fold predictions preserve original identifiers, fold index, labels, and probabilities. `*_mean` / `*_std` summarize completed folds with sample standard deviation; `oof_*` scores pool their held-out predictions. Partial results remain marked incomplete. A complete run covers every original record once in its OOF table.

The engine does not automatically run t-SNE or Grad-CAM.

## Resume and export

Keep `RUN_NAME`, metadata, model settings, and engine unchanged. Rerunning with restored outputs resumes unfinished folds from their last saved epoch and skips validated completed folds. Best checkpoints remain after completion. By default, large `last.pth` files are removed from completed folds, while unfinished-fold states are retained.

Set `EXPORT_ZIP=True` in the final section to package a complete or partial run. In a fresh session, set `RESUME_LOCAL_ZIP` or `RESUME_OUTPUTS_ZIP_URL` to that exported archive. It must have the same `RUN_NAME` as its top-level directory; conflicting existing files are rejected. CNN archives and single-split transformer checkpoints are not interchangeable with these resume archives.

The notebook writes the engine from `ENGINE_SOURCE` and records its source hash. Changes to the embedded code or incompatible run settings require a new run name.

[K-fold overview](../ReadMe.md)
