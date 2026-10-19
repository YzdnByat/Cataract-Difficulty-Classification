# Cataract severity: transformer K-fold

[transformer-k-fold-severity.ipynb](transformer-k-fold-severity.ipynb) evaluates transformer candidates for `severity` using five video-grouped folds. All required worker code is embedded in the notebook.

## Inputs

| Item | Expected value |
| --- | --- |
| Model-label order | Low, dense, mature, brunescent |
| Original metadata | XLSX, sheet `Dataset_Severity_V2` |
| Required columns | `filename`, `videoname`, `low nuclear density`, `dense`, `mature`, `brunescent` |
| Canonical unique records / video-ID groups | 2,541 / 645 |
| Class counts in order | 795 / 1,047 / 614 / 85 |
| Images | Original-ID image ZIP, with frames within video-ID folders |
| Optional exact folds | Matching CNN `fold_assignments.csv` |
| Optional model ranking | Same-task transformer summary containing validation macro-F1 |

Original source URLs are configured in Section 2. Use `LOCAL_FILES` for locally attached original data. The loader validates one-hot labels, exact metadata checksum, group/record counts, image paths, and optionally image readability. Repeated identical records are collapsed; conflicting labels are rejected.

## Run

Open in Kaggle, enable Internet and a GPU (T4 x2 preferred), then run sections in order. Defaults are seed 42, five folds, four candidates, 20 maximum epochs, batch size 32, and early-stopping patience 4. The backbone/head learning rates are 1e-5/1e-4 with full fine-tuning, AdamW, and cosine decay.

Set `CNN_FOLDS_CSV` to the exact matching CNN manifest to reuse its assignments. Otherwise the notebook creates and saves grouped folds. Default candidates are ViT-B/16, DeiT-Small, Swin-Tiny, and Swin-Small. They are not claimed to be the top four unless a compatible prior validation summary supplies that ranking through `PRIOR_RESULTS_CSV`.

`RUN_TRAINING=False` permits a data/fold audit without training. During training, up to two independent model/fold jobs run concurrently, one per GPU; one GPU also works sequentially.

## Saved results

The default task root is `/kaggle/working/outputs/transformer_kfold_severity_original_ids_fast_v1/severity/`.

It contains the fold manifest/audit, data and engine provenance, candidate-selection record, status and model summary, logs, per-fold checkpoints/histories/metrics/predictions, and per-model OOF and per-class summaries. Completed-fold counts are explicit. The held-out fold selects the checkpoint by minimum validation loss and supplies its reported metrics; these are validation CV scores, not an additional test set.

## Continue after interruption

Use the same run name, data, settings, and embedded engine. Unfinished folds retain `last.pth` for last-saved-epoch resume; completed folds are checked and skipped. Their best weights remain even when completed-fold resume states are removed to save disk.

Set `EXPORT_ZIP=True` after a complete or partial session. In a new session, restore that archive through `RESUME_LOCAL_ZIP` or `RESUME_OUTPUTS_ZIP_URL`. Preserve the whole exported run structure. Model checkpoints from the single-split notebooks cannot substitute for K-fold resume archives.

[Full transformer K-fold guide](../README.md) · [Corresponding single-split benchmark](../../../notebooks/transformer/cataract%20severity%20task/)
