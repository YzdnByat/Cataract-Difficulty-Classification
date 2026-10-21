# CNN K-fold results

Verified CNN experiment outputs for cataract severity classification and poor pupil dilation detection. These files accompany the project's [CNN K-fold Python implementation](../../../Codes/K%20Fold/cnn%20models/ReadMe.md).

| Task | Models | Completed fold runs | Unique frame records | Video-ID groups | Seed |
| --- | --- | --- | --- | --- | --- |
| [Cataract severity](severity/) | 11 | 55 | 2,541 | 645 | 42 |
| [Poor pupil dilation](poor_dilation/) | 10 | 50 | 3,081 | 700 | 42 |

Every listed model has five completed folds. Each task folder contains a rendered comparison table and its full-precision CSV, the original exact fold assignments, class/fold audit, and protocol, followed by model folders with per-fold and pooled results.

## What is included

- Original per-fold metrics, prediction probabilities, histories and confusion-matrix PNGs.
- Original per-model configurations, fold tables, mean/sample-SD summaries, pooled OOF predictions and matrices.
- Original task-level fold manifests, audits and protocol fingerprints; original launcher logs and the available launcher status.
- Derived per-class fold/pooled reports, numeric count and row-normalized confusion matrices, and five-fold learning-curve figures.
- Task-wide and combined model-comparison CSVs, source/archive hashes, a file inventory, and metric-verification records.

[Combined model comparison](model_comparison.csv) · [Verification summary](verification_summary.csv) · [Verification details](verification_report.json) · [Source archives](source_archives.json) · [File inventory](files_manifest.csv)

## Reading the scores

The split is video-grouped `StratifiedGroupKFold`. The held-out fold is used to select the minimum-validation-loss checkpoint and to report performance. These are validation cross-validation results, not nested CV or an additional untouched test evaluation.

Fold means and sample standard deviations use the five recorded folds (`ddof=1`). Pooled OOF metrics are recomputed from all held-out frames and can differ from the fold mean. Numeric CSV/JSON scores and SDs are fractions in [0, 1]; README tables explicitly marked percent multiply both by 100. They are frame-level scores, not video-level scores.

Severity indices are `0=low`, `1=dense`, `2=mature`, `3=brunescent`. Dilation indices are `0=normal`, `1=poor_dilation`. Confusion-matrix rows are true labels and columns are predictions; normalized CSV rows sum to one.

## Verification and provenance

All **105 folds** were checked against their original manifests. Accuracy, macro-F1, weighted-F1, and ROC-AUC were recomputed from saved labels/probabilities; fold tables, pooled OOF summaries, selected epochs, and histories were cross-checked. Original PNGs were checked for readability. No retraining was performed.

The actual archived runs use **seed 42**. A different seed mentioned in another project run does not change these recorded results. Original GPU-specific shared files were identical and have been stored once per task. Original member locations and SHA-256 hashes are recorded in the file inventory. All other original result/launcher files are retained byte-for-byte.

The supplied archives contain no trained checkpoints, exact training dependency lock, or authenticated training-code commit identifier. Configurations and logs preserve the available execution evidence. The current GitHub implementation is a navigation reference, not a claimed code revision for the historical training run.

Additional tables, numeric matrices, figures and READMEs were created from the archived outputs and are marked `derived` in the inventory. `files_manifest.csv` covers every published file except itself, to avoid a circular checksum.

The verification checks stored outputs; it does not establish clinical annotation correctness, distinct patient identities across video IDs, or external clinical performance.
