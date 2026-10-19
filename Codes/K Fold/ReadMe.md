# Grouped K-fold evaluation

This folder evaluates fixed model configurations across repeated held-out video groups. It contains two different implementations because the CNN and transformer experiments were developed in different forms.

| Folder | Implementation | Coverage |
| --- | --- | --- |
| [cnn models](cnn%20models/ReadMe.md) | Modular `.py` files, launched through `run_kfold.py` | 11 severity CNNs and 10 dilation CNNs, using task/model-specific final-notebook configurations |
| [transformer models](transformer%20models/) | Two standalone `.ipynb` notebooks with an embedded Python worker | One notebook per task; four configurable candidates by default |

## Common evaluation design

Both use `StratifiedGroupKFold` with five folds and seed 42 by default. A video ID belongs to only one held-out fold, and each training/validation partition must contain every task class. Numeric labels are zero-based: severity `[low, dense, mature, brunescent]`, dilation `[normal, poor dilation]`.

The held-out fold selects the minimum-validation-loss checkpoint and supplies the reported fold metrics. These are validation cross-validation scores. There is no separate outer test fold and no nested hyperparameter search inside these implementations. In particular, reusing settings selected earlier does not make the earlier selection procedure independent of all subsequent folds.

Video-ID isolation does not prove distinct patient identities across videos. Exact shared folds also require matching records, ordering, splitter version, and saved assignments; a shared seed alone is insufficient. The transformer notebooks accept the original CNN `fold_assignments.csv` for exact fold reuse after validating identifiers and labels.

## Main implementation differences

| Detail | CNN Python runner | Transformer notebooks |
| --- | --- | --- |
| Settings | Loaded from `configs/winning_configs.json` | Section 2 settings plus embedded engine |
| Input metadata | CSV with one normalized `label` column | Original one-hot severity XLSX / dilation CSV, with checksum checks |
| Schedule | Sequential models/folds, one selected device per process | Up to two model/fold workers, one per GPU |
| Maximum epochs | 30 from retained configurations | 20 by default |
| Early stopping | None | Patience 4 by default |
| Completed jobs | Skip with `--skip-existing` | Validate and skip automatically |
| Incomplete jobs | Restart the fold | Resume the last saved epoch |
| Aggregation | Fold tables, mean/std and pooled OOF metrics | Fold tables, mean/std, pooled OOF and per-class metrics, job status |

The different epoch budgets, losses, sampling, fine-tuning, and augmentation settings matter when interpreting comparisons. These are the actual implementations, not an assertion that every family was trained under an identical protocol.

## Read the outputs correctly

An out-of-fold (OOF) table collects predictions from the folds where frames were held out. A complete five-fold model should cover every input record once. Incomplete runs cover only completed folds and must remain labeled partial.

Fold mean ± sample standard deviation measures variation across completed folds. Pooled OOF metrics are recomputed from all saved held-out predictions; they need not equal the fold mean. Preserve the fold manifest, completed-fold count, resolved settings, class order, and prediction files alongside any reported score.

See the implementation READMEs for exact setup commands, inputs, outputs, and restart behavior.

[Code overview](../ReadMe.md)
