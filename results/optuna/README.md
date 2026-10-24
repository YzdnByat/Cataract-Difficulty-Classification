# Optuna results

I used Optuna to select CNN training hyperparameters for cataract severity classification and poor pupil dilation detection. This section presents the selected settings and resulting validation and test performance.

| Task | Models | Trial attempts | Results |
| --- | ---: | ---: | --- |
| Cataract severity | 10 | 120 | [Comparison and findings](severity/README.md) |
| Poor pupil dilation | 10 | 120 | [Comparison and findings](poor_dilation/README.md) |

## What I optimized

I optimized four parameters: the classification-head learning rate, backbone learning rate, weight decay, and label smoothing. I ran 12 trial attempts per architecture and task, with an eight-epoch budget per trial. Across both tasks, 106 trials completed and 134 were pruned.

I selected the trial with the highest validation macro-F1, then trained a final model with its selected parameters. Final training used AdamW and ReduceLROnPlateau, allowed up to 50 epochs, and stopped after 10 epochs without improvement in validation macro-F1. I evaluated the best validation checkpoint on the held-out test partition.

## How to read the scores

| Score | Meaning |
| --- | --- |
| Search validation macro-F1 | The selected Optuna trial's objective. |
| Final validation macro-F1 | The best checkpoint score during the separate final training run; included in `results.csv`. |
| Test scores | Performance of that final checkpoint on the held-out test partition. |

I kept training, validation, and test partitions separate by video ID, using split seed 42. I report frame-level, single-split scores here; K-fold results are presented separately. My task pages order models by search validation performance rather than test performance.

## Files

Each task has only four files: a README, `results.csv`, `best_hyperparameters.csv`, and `optimization_summary.png`. The README provides the comparison directly on GitHub; the CSV files retain full numerical precision. I retain detailed trial histories and execution records in the original `optuna_outputs.zip`.
