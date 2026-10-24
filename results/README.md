# Results

I report K-fold evaluations and CNN hyperparameter optimization for cataract severity classification and poor pupil dilation detection.

| Experiment | Cataract severity | Poor pupil dilation |
| --- | --- | --- |
| CNN K-fold | [11 models](kfold/cnn/severity/) | [10 models](kfold/cnn/poor_dilation/) |
| Transformer K-fold | [4 models](kfold/transformers/severity/) | [4 models](kfold/transformers/poor_dilation/) |
| Optuna optimization | [10 CNNs](optuna/severity/) | [10 CNNs](optuna/poor_dilation/) |

## K-fold summary

These models achieved the highest mean macro-F1 per task and family. Scores are **five-fold mean ± sample standard deviation (%)**.

| Task | Family | Model | Accuracy (%) | Macro-F1 (%) |
| --- | --- | --- | ---: | ---: |
| Cataract severity | CNN | [ResNet50](kfold/cnn/severity/resnet50/) | 86.42 ± 2.78 | 86.05 ± 2.07 |
| Poor pupil dilation | CNN | [ResNet50](kfold/cnn/poor_dilation/resnet50/) | 93.87 ± 2.68 | 88.87 ± 3.70 |
| Cataract severity | Transformer | [Swin-Small](kfold/transformers/severity/Swin-Small/) | 88.59 ± 1.85 | 89.54 ± 2.61 |
| Poor pupil dilation | Transformer | [ViT-B/16](kfold/transformers/poor_dilation/ViT-B_16/) | 96.68 ± 1.29 | 94.33 ± 1.00 |

I used identical stratified, video-grouped folds for both families (seed **42**): **2,541 severity frames/645 groups** and **3,081 dilation frames/700 groups**. Training configurations differ.

I selected checkpoints by held-out validation loss and scored the same folds. These are frame-level validation cross-validation results; rankings are descriptive. Separate patient identities across video IDs are unverified.

## Optuna results

I ran **12 trial attempts per CNN and task (240 total)**, selected hyperparameters by validation macro-F1, and trained final models using fixed training/validation/test partitions.

The [Optuna section](optuna/README.md) contains comparisons, selected hyperparameters, and charts. I distinguish search validation, final training validation, and test scores, keeping them separate from K-fold results.

CSV scores use fractions in [0, 1]; README tables use percentages.
