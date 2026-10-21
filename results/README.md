# Results

Five-fold validation results for cataract severity classification and poor pupil dilation detection from surgical microscope frames.

| Model family | Cataract severity | Poor pupil dilation |
| --- | --- | --- |
| CNNs | [11 models](kfold/cnn/severity/) | [10 models](kfold/cnn/poor_dilation/) |
| Transformers | [4 models](kfold/transformers/severity/) | [4 models](kfold/transformers/poor_dilation/) |

## Highest mean macro-F1 in each group

Values are mean ± sample standard deviation over five folds, in percent. Ranking is descriptive and does not establish statistical significance.

| Task | Family | Model | Accuracy (%) | Macro-F1 (%) |
| --- | --- | --- | ---: | ---: |
| Cataract severity | CNN | [ResNet50](kfold/cnn/severity/resnet50/) | 86.42 ± 2.78 | 86.05 ± 2.07 |
| Poor pupil dilation | CNN | [ResNet50](kfold/cnn/poor_dilation/resnet50/) | 93.87 ± 2.68 | 88.87 ± 3.70 |
| Cataract severity | Transformer | [Swin-Small](kfold/transformers/severity/Swin-Small/) | 88.59 ± 1.85 | 89.54 ± 2.61 |
| Poor pupil dilation | Transformer | [ViT-B/16](kfold/transformers/poor_dilation/ViT-B_16/) | 96.68 ± 1.29 | 94.33 ± 1.00 |

## Evaluation

The recorded runs use five stratified folds grouped by video ID and seed **42**. Severity uses **2,541 frames in 645 video-ID groups**; poor dilation uses **3,081 frames in 700 groups**. These are the records represented in the saved experiment outputs.

CNN and transformer fold assignments match for each task. Their training configurations differ. The held-out fold supplies both checkpoint selection by validation loss and final scores, so these are validation cross-validation results. Scores are calculated per frame; grouping by video ID does not itself establish separate patient identities across videos.

Task pages provide comparison tables and class definitions. Every model folder contains the same five file types: fold scores, pooled per-class scores, a pooled confusion matrix, its numeric counts, and combined training curves. CSV scores use fractions; rendered tables use percent. Pooled out-of-fold (OOF) metrics combine all held-out predictions once and can differ from fold means.

## Full records

The complete supplied packages are `CNN-KFold-Results(1).zip` and `Transformer-KFold-Results(1).zip`. They retain the detailed fold outputs and execution records omitted from this compact folder. Trained checkpoints are not included in those packages. No models were retrained to prepare this presentation.
