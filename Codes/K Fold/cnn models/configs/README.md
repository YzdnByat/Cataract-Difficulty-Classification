# Retained CNN configurations

[winning_configs.json](winning_configs.json) is the configuration source read by `run_kfold.py`. It stores **21 task/model entries** extracted from the final CNN notebooks: 11 for severity and 10 for poor dilation.

The filename uses the project's historical term `winning`. These are retained selected notebook settings, not the output of a new search performed by the K-fold engine. The runner does not import the separate Optuna folder's `best_hyperparameters.json` files automatically.

## Coverage and provenance

| Task | Model key | Input side, pixels | Current source notebook |
| --- | --- | --- | --- |
| `poor_dilation` | `densenet121` | 224 | [Final_2_DenseNet_121.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_DenseNet_121.ipynb) |
| `poor_dilation` | `densenet169` | 224 | [Final_2_DenseNet_169.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_DenseNet_169.ipynb) |
| `poor_dilation` | `efficientnet_b1` | 224 | [Final_2_EfficientNet_B1.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_EfficientNet_B1.ipynb) |
| `poor_dilation` | `efficientnet_b2` | 224 | [Final_2_EfficientNet_B2.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_EfficientNet_B2.ipynb) |
| `poor_dilation` | `efficientnet_b3` | 224 | [Final_2_EfficientNet_B3.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_EfficientNet_B3.ipynb) |
| `poor_dilation` | `efficientnet_b4` | 224 | [Final_2_EfficientNet_B4.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_EfficientNet_B4.ipynb) |
| `poor_dilation` | `efficientnet_b5` | 224 | [Final_2_EfficientNet_B5.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_EfficientNet_B5.ipynb) |
| `poor_dilation` | `resnet18` | 224 | [Final_2_ResNet_18.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_ResNet_18.ipynb) |
| `poor_dilation` | `resnet34` | 224 | [Final_2_ResNet_34.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_ResNet_34.ipynb) |
| `poor_dilation` | `resnet50` | 224 | [Final_2_ResNet_50.ipynb](../../../notebooks/cnn/poor%20pupil%20dilation%20task/Final_2_ResNet_50.ipynb) |
| `severity` | `densenet121` | 224 | [Final_DenseNet_121.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_DenseNet_121.ipynb) |
| `severity` | `densenet169` | 224 | [Final_DenseNet_169.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_DenseNet_169.ipynb) |
| `severity` | `efficientnet_b1` | 240 | [Final_EFF_B1.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_EFF_B1.ipynb) |
| `severity` | `efficientnet_b2` | 260 | [Final_EFF_B2.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_EFF_B2.ipynb) |
| `severity` | `efficientnet_b3` | 260 | [Final_Eff_B3.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_Eff_B3.ipynb) |
| `severity` | `efficientnet_b4` | 380 | [Final_EFF_B4.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_EFF_B4.ipynb) |
| `severity` | `efficientnet_b5` | 456 | [Final_EFF_B5.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_EFF_B5.ipynb) |
| `severity` | `resnet18` | 224 | [Final_ResNet_18.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_ResNet_18.ipynb) |
| `severity` | `resnet34` | 224 | [Final_ResNet_34.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_ResNet_34.ipynb) |
| `severity` | `resnet50` | 224 | [Final_ResNet_50.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_ResNet_50.ipynb) |
| `severity` | `vgg16` | 224 | [Final_VGG_16.ipynb](../../../notebooks/cnn/cataract%20severity%20task/Final_VGG_16.ipynb) |

Some `source_notebook` strings in the JSON retain historical download suffixes such as ` (1)`. The links above point to the current checked-in notebook filenames; the metadata string is not a runtime import path.

## Entry fields

| Field | Meaning |
| --- | --- |
| `task`, `model`, `num_classes` | Task/architecture key and classifier output size |
| `source_notebook` | Provenance of the retained settings |
| `input_resolution`, `batch_size`, `max_epochs` | Physical training dimensions and budget |
| `pretrained` | Initialize from pretrained backbone weights |
| `fine_tuning.unfreeze_scopes` | Trainable model submodules |
| `optimizer.parameter_groups` | Exact parameter scopes, learning rates, and weight decay |
| `loss` | Class weights and label smoothing |
| `scheduler` | Validation-loss plateau scheduler settings |
| `sampler` | Per-class training quotas and replacement behavior |
| `augmentation` | Padding, resizing, geometric/photometric transforms, normalization |
| `checkpoint_monitor` | Validation-loss checkpoint selection |
| `reference_run`, `extraction_issues` | Historical printed-run context and extraction notes, where present |

Reference-run values are not K-fold summaries and must not be presented as current cross-validation results. The earlier checkpoint-name and RNG issues recorded in extraction notes explain why the shared runner uses distinct output paths and explicit seeds.

Severity and dilation use different settings for the same architecture. In particular, all current dilation entries use 224-pixel inputs, while severity EfficientNet sizes vary. The severity EfficientNet entries omit an explicit minimum scheduler LR (`null`), whereas other entries can set one. Preserve this distinction rather than substituting a single generic configuration.

## Protocol settings and changes

The JSON includes descriptive protocol defaults, but actual `--seed` and `--n-splits` values come from the CLI. `resolved_config.json` and the task `protocol.json` record each execution. Saved fold assignments are the definitive membership record.

Validate from the parent implementation directory with `python validate_project.py`. After deliberately changing settings, use a new output directory. Existing-run guards prevent incompatible resolved configurations from being mixed. There is no poor-dilation VGG-16 entry to infer from the severity settings.

[CNN K-fold implementation](../ReadMe.md)
