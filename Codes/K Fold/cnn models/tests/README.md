# CNN K-fold unit tests

These are the existing lightweight checks for the CNN K-fold implementation. They test configuration loading and synthetic fold construction without running GPU training or downloading model weights.

| File | Coverage |
| --- | --- |
| [test_config_loader.py](test_config_loader.py) | All 21 configurations load; task/model names normalize; selected retained values match expectations |
| [test_dual_config.py](test_dual_config.py) | 11 severity and 10 dilation entries; task-specific optimizer/settings differences; missing dilation VGG-16 |
| [test_folds.py](test_folds.py) | Video isolation and class coverage on synthetic folds; legacy filename/video column aliases |

## Run

From `Codes/K Fold/cnn models/`, after installing its requirements:

```bash
python -m unittest discover -s tests -v
```

Run from the parent implementation directory so local imports such as `config_loader` and `folds` resolve. Do not run from inside `tests/` without configuring the import path.

These tests do not establish correct patient identities, image/label validity, actual dataset fold distributions, model accuracy, complete artifact integrity, or epoch-resume behavior. Audit real metadata with `run_kfold.py --dry-run`. The separate `python validate_project.py --build-models` check additionally instantiates the configured architectures and checks optimizer parameter coverage without pretrained downloads.

[CNN K-fold implementation](../ReadMe.md)
