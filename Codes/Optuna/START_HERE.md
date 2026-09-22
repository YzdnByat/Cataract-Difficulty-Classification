# Running in Kaggle

1. Upload optuna_complete.zip as a Kaggle dataset/input and attach it to a new notebook.
2. Import Kaggle_Optuna.ipynb (inside this ZIP) into the notebook editor.
3. Enable Internet and a GPU (T4 is appropriate for the previously used environment).
4. Cell 1 chooses TASK and MODELS. Start with resnet18; this performs the full 12-trial search plus final training, not a short smoke test. Cells 1-5 only prepare and validate.
5. Run cells 1-5 and inspect the splits; cell 6 trains; cell 7 lists saved outputs.
6. To train all 10 models, use MODELS=["all"] with a new RUN_NAME, or list only models not already completed. Use separate task runs as needed.

This package includes all 11 .py files, including the original analyze_datset.py auxiliary utility (not needed by the launcher). Training code retains original fixed unfreezing depths and search settings. metadata_utils.py, dataset.py and config.py incorporate the previously supplied label/path patch.

Bundled CSVs are the original-name snapshots you uploaded: Dataset_Severity_V2(1).csv and Dataset_poordilation.csv. The notebook downloads the corresponding image ZIP from the supplied Drive ID. If your Drive data has changed since those snapshots, update the CSV before running. The notebook checks every image and split, but cannot verify clinical annotation correctness or that different video IDs belong to distinct patients.

The notebook uses one GPU, as does the original Optuna runner. It does not add concurrent studies or interrupted-trial resume. All models may exceed one Kaggle session; schedule model subsets. Completed reports are preserved by the output-exists guard. An interrupted model needs a fresh run/output name; it does not resume its previous trial. Keep prior outputs before starting a new session.

No GPU training or Drive download was performed while preparing this package. Python syntax and metadata conversions were checked locally. Environment versions are recorded by the notebook; dependencies are not fully pinned. Do not install a replacement torch build merely to run these cells.
