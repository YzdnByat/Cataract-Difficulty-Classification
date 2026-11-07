# Swin-Small severity comparisons

The presets use the current **448 × 448** Swin-Small setup with the same edge-blended padding, Albumentations training transforms, ImageNet normalization, and training loop. The backbone is constructed at 448 pixels. The default comparison config uses batch size 8, seed 42, up to 30 epochs, and the best validation macro-F1 checkpoint. The existing validation-loss stopping rule, optimizer, cosine schedule, and weighted training sampler are used.

## Experiments

| Preset | Training cohort | Validation and test cohort | Color correction |
|---|---|---|---|
| `baseline` | All sources, classes 0–3 | All sources, classes 0–3 | None |
| `farabi3` | Farabi, classes 0–2 | Farabi, classes 0–2 | None |
| `pooled3` | All available sources for classes 0–2 | Farabi, classes 0–2 | None |
| `gray_world` | All sources, classes 0–3 | All sources, classes 0–3 | Gray World on Noor frames |
| `white_patch` | All sources, classes 0–3 | All sources, classes 0–3 | White Patch on Noor frames |

The label order is 0 low nuclear density, 1 dense, 2 mature, and 3 brunescent. The source rule uses `S1` for Farabi, `S2` for Noor, and `Brunescent_*` for YouTube/external, following the image audit. The severity data contains Noor frames only for class 0 and YouTube frames only for class 3.

Preparation validates the original severity XLSX and ZIP and makes a shared 70/15/15 video-level split using seed 42. The current dataset yields 1,795 train, 385 validation, and 361 test frames. All presets filter this same manifest after the split. `farabi3` uses 1,561 train, 330 validation, and 304 test frames. `pooled3` adds 181 Noor class-0 training frames to `farabi3` and uses the same Farabi validation and test frames. The added frames also change the amount of training data and the number of sampled updates per epoch.

## Run on Kaggle

Upload the project files under `Codes/swin-small-comparisons/` in the repository. Import the separate `Swin_Small_Comparisons_Kaggle.ipynb` launcher into Kaggle. Enable Internet and select a GPU for training. Set `ACTION` to a preset name for one training run, `preview` for the color comparison panels, or `compare` for the report. The default action is `baseline`. The launcher passes the audited severity Drive IDs to the preparation command and writes outputs under `/kaggle/working/swin-comparisons`.

To compare separate saved runs, attach their outputs as notebook inputs and set `EXTRA_RUN_ROOTS` to their run folders or the folders directly containing them. The comparison reads saved predictions and split files. It can run without the image archive.

## Run from the terminal

Install the requirements in an environment with matching CUDA-enabled PyTorch and torchvision. Run the commands from this project folder.

```bash
python -m pip install -r requirements.txt

python experiments.py prepare --data-dir ./severity-data \
  --images-zip /path/to/Dataset_Severity_V2.zip \
  --metadata /path/to/Dataset_Severity_V2_1.xlsx

python experiments.py preview --data-dir ./severity-data --output-dir ./outputs/noor-previews

python train.py --preset baseline --data-dir ./severity-data --output-dir ./outputs --gpu 0
python train.py --preset farabi3 --data-dir ./severity-data --output-dir ./outputs --gpu 0
python train.py --preset pooled3 --data-dir ./severity-data --output-dir ./outputs --gpu 0
python train.py --preset gray_world --data-dir ./severity-data --output-dir ./outputs --gpu 0
python train.py --preset white_patch --data-dir ./severity-data --output-dir ./outputs --gpu 0

python experiments.py compare --runs-dir ./outputs --output-dir ./outputs/comparison
```

For Drive downloads, pass `--images-drive-id` and `--metadata-drive-id` to `prepare` in place of the local paths. Append `--prepare-only` to any training command to check the preset and write its effective config and manifests on CPU. For comparison across output roots, repeat `--runs-dir`, passing one completed run per preset. Training resumes from the saved best epoch when the matching run directory is writable. Later unsaved epochs are repeated.

## Color correction

Both methods operate on decoded RGB before resize and padding. Gray World scales each channel mean toward the average of the three channel means. White Patch uses separate channel maxima and scales each toward 255. The implementation leaves an all-zero channel at zero and clips and rounds the corrected pixels to 8-bit RGB. Noor frames receive the selected correction consistently in training, validation, and test. Training then uses the current padding and augmentation pipeline.

The methods estimate channel gains from scene statistics. Actual tissue color and saturated highlights affect those statistics. The preview picks one training frame per Noor video up to the requested count and writes an original / Gray World / White Patch panel and a CSV of channel statistics and changed-pixel fractions. In the audited archive, every Noor frame already reaches 255 in each channel. Classic White Patch produces identical pixels on those 261 frames. [Finlayson and Trezzi, *Shades of Gray and Colour Constancy*](https://doi.org/10.2352/CIC.2004.12.1.art00008)

## Read the comparison

- `shared_farabi_0_2` uses the same 304 test frames from 86 Farabi videos for every preset. The closed three-class score selects the largest score among classes 0–2. The report also gives native accuracy and counts native class-3 predictions on those frames.
- `full_4_class` compares the baseline and the two color variants on the same 361 test frames. `noor_class_0` reports class-0 accuracy on the same 41 Noor test frames from nine videos.
- Use `farabi3` versus `pooled3` to measure the effect of adding the available Noor training examples under the same three-class task and Farabi evaluation cohort. The baseline versus `farabi3` comparison also changes the label space and the checkpoint selection cohort.
- `comparison_metrics.csv` contains matched-cohort accuracy and macro-F1. `comparison_deltas.csv` expresses differences as candidate minus reference in percentage points. `shared_farabi_predictions.csv` preserves the frame identities and both closed and native predictions.

Each training run writes its effective config, shared and filtered split manifests, source/class counts, training logs, best checkpoint, validation/test probabilities, learning curves, confusion matrices, per-class reports, and test-frame audit to `<output-dir>/<preset>/`. `archive_results.py` can archive a completed run for download.

## Project entry points

| File | Purpose |
|---|---|
| `experiments.py` | Data preparation, Noor preview, and saved-run comparison commands |
| `train.py` | Launch one preset on a GPU |
| `configs/severity_comparisons.yaml` | Shared training and preprocessing defaults |
| `cataract_swin/color_constancy.py` | Gray World and White Patch on decoded RGB |
| `cataract_swin/experiments.py` | Preset definitions and shared video split handling |
| `cataract_swin/comparison.py` | Matched held-out reports |
