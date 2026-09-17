# Nasal septum centreline segmentation and deviation measurement from CT

Biofluids Lab, IIT(ISM) Dhanbad. Coronal CT slices of the nasal cavity, with the
septum traced by hand in Label Studio. The goal is to segment the septum on each
slice and measure how far it bends away from a straight septum, as input to
pre/post-septoplasty flow studies.

## Data

- 899 coronal slices from 23 CT series, cropped around the nasal cavity at native resolution
  (132-183 x 193-257 px). Every slice has a hand-drawn septum trace.
- Scan identity is recovered from the crop geometry: each series was cropped once, so crop
  size identifies the series, and slices within a series stay in acquisition order
  (adjacent slices differ by 7.0 grey levels on average vs 19.3 for random slices of the same scan).
- Label Studio exports brush strokes as anti-aliased 1-px lines. Thresholding them at >127
  (as the 2024 pipeline did) leaves only 32% of traces connected and empties 80 masks;
  thresholding at >0 keeps 98% as one continuous line. Targets are therefore the traced
  centreline (skeleton of mask > 0), dilated to a 3-px band for training.

### Data availability

The CT crops and annotations are not included in this repository. They can be shared for
research use on request; please open an issue or contact the author.

## Method

- 5-fold cross-validation **grouped by scan**: no series contributes slices to both training
  and test. Every slice is predicted exactly once, by a model that never saw its scan.
- Slices are zero-padded to 272 x 272 without resampling, so a 1-px structure is not blurred.
- Augmentation: horizontal flip, small affine (±10°, ±10% scale, ±5% shift), brightness/contrast.
- AdamW, 100 epochs, warm-up + cosine schedule, bfloat16 autocast. The last-epoch model is
  evaluated; no checkpoint is chosen on the test fold.

Ablation:

| | Model | Loss | Input |
|---|---|---|---|
| A | U-Net from scratch (the 2024 architecture) | BCE | slice |
| B | U-Net from scratch | BCE + Dice | slice |
| C | U-Net, ImageNet ResNet34 encoder | BCE + Dice | slice |
| D | U-Net, ImageNet ResNet34 encoder | BCE + Dice + clDice | slice |
| E | U-Net, ImageNet ResNet34 encoder | BCE + Dice + clDice | 2.5D (previous / current / next slice) |

## Metrics

A 1-px shift halves the overlap of a thin structure, so overlap alone misleads. Reported:

- **Centreline F1 @ 2 px**: share of predicted centreline within 2 px of the traced one
  (precision) and vice versa (recall).
- **clDice** (Shit et al., CVPR 2021), **mean / 95th-percentile centreline distance**.
- **Dice on the 3-px band**, for reference.
- **Septal deviation**: the centreline is reduced to one x per row, smoothed over 5 rows, and
  the largest perpendicular distance from the chord joining its ends is taken (px, and % of
  septum length). Agreement between deviation measured on predictions and on the traces is
  reported per slice (MAE, Pearson r, Bland-Altman) and per scan (maximum over slices).

## Results

See `results/summary.md` and `results/figures/`.

## Reproduce

```bash
python prepare_data.py                       # data/final-v4 -> data/
python train.py --config configs/d_resnet34_cldice.yaml --fold 0
scripts/run_queue.sh <gpu> scripts/jobs_s0a.txt   # a queue of (config, fold) jobs
python aggregate.py                          # results/summary.md, results/figures/
```

Environment: Python 3.11, PyTorch 2.6 (CUDA 12.4), segmentation-models-pytorch 0.5,
scikit-image, SciPy, albumentations.

## Limitations

- 23 series; the number of distinct patients is not recorded in the crops (likely fewer than 23),
  so folds are grouped by series, not verified by patient.
- No DICOM headers survive the crops, so distances are in pixels, not mm.
- One annotator; inter-observer variability is not measured.
