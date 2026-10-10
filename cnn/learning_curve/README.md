# CNN performance versus training-set size

Train the VGG16 whistle detector on nested subsets containing 5%, 10%, 25%,
50%, and 100% of the training sessions, with seeds 7, 17 and 27.
Each run uses ImageNet initialization and selects its best checkpoint by
validation loss. Validation and test splits stay fixed and session-disjoint.

## Protocol

Sessions are sorted, permuted with `numpy.random.default_rng(seed)`, and the
first `ceil(fraction * number_of_sessions)` sessions are retained. All windows
from a selected session are included. Subsets are nested within each seed.

Every run uses spectrogram inputs, batch size 4, up to 50 epochs, patience 10,
learning rate `1e-5` and AMP on GPU. The recipe is fixed across dataset sizes.
The primary metric is binary F1 for the whistle class on the test split.
Report the mean and sample standard deviation across three seeds.

The figure uses session fractions on the x-axis. The CSV also records the
actual number of windows and their duration in window-hours (0.4 s per window).
Window-hours are not the duration of the original continuous recordings.
This experiment measures performance under a fixed training recipe; it does
not optimize hyperparameters independently at each dataset size.

## Run

From the repository root in the CNN environment:

```bash
python cnn/learning_curve/run_learning_curve.py \
  --fractions 0.05 0.1 0.25 0.5 1.0 \
  --seeds 7 17 27 \
  --num-workers 8
```

Outputs default to `cnn/runs/learning_curve/`. Completed runs with matching
settings are skipped. `--dry-run` lists planned runs without training or
writing results; `--overwrite` reruns selected combinations.

## Rebuild the figure

From the per-run CSV, without training or downloading audio/checkpoints:

```bash
python cnn/learning_curve/run_learning_curve.py \
  --plot-only \
  --runs-csv cnn/runs/learning_curve/learning_curve_runs.csv
```

Alternatively, `--plot-only` without `--runs-csv` reads the individual reports
in the output directory. Match the settings used to create those reports:

```bash
python cnn/learning_curve/run_learning_curve.py --plot-only --num-workers 8
```

## Outputs

```text
cnn/runs/learning_curve/
├── protocol.json
├── learning_curve_runs.csv
├── learning_curve_summary.csv
├── learning_curve_test_f1.png
├── learning_curve_test_f1.pdf
└── fraction_*/seed_*/
    ├── run_metadata.json
    ├── models/model_best.pt
    ├── reports/
    └── figures/
```

`learning_curve_runs.csv` records scores, selected training-set sizes and best
epoch per seed. `learning_curve_summary.csv` records means and standard
deviations. `run_metadata.json` stores each run's training settings.

## Results

| Training sessions | Test whistle F1 (%) |
|---|---:|
| 5% (10 sessions) | 96.84 ± 0.48 |
| 10% (20 sessions) | 97.17 ± 0.18 |
| 25% (49 sessions) | 97.47 ± 0.22 |
| 50% (98 sessions) | 97.46 ± 0.21 |
| 100% (195 sessions) | 97.43 ± 0.15 |

Under the descriptive tolerance of one absolute F1 percentage point below
the full-data mean, the smallest tested subset (5% of sessions) meets the
criterion: its mean is 0.58 points below the full-data mean.
