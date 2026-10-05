# Watkins transfer evaluation

The rebuttal calls the Watkins metric macro-F1, but its recorded 76.99% and
78.47% scores are accuracy. This runner selects C using validation **macro-F1**
and evaluates the selected train-fitted classifier once on test.

## Results from historical embedding caches

Recomputed on 2026-10-05, using 1,695 clips and 31 classes:

| Model cache | Selected C | Test macro-F1 | Bootstrap standard deviation |
|---|---:|---:|---:|
| OpenWhistle Wav2Vec2 | 3 | 75.88% | 2.71 percentage points |
| AVES-bio | 1 | 77.58% | 2.73 percentage points |

These are **new probe results on existing embeddings**, not a reproduction of
audio extraction or a guarantee that the caches use the final intended model
and preprocessing. The old JSON files are preserved as `historical_metrics.json`.
With accuracy selection in this environment, AVES reproduces its old metrics
exactly. Wav2Vec2 selects the same C=3 but differs by one correct test prediction
(76.6962% versus 76.9912% accuracy; 75.8768% versus 76.0820% macro-F1).
No historical test predictions or complete historical dependency lock were
found, so the precise cause of that difference has not been established.
Other installed environments also changed validation scores and selected C;
do not combine their scores with this run.

## Protocol

- Fixed cached split: train 1,017, validation 339, test 339 (60/20/20).
- All 31 classes appear in each split. Exact ordered manifests are exported.
- StandardScaler fitted on train; multinomial logistic regression with lbfgs,
  `class_weight="balanced"`, `max_iter=20000`, `tol=1e-4`.
- C grid: 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100.
- Select highest validation macro-F1; ties choose the first C in this ascending
  grid. Neither scaler nor classifier is refitted on train + validation.
- Report the full-test macro-F1 as the point estimate. The ± value is the
  population standard deviation (`ddof=0`) of 1,000 unstratified resamples of
  339 test clips, with replacement, using NumPy default_rng seed 42.
  Average F1 over the same 31 classes in every resample, with `zero_division=0`.
  It measures test sampling uncertainty, not variation across training seeds.
  The bootstrap mean and percentile interval are saved separately.

This retains the historical Watkins probe protocol except for selection by
macro-F1. It differs from the OpenWhistle benchmark's C grid, class weighting,
split proportions and refit policy. Dependency versions also differ from the
unknown historical environment.

## Recalculate from caches

From the repository root, in a separate Python 3.13 environment:

```bash
python -m pip install -r experiments/benchmark/requirements-watkins.txt
python experiments/benchmark/src/train_lr_watkins.py \
  --cache-root /path/to/beans_watkins_classification \
  --output-dir /tmp/watkins-recomputed \
  --num-bootstrap 1000 --seed 42 --threads 2
```

The cache root must contain `dolphinteam_OpenWhistle_Wav2Vec2.0/` and
`aves_bio_rerun/`, each with an aligned `embeddings.npy` matrix and
`embedding_index.csv` containing `audio_path,label,split`. Optionally include
the historical `metrics.json` to enable the comparison. `--models` overrides
the directory names. Inputs are read only; embeddings are not copied into Git.

The outputs include test predictions, portable split manifests, the complete C
sweep, bootstrap scores, metric summaries, input SHA-256 hashes and environment
provenance. Audio files are not required to rerun the cached probe.

To check the reported uncertainty directly from the committed predictions:

```bash
PYTHONPATH=experiments/benchmark/src python - <<'PY'
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from train_lr_watkins import bootstrap_macro_f1

root = Path('experiments/benchmark/results/watkins_macro_f1')
for model in ('dolphinteam_OpenWhistle_Wav2Vec2.0', 'aves_bio_rerun'):
    manifest = pd.read_csv(root / model / 'split_manifest.csv')
    test = pd.read_csv(root / model / 'test_predictions.csv')
    encoder = LabelEncoder().fit(manifest.label)
    values = bootstrap_macro_f1(encoder.transform(test.label),
                               encoder.transform(test.prediction),
                               range(len(encoder.classes_)), 1000, 42)
    print(model, 100 * values.std(ddof=0))
PY
```

## Still required for audio-to-score reproduction

Verify the cache provenance against the intended checkpoint revisions and
extraction settings, including sample rate, clip truncation and pooling.
Adapt and validate the external extraction runner, provide a data acquisition
recipe and versioned dependencies, and reproduce these results starting from
audio. The original local runner uses 16 kHz and up to 10 seconds; those
settings alone do not establish the provenance of both saved caches.
