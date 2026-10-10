# Dataset overview figures

This folder builds the two **OpenWhistle** manuscript figures that summarize the pretraining corpus and the classification finetuning subset. Outputs are written to `output/` (PNG and PDF).

## Organization

```text
datasets_figures/
├── plot_dataset_overview.py          # recording coverage and CNN evaluation
├── plot_classification_overview.py   # whistle properties, including SNR
├── data/                            # input statistics for the manuscript
│   ├── pretraining_recording_hours.csv
│   ├── inter_detected_whistle_intervals.csv
│   └── audio_segment_durations.csv
├── scripts/                         # data preparation and drawing helpers
│   ├── paths.py
│   ├── sidecars.py
│   ├── pretraining_dataset_overview.py
│   └── classification_overview.py
└── output/                          # two figures, each in PNG and PDF
```

## Scripts

Use the [CNN Python environment](../cnn/README.md#installation). From the
repository root:

```bash
python datasets_figures/plot_dataset_overview.py
python datasets_figures/plot_classification_overview.py
```

Before rendering the classification figure, compute SNR once:

```bash
python cnn/analysis/plot_snr.py --workers 16
```

Panel F reads `cnn/runs/snr/` by default (`--snr-dir` overrides it). Rendering
uses the completed CSVs and protocol, without recalculating SNR or decoding
audio. Diagnostic subsets and incomplete CSVs are rejected. Classification
metadata for panels C–E is streamed from the dataset revision recorded in the
SNR protocol, projecting only label, name and duration.

Other auxiliary statistics come from `data/`. Panel B uses the existing
`audio_segment_durations.csv` when available; otherwise it queries the HF
pretraining dataset. Panel F uses only the completed `plot_snr.py` outputs.

Each classification figure writes `fig_classification_overview_sources.json`
with the SNR protocol, dataset revisions, valid counts and input CSV hashes.

## Input data

| Input | Used for |
|---|---|
| `data/pretraining_recording_hours.csv` | Dataset overview A–C: recording dates, durations and detected-whistling durations |
| `cnn/runs/reports/test_confusion_matrix.csv` | Dataset overview D: held-out CNN evaluation |
| `data/inter_detected_whistle_intervals.csv` | Classification overview A: gaps between detected whistle sequences |
| `data/audio_segment_durations.csv` | Classification overview B: pretraining sequence durations |
| HF classification dataset, `all` config, all three splits | Classification overview C–E: dates, classes and durations |
| `cnn/runs/snr/` | Classification overview F: full-corpus SNR measurements and protocol |

The recording-hours and interval CSVs are supplied statistics from the full
recordings and their detections. The public pretraining audio segments alone
do not reconstruct recording coverage or gaps between sequences. Rendering
does not rerun detection on the raw recordings. Missing CNN reports or required
interval statistics produce an error rather than an incomplete figure.

`--output-dir` selects the export directory. `--data-dir` overrides the
classification figure's auxiliary data directory; `--recording-csv` overrides
the dataset overview's recording statistics. Defaults are anchored to this
checkout, so both plot scripts can also run from another working directory.

## Prepare auxiliary statistics

The supplied `data/` CSVs are sufficient for normal rendering. To regenerate
durations or intervals, use the helper commands:

```bash
python datasets_figures/scripts/sidecars.py audio-durations
python datasets_figures/scripts/sidecars.py iwi --hf-id org/predictions-dataset
```

The interval input must contain `file_name`, `initial_point` and `finish_point`.
The helper merges intervals separated by at most 0.05 seconds and exports
positive gaps shorter than 20 seconds. Use repeated `--hf-id` arguments for
multiple prediction datasets. These are prediction tables, not the published
pretraining audio dataset.

`--refresh-data` refreshes supported auxiliary CSVs; its optional dataset IDs
and filenames are configured through the environment variables documented in
`scripts/sidecars.py`. SNR is computed separately with `cnn/analysis/plot_snr.py`.

## Figure captions

### `fig_dataset_overview`

**OpenWhistle: Longitudinal Extent and Temporal Distribution of the Dataset.**

- **A)** Cumulative recording hours over time, showing dataset growth and changes in pod composition.
- **B)** Distribution of recording hours across the day, indicating alignment with periods of human activity.
- **C)** Cumulative detected whistling hours over time, obtained by applying the whistle presence detection CNN to the raw recordings.
- **D)** Row-normalized confusion matrix (%) of the whistle presence detection CNN on the held-out test set.

### `fig_classification_overview`

**Analyses of Whistle Properties in OpenWhistle.**

- **A–B)** Temporal structure of dolphin vocalizations: distributions of inter-whistle intervals (A) and whistle sequence durations (B).
- **C–E)** Properties of the expert-annotated subset: temporal coverage (C), class distribution (D), and whistle duration (E).
- **F)** Estimated signal-to-noise ratio (SNR) of the published pretraining
  sequences (left, 31,780 rows) and the classification dataset, `all` config
  across train, validation and test (right, 8,353 valid rows). Each audio row
  contributes one spectrogram-ridge estimate using noise-subtracted flanking
  bands. One classification clip is too short for estimation and is excluded.
  Dashed lines within the violins indicate quartiles. The displayed range is
  −10 to 36 dB; values outside it remain in the source CSVs. The horizontal
  3/6/10 dB guides retain the original figure's visual convention.

## CNN confusion matrix

Panel D uses `cnn/runs/reports/test_confusion_matrix.csv` from evaluation
of the published `dolphinteam/OpenWhistle-CNN-VGG16` checkpoint.
Generate the evaluation reports and figure with:

```bash
python cnn/train.py --test-only --no-wandb-enabled
python datasets_figures/plot_dataset_overview.py
```

Use `--confusion-csv /path/to/test_confusion_matrix.csv` to select another
matrix explicitly.
