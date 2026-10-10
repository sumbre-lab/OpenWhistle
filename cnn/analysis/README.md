# CNN error analysis and audio-derived SNR

Run the commands below from the repository root in the CNN environment.

## SNR figure

```bash
python cnn/analysis/plot_snr.py
```

The script computes SNR directly from the audio of the datasets in the
[OpenWhistle collection](https://huggingface.co/collections/dolphinteam/neurips26-openwhistle):

- **Pretraining**, left: `dolphinteam/OpenWhistle-Pretraining`, `default` config,
  train and validation splits.
- **Classification (all)**, right:
  `dolphinteam/OpenWhistle-Classification-Finetuning`, `all` config,
  train, validation and test splits (all ten classes).

Each published audio row contributes one SNR estimate. The default processes all
rows, streamed with a bounded queue; it does not load the full corpus into RAM.
Audio decoding and SNR calculations run concurrently on up to 16 CPU workers
by default. `--workers 1` runs sequentially; `--workers 8` reduces CPU and memory
use. The GPU is not used by this SciPy-based estimator. Parallel execution keeps
the same scores and row order as sequential execution. The full
run reads a large amount of audio and can take substantial time. A diagnostic
using the first three rows of each dataset can be run separately:

```bash
python cnn/analysis/plot_snr.py --limit 3 --output-dir cnn/runs/snr-smoke
```

This limited run is not representative of the full datasets. Dataset revisions
can be pinned with `--pretraining-revision` and `--classification-revision`.

## SNR calculation

`snr.py` decodes embedded audio with SoundFile and averages stereo channels.
Both datasets use the same spectrogram-ridge estimator: Blackman window,
2048-point FFT, 256-sample hop, 3–25 kHz band (bounded by the native Nyquist
frequency). A background-subtracted, median-filtered ridge locates the dominant
frequency. Signal energy is pooled within ±300 Hz of the ridge; noise is estimated
in adjacent 2 kHz bands starting 400 Hz from it. Its estimated contribution to
the signal band is subtracted before computing the dB ratio.

Silent, too-short or unmeasurable clips receive an explicit invalid status and
are excluded from the violins; their counts remain in the protocol. Unexpected
decoding/download errors stop the run rather than silently dropping audio.

These are estimated band SNRs, calculated from each published audio row.
The script recomputes from audio in both groups, using the same estimator.
Existing HF `snr_db` and F0 columns are not used.

## Outputs

Under ignored `cnn/runs/snr/` (override with `--output-dir`):

- `snr_pretraining_vs_classification_all.png` and `.pdf`;
- `snr_pretraining.csv` and `snr_classification_all.csv`, computed scores,
  row indices, split names, sampling rates, durations and statuses;
- `snr_protocol.json`, exact HF commit hashes, subsets/splits, estimator settings,
  processed counts, invalid counts and any diagnostic limit.

The plot reuses panel F's visual style and 3/6/10 dB guides, with a displayed
range of −10 to 36 dB. Values outside the display range remain in the score CSVs.
To rebuild only the figure from a completed computation:

```bash
python cnn/analysis/plot_snr.py --plot-only
```

A normal invocation recomputes the scores; `--plot-only` requires this script's
CSV files and protocol in the chosen output directory, and uses no network.

## False-negative analysis

Evaluate the published CNN checkpoint on the `dolphinteam/OpenWhistle-CNN`
test split and describe the whistle clips it detects or misses:

```bash
python cnn/analysis/analyze_missed_whistles.py
```

The default decision threshold is 0.5. Use `--checkpoint-path /path/to/model.pt`
to evaluate another checkpoint, or `--cpu-only` to run inference on CPU.

Outputs go to `cnn/runs/analysis/`, anchored to the checkout (override with
`--output-dir`):

- `test_whistle_measures.csv`: predictions, scores and acoustic measures for
  each whistle-labelled test clip;
- `missed_whistles_summary.json`: confusion counts over all test rows and
  summaries of detected and missed whistles;
- `figures/snr_and_score_violin.png`;
- `figures/frequency_descriptors_violin.png`;
- `figures/miss_rate_by_whistle_type.png`;
- `figures/miss_rate_by_recording.png`.

This command evaluates the CNN; `plot_snr.py` generates the dataset SNR
distributions described above.

## Include SNR in the manuscript figure

After a completed full-corpus run:

```bash
python datasets_figures/plot_classification_overview.py
```

Panel F uses these same computed CSVs: Pretraining on the left and
Classification (all) on the right. Use `--snr-dir /path/to/completed/snr` to
select another completed run. The figure rejects diagnostic/partial inputs
and records its sources beside the PNG/PDF. Other panels use classification
metadata from the same HF revision and the existing auxiliary statistics.
