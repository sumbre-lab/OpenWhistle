# Audio-derived SNR figure

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

These are estimated band SNRs, not reference SNRs obtained from isolated clean
signals. The method and population differ from the historical CREPE-based CSVs:
pretraining now uses published sequences rather than the old 4,000 sampled
windows. The script ignores both local manuscript CSVs and any HF `snr_db` or
F0 columns; it recomputes from audio in both groups. Numerical agreement with
the original panel F is not claimed.

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
