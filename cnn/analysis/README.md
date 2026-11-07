# CNN error analysis and audio-derived SNR

Run the commands below from the repository root in the CNN environment.

## SNR figure

```bash
python cnn/analysis/compute_f0_crepe.py   # step 1: CREPE F0 tracks (GPU recommended)
python cnn/analysis/plot_snr.py           # step 2: SNR around the tracks (CPU) and figure
```

The two steps compute SNR directly from the audio of the datasets in the
[OpenWhistle collection](https://huggingface.co/collections/dolphinteam/neurips26-openwhistle):

- **Pretraining**, left: `dolphinteam/OpenWhistle-Pretraining`, `default` config,
  train and validation splits.
- **Expert-annotated set**, right:
  `dolphinteam/OpenWhistle-Classification-Finetuning`, `unbalanced` config
  (formerly `all`), train, validation and test splits (all ten classes).

Each published audio row contributes one SNR estimate. Both steps process all
rows, streamed with a bounded queue; they do not load the full corpus into RAM.

`compute_f0_crepe.py` downloads the CREPE weights once (see below), then tracks
`--workers 4` clips concurrently: noise reduction and resampling run on the CPU,
the network on the GPU (`--device cpu` works, much more slowly). On an RTX 2080 Ti
tracking takes about 0.1 s per second of audio, i.e. several hours for the
pretraining corpus. `plot_snr.py` then decodes the audio and computes SNR on up to
16 CPU workers (`--workers`), without a GPU. Parallel execution keeps the same
scores and row order as sequential execution. It refuses tracks computed for a
different dataset revision, config or row subset.

A diagnostic using the first three rows of each dataset can be run separately:

```bash
python cnn/analysis/compute_f0_crepe.py --limit 3 --output-dir cnn/runs/f0_crepe-smoke
python cnn/analysis/plot_snr.py --limit 3 --f0-dir cnn/runs/f0_crepe-smoke --output-dir cnn/runs/snr-smoke
```

This limited run is not representative of the full datasets. Dataset revisions
can be pinned with `--pretraining-revision` and `--classification-revision`
(both scripts). `python cnn/analysis/plot_snr.py --f0 ridge` skips step 1 and uses
the model-free spectrogram-ridge estimator described below.

## SNR calculation

`snr.py` decodes embedded audio with SoundFile and averages stereo channels.

**F0 track (`crepe_f0.py`).** Whistle F0 is tracked with the bottlenose-dolphin
CREPE model of
[mim-team/bioacoustic_F0_estimation](https://github.com/mim-team/bioacoustic_F0_estimation)
(`model_only-0_bottlenose_dolphins.pth`, pinned to commit `b57cc78` and checked by
sha256; cached under `~/.cache/openwhistle/crepe/`, or pass `--weights`). The audio
is noise-reduced (stationary `noisereduce`), resampled to 320 kHz and presented to
CREPE as 16 kHz audio (20× frequency compression), with a 5 ms step and
weighted-argmax decoding. torchcrepe's random pitch dither is drawn from a fixed
seed so tracks are reproducible. Clips longer than 120 s are tracked in equal
chunks of at most 120 s.

**SNR (`estimate_snr_with_f0`).** The track is interpolated onto a centred Hann
STFT (2048-point FFT, 256-sample hop, 3–25 kHz band). Only frames where CREPE's
confidence is at least 0.05 and the F0 lies in the band count, so the silence
between whistles does not dilute the estimate. In those frames, signal energy is
pooled within ±300 Hz of the track; noise is estimated in 2 kHz bands starting
400 Hz from it on either side, and its expected contribution to the signal band is
subtracted before computing the dB ratio. This reproduces DolphinDataset's
`estimate_snr_db_with_time_series_f0`, which produced the HF `snr_db` column: on
100 classification test clips the median difference is −0.01 dB (median
|difference| 0.02 dB, from the unseeded dither of the published run). Clips
whose signal does not exceed the noise are floored at −120 dB.

**Ridge alternative (`--f0 ridge`, `estimate_snr`).** A background-subtracted,
median-filtered spectrogram ridge (Blackman window) replaces the CREPE track, and
energies are pooled over all frames of the clip, including the gaps between
whistles. On pretraining sequences this lowers the estimate by about 2.5 dB
relative to the CREPE-gated SNR.

Silent, too-short or unmeasurable clips receive an explicit invalid status (for
example `no_confident_f0` when CREPE finds no whistle) and are excluded from the
violins; their counts remain in the protocol. Unexpected decoding/download errors
stop the run rather than silently dropping audio.

These are estimated band SNRs, recalculated from each published audio row with
the same estimator for both groups. The HF `snr_db` and F0 columns are not read.

## Outputs

Under ignored `cnn/runs/f0_crepe/` (`compute_f0_crepe.py --output-dir`; read by
`plot_snr.py --f0-dir`):

- `f0_pretraining.parquet` and `f0_classification_all.parquet`: split, row index,
  sampling rate, duration, status and the float32 CREPE track (`f0_time`, `f0_hz`,
  `f0_conf`) of every audio row, in published row order;
- `f0_protocol.json`: HF commit hashes, subsets/splits, CREPE settings, weights
  URL and sha256, device, package versions, counts and Parquet sha256 hashes.

Under ignored `cnn/runs/snr/` (override with `--output-dir`):

- `snr_pretraining_vs_classification_all.png` and `.pdf`;
- `snr_pretraining.csv` and `snr_classification_all.csv`, computed scores,
  row indices, split names, sampling rates, durations and statuses;
- `snr_protocol.json`, exact HF commit hashes, subsets/splits, F0 method,
  estimator settings, the F0-track protocol, processed counts, invalid counts and
  any diagnostic limit.

The plot reuses panel F's visual style and 3/6/10 dB guides, with a displayed
range of −10 to 36 dB. Values outside the display range remain in the score CSVs.
To rebuild only the figure from a completed computation:

```bash
python cnn/analysis/plot_snr.py --plot-only
```

A normal invocation recomputes the scores; `--plot-only` requires this script's
CSV files and protocol in the chosen output directory (computed with the same
`--f0` choice), and uses no network.

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
the expert-annotated set on the right. Use `--snr-dir /path/to/completed/snr` to
select another completed run. The figure rejects diagnostic/partial inputs
and records its sources beside the PNG/PDF. Other panels use classification
metadata from the same HF revision and the existing auxiliary statistics.
