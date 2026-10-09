# False-negative analysis

## Manuscript SNR panel

To render panel F with **Pretraining** on the left and **Classification (all)**
on the right, without downloading a model or running inference:

```bash
python cnn/analysis/analyze_missed_whistles.py \
  --plot-snr-only --output-dir cnn/runs/analysis
```

This uses `datasets_figures/data/snr_detection_windows.csv` (pretraining windows)
and `datasets_figures/data/snr_classification.csv` (all classification splits
and classes), keeping finite SNR values with `status=ok`. The former is a sample
of 4,000 windows, rather than every pretraining recording. These sidecars use
CREPE frequency tracking; the false-negative diagnostic below estimates SNR
using a spectrogram ridge, so its values belong to a different analysis.

Outputs are `figures/snr_pretraining_vs_classification_all.png` and `.pdf`.
The style and 3/6/10 dB reference lines reuse the manuscript plotting helper,
including its displayed range of −10 to 36 dB. Override inputs with
`--pretraining-snr-csv` and `--classification-snr-csv`.

## CNN errors

`analyze_missed_whistles.py` evaluates the published detector on the held-out
`test` split of `dolphinteam/OpenWhistle-CNN` and compares detected and missed
whistles using SNR estimates, frequency and spectral measures.

From the repository root, in the CNN environment:

```bash
python cnn/analysis/analyze_missed_whistles.py \
  --output-dir cnn/runs/analysis
```

Use `--checkpoint-path /path/to/model.pt` for local weights, `--cpu-only` for
CPU execution, or `--limit` for a partial diagnostic. A limited run is not the
full test-set result. Check `--help` for the complete options.

Generated CSVs, JSON summaries and figures stay in ignored `cnn/runs/analysis/`.
The presence of this script does not establish that a particular manuscript
result has been reproduced; see the
[reproducibility audit](../../docs/provenance/reproducibility-audit.md).
