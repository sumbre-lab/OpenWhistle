# False-negative analysis

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
