# Inference on external datasets

These tools reuse [the main inference implementation](../inference.py) with
WMMSD/Watkins and DCLDE directory presets. They do not change the CNN model.

| File | Purpose |
|---|---|
| `inference_conf.py` | Dataset presets and environment overrides |
| `run_inference_dataset.py` | Run inference using one or more presets |
| `setup_external_inference.py` | Prepare local paths and optionally the Python environment |

From the repository root, in the CNN environment:

```bash
export OPENWHISTLE_CNN_WMMSD_DIR=/path/to/wmmsd
export OPENWHISTLE_CNN_DCLDE_DIR=/path/to/dclde
python cnn/external/run_inference_dataset.py wmmsd dclde
```

A preset accepts a raw audio folder or a prepared folder containing
`flat_recordings/` and optionally `specific_files.txt`. Defaults point to
`~/Documents/DolphinWhistleExtractor/benchmark_data/`; override them with the
variables above or `DOLPHIN_WHISTLE_EXTRACTOR_ROOT`.
Outputs default to `cnn/runs/external_inference/`, resolved from the repository
root. Override them with `--output-root`.

If the separate `DolphinWhistleExtractor` checkout is available, the setup tool
can connect its prepared folders:

```bash
python cnn/external/setup_external_inference.py --skip-deps
source cnn/.external_inference.env
python cnn/external/run_inference_dataset.py wmmsd dclde
```

Without `--skip-deps`, setup creates/reuses `.venv` and installs the CNN
requirements. Its `--prepare-wmmsd-hf` and `--download-dclde` options invoke
preparation tools from the separate checkout. See `--help` before using them.

For a small local-checkpoint run:

```bash
python cnn/external/run_inference_dataset.py wmmsd \
  --limit 1 --cpu-only --checkpoint-path /path/to/model.pt
```

The generated `cnn/.external_inference.env`, audio and predictions remain local.
