# CNN whistle detector

VGG16 classifier for whistle/noise detection in 0.4-second spectrogram windows.
Training uses `dolphinteam/OpenWhistle-CNN`; inference and test-only evaluation
use the published `dolphinteam/OpenWhistle-CNN-VGG16` checkpoint by default.

## Organization

```text
cnn/
├── train.py                        # training and held-out evaluation
├── inference.py                    # predictions on WAV/FLAC recordings
├── create_sequences_whistles.py    # predictions to whistle intervals
├── requirements.txt
├── utils/                          # shared model, data, metrics and reports
├── learning_curve/                 # training-set size experiment
│   ├── run_learning_curve.py
│   └── results/                    # manuscript curve and per-run summaries
├── analysis/                       # false-negative acoustic analysis
└── external/                       # WMMSD/DCLDE presets and local preparation
```

Start with the three commands below. The [learning curve](learning_curve/README.md),
[false-negative analysis](analysis/README.md) and
[external datasets](external/README.md) have their own instructions.
Run the examples from the repository root. Use `--help` to inspect each CLI.

## Installation

Use Python 3.12. Python 3.13 is not recommended for this environment: some
scientific packages may not have compatible wheels for the pinned stack and pip
can fall back to building SciPy locally, which requires a Fortran compiler.

The recommended setup is a dedicated Python environment installed with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r cnn/requirements.txt
```

If you need a specific CUDA wheel, install PyTorch with the index URL matching
the machine before installing the requirements:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r cnn/requirements.txt
```

Use the CPU wheel if the GPU driver or card is not compatible:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r cnn/requirements.txt
```

If `torch.cuda.is_available()` is `False`, either install a PyTorch wheel built
for a CUDA version supported by the NVIDIA driver, or run the scripts with
`--cpu-only`.

## Train or evaluate

```bash
python cnn/train.py
python cnn/train.py --test-only --no-wandb-enabled
```

The second command evaluates the published checkpoint on the held-out test
split. Override the weights with `--checkpoint-path /path/to/model.pt`.
Training uses ImageNet initialization by default; its best checkpoint is selected
by validation loss. Weights & Biases logging is disabled by default.

`TRAIN_DATASET_SOURCE` overrides the dataset with a Hugging Face repo id or a
local `DatasetDict` saved with `save_to_disk`. Training defaults and environment
variables live in `utils/config.py`: batch size 4, 50 epochs, patience 10,
learning rate `1e-5`, seed 7. Use `--cpu-only` to run on CPU.

## Predict and extract intervals

```bash
python cnn/inference.py \
  --recordings-dir /path/to/recordings \
  --output-dir cnn/runs/inference \
  --recursive

python cnn/create_sequences_whistles.py \
  cnn/runs/inference \
  --output-csv cnn/runs/inference/whistle_sequences.csv
```

Inference writes per-recording prediction CSVs and a `detections.csv` summary.
Sequence extraction groups positive windows into intervals. It also accepts
Hugging Face CSV/Parquet inputs via `--source hf`; see its `--help` for options.

## Outputs and manuscript results

New checkpoints, caches, plots and reports go under ignored `cnn/runs/` by
default. Training paths are relative to the working directory; supply separate
`--models-dir`, `--figs-dir` and `--reports-dir` for independent runs.

Training writes `reports/run_summary.json`; test-only evaluation writes
`reports/test_summary.json`, `reports/test_roc_curve.csv` and
`reports/roc_summary.json`, with plots under `figures/`. These scores describe
windows, rather than complete recordings or whistle events. The positive class
is whistle (label 1); CNN F1 is the binary whistle-class F1.

The tracked learning-curve results are in
[learning_curve/results/](learning_curve/results/). Start with
[the summary CSV](learning_curve/results/learning_curve_summary.csv) or
[the figure](learning_curve/results/learning_curve_test_f1.png). Detailed
historical diagnostics are in the [local archive](../docs/results-organization.md).
Stored run metadata retains its original paths.

## Shared implementation

| File in `utils/` | Responsibility |
|---|---|
| `config.py` | Defaults, environment variables and training CLI |
| `data.py` | Dataset loading, split checks and data loaders |
| `model.py` | VGG16, spectrograms and normalization |
| `model_runtime.py` | Optimizer, scheduler and model setup |
| `metrics.py` | Epoch loops and metrics |
| `artifacts.py` | Checkpoints, plots and CSV/JSON reports |
| `runtime_utils.py` | Seeds, devices and session IDs |
| `wandb_logging.py` | Optional experiment logging |
