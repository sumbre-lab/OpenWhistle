# CNN Training

This folder contains the Torch/VGG16 CNN training code used for binary dolphin
whistle detection.

## Files

- `train.py`: training entry point and run orchestration.
- `inference.py`: command-line inference on WAV/FLAC recordings.
- `create_sequences_whistles.py`: post-process inference CSVs into whistle
  sequence intervals.
- `utils/config.py`: training defaults, environment overrides, and CLI parsing.
- `utils/data.py`: Hugging Face dataset loading, split checks, and Torch datasets.
- `utils/metrics.py`: epoch loop and classification metrics.
- `utils/artifacts.py`: checkpoints, plots, and CSV/JSON reports.
- `utils/model.py`: VGG16 model and spectrogram utilities.
- `requirements.txt`: Python dependencies needed by this extraction.

## Dataset

By default, training reads the public Hugging Face dataset:

```bash
OpenWhistleNeurIPS26/OpenWhistle-CNN
```

Override it with `TRAIN_DATASET_SOURCE`. The value can be either a Hugging Face
dataset repo id or a local `datasets.DatasetDict` saved with `save_to_disk`.

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

## Example

```bash
git clone https://github.com/OpenWhistleNeurIPS26/OpenWhistle.git
cd OpenWhistle
python cnn/train.py
```

Inference on a folder of recordings:

```bash
python cnn/inference.py \
  --recordings-dir /path/to/recordings \
  --output-dir /path/to/predictions
```

For nested external recording folders, use recursive discovery:

```bash
python cnn/inference.py \
  --recordings-dir /path/to/external_dataset \
  --output-dir cnn/runs/inference_external \
  --recursive
```

External dataset presets are managed in `cnn/inference_conf.py`. To run the CNN
inference with those presets:

```bash
python cnn/run_inference_dataset.py wmmsd dclde
```

Create whistle sequences from an inference output directory. Local paths are
detected automatically, so `--source local` is optional:

```bash
python cnn/create_sequences_whistles.py \
  cnn/runs/inference \
  --output-csv cnn/runs/inference/whistle_sequences.csv
```

Create whistle sequences from inference CSVs or a tabular Parquet split stored
in a Hugging Face dataset repo. Use `--source hf` when the input is a Hugging
Face repo id:

```bash
python cnn/create_sequences_whistles.py \
  organization/private-inference-dataset \
  --source hf \
  --hf-split train \
  --output-csv cnn/runs/sequences/hf_sequences.csv
```

By default, inference downloads and uses:

```bash
OpenWhistleNeurIPS26/OpenWhistle-CNN-VGG16
```

Use `--checkpoint-path /path/to/model.pt` only when evaluating a local
checkpoint.

Evaluate the published model on the test split only:

```bash
python cnn/train.py --test-only --no-wandb-enabled
```

This uses `OpenWhistleNeurIPS26/OpenWhistle-CNN` split `test` and downloads the
default checkpoint from `OpenWhistleNeurIPS26/OpenWhistle-CNN-VGG16`.

Useful environment variables:

- `TRAIN_BATCH_SIZE` default: `4`
- `TRAIN_NUM_EPOCHS` default: `50`
- `TRAIN_PATIENCE` default: `10`
- `TRAIN_LEARNING_RATE` default: `1e-5`
- `TRAIN_INPUT_SOURCE` default: `spectrogram`, can be `audio` to regenerate
  spectrograms from waveform payloads.
- `TRAIN_PRETRAINED_BACKBONE` default: `1`
- `TRAIN_FREEZE_BACKBONE` default: `0`
- `TRAIN_NORMALIZATION_MEAN`/`TRAIN_NORMALIZATION_STD`: defaults are the
  torchvision ImageNet normalization used by pretrained image backbones.
- `TRAIN_CPU_ONLY` default: `0`
- `TRAIN_EVAL_ONLY` default: `0`
- `TRAIN_CHECKPOINT_PATH` default: the run's `model_best.pt`
- `WANDB_ENABLED` default: `0`; use `--wandb-enabled` to log a run.

The script writes checkpoints, figures, and reports under `cnn/runs`
relative to the current working directory by default.
