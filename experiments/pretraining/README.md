# Wav2Vec2.0 Pretraining

This folder contains the self-supervised Wav2Vec2.0 pretraining code used for
the OpenWhistle NeurIPS paper.

The training source is the public Hugging Face corpus:

```text
dolphinteam/OpenWhistle-Pretraining
```

The resulting checkpoint corresponds to:

```text
dolphinteam/OpenWhistle-Wav2Vec2.0
```

## Installation

Use Python 3.12 and install the pretraining dependencies with pip:

```bash
python -m pip install --upgrade pip
python -m pip install torch transformers datasets submitit julius numpy pandas
```

From the repository root, expose the pretraining package:

```bash
export PYTHONPATH="$PWD/experiments/pretraining:$PYTHONPATH"
```

If launching on SLURM, set the cluster fields used by `submitit`:

```bash
export OPENWHISTLE_SLURM_PARTITION=your_gpu_partition
export OPENWHISTLE_SLURM_ACCOUNT=your_slurm_account
```

Set a Hugging Face token only if your environment needs authenticated dataset
access:

```bash
export HF_TOKEN=hf_your_token
```

## Run

Smoke test on the small review split:

```bash
python experiments/pretraining/ANNpretraining/pretraining/submit_dolphin.py \
  --path_data dolphinteam/OpenWhistle-Pretraining \
  --path_data_config review-sample \
  --output_dir experiments/pretraining/artifacts/outputs/review_sample \
  --nb_nodes 1 \
  --nb_gpu 1 \
  --timeout_min 60
```

Full pretraining run:

```bash
python experiments/pretraining/ANNpretraining/pretraining/submit_dolphin.py \
  --path_data dolphinteam/OpenWhistle-Pretraining \
  --path_data_config default
```

## Outputs

By default, checkpoints and logs are written under:

```text
experiments/pretraining/artifacts/outputs/pretraining_run/
```

Audio is loaded from Hugging Face and cast to the 44.1 kHz sampling rate defined
in `ANNpretraining/models/wav2vec2/config/preprocessor_dolphin.json`.
