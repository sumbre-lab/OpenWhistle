# OpenWhistle: Self-Supervised Pretraining

Native Fairseq training recipe for
[OpenWhistle wav2vec 2.0](https://huggingface.co/dolphinteam/OpenWhistle-Wav2Vec2.0):
44.1 kHz audio, stride 960, 12 Transformer layers, hidden size 768 and 8 attention
heads. The code prepares the public audio dataset, trains from scratch and
exports checkpoints to Transformers.

[Installation](#installation) · [Data](#data-preparation) ·
[Pretraining](#pretraining) · [Protocol](#training-protocol) · [Export](#export)

## Installation

Create and activate a separate **Python 3.10** environment:

```bash
conda create -n openwhistle-pretraining python=3.10 pip=23.3.2 -y
conda activate openwhistle-pretraining
```

Install matching **PyTorch 2.0.1** and **TorchAudio 2.0.2** builds for your
CPU/CUDA setup before the commands below. For a CPU-only smoke test:

```bash
python -m pip install 'torch==2.0.1+cpu' 'torchaudio==2.0.2+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
```

The benchmark and fine-tuning environments use newer dependencies and are separate.

Run all commands from the repository root:

```bash
python -m pip install 'pip==23.3.2' 'setuptools==68.2.2' 'wheel==0.41.3' \
  'cython==0.29.36' 'numpy==1.23.5'
MAX_JOBS=2 env -u CUDA_HOME python -m pip install --no-build-isolation \
  -r experiments/pretraining/requirements.txt
```

The requirements pin Fairseq v0.12.2 to the complete official source archive at
commit `4a388e64cd646ed7d7ad8de8fae55df2b8eea91d`, with a SHA-256 check.
The PyPI source package omits a required C++ file. A C++ compiler is required;
unsetting `CUDA_HOME` disables obsolete optional translation extensions while
retaining PyTorch CUDA support for Wav2Vec2. Fairseq is licensed under MIT.

## Data preparation

```bash
python experiments/pretraining/src/prepare_data.py \
  --streaming --dataset_config default \
  --revision eea37423c7841b0f99a7c0380f945999c0a2c489
```

The default output is `runs/pretraining/data/`: mono 44,100 Hz PCM_16 FLACs under
`audio/`, Fairseq manifests under `manifests/`, and metadata recording the dataset
revision. Use `--audio_dir` and `--manifest_dir` for separate datasets. Existing
audio is checked against the expected headers and waveform samples; manifests
are not overwritten without `--overwrite`.

The full corpus contains 28,410 training and 3,370 validation examples; Fairseq
excludes two training examples shorter than five seconds. For a small trial,
use `--dataset_config review-sample --max_train 8 --max_validation 4` with
separate audio and manifest directories.

## Pretraining

```bash
python experiments/pretraining/src/run_pretraining.py \
  --output_dir runs/pretraining/reproduction
```

The launcher reads `runs/pretraining/data/manifests/` by default. Use `--data_dir`
for another manifest directory. Without `--output_dir`, it creates a run under
`runs/pretraining/run-YYYYMMDD-HHMMSS/`. Checkpoints and logs stay in that local
run directory, ignored by Git.

Add `--dry_run` to inspect the command without loading Fairseq or creating files.
Fairseq resumes `checkpoint_last.pt`, including optimizer and schedule, when it
exists in the chosen output directory. Use a new directory for a fresh run.

To distribute training over four GPUs on one machine with accumulation:

```bash
python experiments/pretraining/src/run_pretraining.py \
  --output_dir runs/pretraining/four-gpus \
  --world_size 4 --update_freq 4
```

`--max_tokens` sets the maximum audio sample budget per GPU and microbatch;
`--max_sample_size` caps each crop. Additional Fairseq/Hydra overrides are passed
through, for example `optimization.max_update=40000 lr_scheduler.warmup_updates=3200`.
The configuration is in [pretraining_config.yaml](src/pretraining_config.yaml).

## Training protocol

| Setting | Value |
|---|---|
| Audio / frontend | 44,100 Hz, normalized, stride 960, no waveform padding |
| Transformer | 12 layers, hidden size 768, 8 attention heads |
| Crop duration | 5–20 seconds |
| Maximum audio samples per GPU / microbatch | 882,000 |
| Optimizer | Adam, peak LR 0.0005, weight decay 0.01, gradient clip 10 |
| Schedule | Polynomial decay, 32,000 warmup / 400,000 total updates |
| Precision / seed / data workers | FP16 / 42 / 0 |
| Checkpoint and validation interval | 5,000 updates, plus epoch/run boundaries |

The launcher defaults to one GPU and accumulation 1. The published run used
16 GPUs and accumulation 1, with a maximum global budget of 14,112,000 audio
samples per optimizer update. Four GPUs with accumulation 4, or one GPU with
`--update_freq 16`, give the same maximum budget. Actual batches depend on audio
durations; changing hardware or accumulation does not guarantee identical results.

For CPU checks, add `--no-fp16 common.cpu=true`. Use a small dataset and a short
update budget; training the full encoder is intended for GPUs. The launcher
uses native CPU kernels to avoid a PyTorch 2.0.1 MKLDNN Conv1d backward issue;
CUDA kernels are unaffected.

The main published model completed 400,000 updates; the
[10% checkpoint](https://huggingface.co/dolphinteam/OpenWhistle-Wav2Vec2.0-10pct)
completed 40,000 with 3,200 warmup updates. Their Hub repositories include
training settings, validation history and conversion provenance; model weights
are licensed under CC BY 4.0. The exact historical 10% subset remains private.
`--train_fraction` prepares deterministic hash-based subsets, which do not
establish the identity of the historical splits.

## Export

```bash
python experiments/pretraining/src/export_model.py \
  --checkpoint runs/pretraining/reproduction/checkpoint_last.pt \
  --output_dir runs/pretraining/hf-export
```

Select the final checkpoint to follow the published checkpoint selection.
The exporter requires an empty output directory, maps every tensor and checks
Fairseq/Transformers embeddings in float32 (`atol=rtol=1e-4`). It records the
checkpoint hash, completed updates and preprocessing settings. The export
supports inference; native training resume requires the original Fairseq state.
The [benchmark README](../benchmark/README.md#representations) identifies the
checkpoints used for evaluation.

Tests live in the repository's `tests/` directory. In the pretraining environment,
install `pytest==7.4.4` and run
`python -m pytest -q tests/test_pretraining_data.py tests/test_pretraining_pipeline.py`.
They use synthetic data and include a short CPU training/resume and export checks.
