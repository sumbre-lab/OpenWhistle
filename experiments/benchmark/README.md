# OpenWhistle benchmark

Frozen **audio embeddings** followed by **linear probes** (logistic regression) on the expert-annotated Hugging Face datasets. This folder reproduces the paper’s **classification** and **detection** benchmark numbers.

<p align="center">
  <img src="fig_tasks.png" alt="Classification vs detection tasks" width="720">
</p>

## Tasks (what each benchmark measures)

### Classification (`--dataset_name classification`)

- **Goal:** assign each short clip to a **single whistle type** (multiclass).
- **Data:** [dolphinteam/OpenWhistle-1.0-Classification-Finetuning](https://huggingface.co/datasets/dolphinteam/OpenWhistle-1.0-Classification-Finetuning) (default config `balanced`), with **train / validation / test** splits.
- **Head:** one multinomial **logistic regression** on the embedding.
- **Validation:** among inverse-regularization strengths **C ∈ {0.1, 1, 10}**, keep the **C** that maximizes **validation accuracy**.
- **Test:** refit on **train ∪ validation** with the chosen **C**; report **test accuracy** (mean over several RNG seeds, with bootstrap dispersion; see `train_lr_splits.py`).

### Detection (`--dataset_name detection`)

- **Goal:** for each analysis window, predict **which whistle types are present** (multi-label, one binary sub-task per type).
- **Data:** [dolphinteam/OpenWhistle-1.0-Detection-Finetuning](https://huggingface.co/datasets/dolphinteam/OpenWhistle-1.0-Detection-Finetuning) (`default` config), loaded from this repo’s `data/` tree (see `src/hf_datasets.py`). Splits: **train / validation / test**.
- **Label vector:** eight binary dimensions (signature whistles **SW_***, one non-signature bucket **NSW_1**), as in `DETECTION_ONE_HOT_COLUMNS` in `hf_datasets.py`.
- **Head:** **MultiOutputClassifier** over logistic regressions (one classifier per label).
- **Validation:** choose **C** that maximizes **validation mean average precision (mAP)** over labels (probability-based; see `metrics.py`).
- **Test:** same refit protocol as classification; primary score is **test mAP** (again with seed and bootstrap summaries in `train_lr_splits.py`).

Shared preprocessing: embeddings are optionally **standardized** (`StandardScaler` fit on the training portion used for each stage) before logistic regression, unless `--no_normalize_data` is passed.

## Metrics

| Setting | Primary metric | Notes |
|--------|----------------|--------|
| **Classification** | **Accuracy** | Fraction of clips with correct argmax class on the held-out test split. |
| **Detection** | **Mean average precision (mAP)** | Class-wise average precision from ranked scores, then averaged across the eight binary heads (`MeanAveragePrecision` in `metrics.py`). |

## Embedding backbones (`--model`)

All models map each clip’s waveform to a **fixed vector**; only the linear probe is trained for the benchmark.

| `--model` | Description |
|-----------|-------------|
| **`mfcc`** | Time-averaged **MFCC** features (librosa). |
| **`spectrogram`** | Time-averaged **log-mel spectrogram** (128 mels). |
| **`spectral_features`** | Concatenated time-averaged **spectral centroid, bandwidth, contrast, rolloff**. |
| **`dolph2vec`** | **Wav2Vec2**-style pretrained on OpenWhistle corpus. |
| **`biolingual`** | **CLAP**-style audio embedding (`davidrrobinson/biolingual`), 48 kHz pipeline. |
| **`aves_core`** | **AVES** “core” checkpoint (44.1 kHz). |
| **`aves_bio`** | **AVES** “bio” variant. |

## Run everything (all models × both tasks)

From this directory:

```bash
export PYTHONPATH="${PWD}/src"
./run_all.sh
```

`run_all.sh` loops over every `--model` above and invokes `train_lr_splits.py` for **classification** and **detection** with **C ∈ {0.1, 1.0, 10.0}** (open `run_all.sh` for the exact command sequence).

### Single model / single task

```bash
cd experiments/benchmark
export PYTHONPATH="${PWD}/src"
python src/train_lr_splits.py --model dolph2vec --dataset_name classification --inverse_regs 0.1 1.0 10.0
python src/train_lr_splits.py --model dolph2vec --dataset_name detection --inverse_regs 0.1 1.0 10.0
```

Useful flags (see `train_lr_splits.py`): `--seed`, `--num_seeds`, `--num_bootstrap`, `--no_normalize_data`, `--target_sample_rate`.

### Dependencies

The code expects **PyTorch**, **Hugging Face `datasets`**, **librosa**, **scikit-learn**, **tqdm**, and (for AVES) the **`aves`** feature extractor API referenced in `models.py`. Install what your environment is missing before running; exact pins are left to your stack (CUDA vs CPU, etc.).

## Outputs

- Appended lines under **`results/`**, mainly **`results_classification.txt`** and **`results_detection.txt`**, from `train_lr_splits.py`.

For the high-level repository layout, CNN branch, and dataset overview figures, see the [root README](../../README.md) and [`datasets_figures/`](../../datasets_figures/README.md).
