<div align="center">

<h1>🐬 OpenWhistle</h1>

<h3>A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations</h3>

<p>
  <a href="https://sumbre-lab.github.io/openwhistle/">🌐 <b>Project Page</b></a> •
  <a href="https://arxiv.org/abs/2609.34839">📄 <b>Paper</b></a> •
  <a href="https://huggingface.co/collections/dolphinteam/neurips26-openwhistle-6a2ac4d02f951a035d8329dd">🤗 <b>Hugging Face</b></a>
</p>

<p><b>NeurIPS 2026 Datasets & Evaluations Track — Spotlight</b></p>

</div>

OpenWhistle is the largest publicly available dataset of bottlenose dolphin vocalizations: approximately **180,000 whistles (114 hours)** recorded over five years from a stable pod of five individuals in a semi-natural environment. It includes **8,354 expert-annotated whistles**, reproducible detection and classification benchmarks, the full processing pipeline, and a Wav2Vec2.0 model pretrained specifically on dolphin acoustics.

```bibtex
@misc{mustun2026openwhistle,
  title         = {OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations},
  author        = {Mustun, Faadil and Semenzin, Chiara and Dessi, Roberto and Robin Guerrero, Pablo and Orhan, Pierre and Emanuelli, Alexis and Rossi, Emanuele and Lakretz, Yair and de Polavieja, Gonzalo and Sumbre, German},
  year          = {2026},
  eprint        = {2609.34839},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.34839}
}
```

## News

- **[2026/09]** OpenWhistle was accepted as a **Spotlight** at the NeurIPS 2026 Datasets & Evaluations Track.
- **[2026/09]** We released the OpenWhistle datasets and pretrained models on [Hugging Face](https://huggingface.co/collections/dolphinteam/neurips26-openwhistle-6a2ac4d02f951a035d8329dd).

## Overview

<p align="center">
  <img src="Annotation_Pipeline.png" alt="OpenWhistle annotation and processing pipeline" width="900">
</p>

This repository provides the code to access the data and reproduce the paper experiments. It includes the multi-model benchmark, Wav2Vec2.0 pretraining, whistle presence detection and segmentation, and scripts used to generate the dataset figures in the manuscript.

## Hugging Face datasets

| Resource | Hugging Face link |
|----------|-------------------|
| **Pretraining corpus** — long-form unlabeled and weakly processed audio for self-supervised learning | [dolphinteam/OpenWhistle-Pretraining](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Pretraining) |
| **Classification benchmark** — expert-annotated clips and whistle-type labels | [dolphinteam/OpenWhistle-Classification-Finetuning](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Classification-Finetuning) |
| **Detection benchmark** — expert-annotated windows for whistle detection | [dolphinteam/OpenWhistle-Detection-Finetuning](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Detection-Finetuning) |
| **CNN training set** — audio windows, spectrograms, and binary whistle/noise labels | [dolphinteam/OpenWhistle-CNN](https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN) |

The OpenWhistle datasets are released under **CC BY 4.0**.

## Hugging Face models

| Model | Task | Hugging Face link |
|-------|------|-------------------|
| **OpenWhistle Wav2Vec2.0** | Self-supervised acoustic representation learning and downstream feature extraction | [dolphinteam/OpenWhistle-Wav2Vec2.0](https://huggingface.co/dolphinteam/OpenWhistle-Wav2Vec2.0) |
| **OpenWhistle CNN VGG16** | Binary whistle/noise classification from spectrogram windows | [dolphinteam/OpenWhistle-CNN-VGG16](https://huggingface.co/dolphinteam/OpenWhistle-CNN-VGG16) |

## Repository layout

```text
.
├── cnn/                    # CNN whistle presence detection and segmentation
├── datasets_figures/       # scripts for the dataset overview figures
├── experiments/
│   ├── benchmark/          # frozen-embedding + logistic-regression benchmark
│   └── pretraining/        # self-supervised pretraining code
├── Annotation_Pipeline.png
├── LICENSE-DATA            # CC BY 4.0 terms for the OpenWhistle datasets
└── README.md
```

- [`experiments/benchmark/`](experiments/benchmark/) reproduces the paper benchmark on the classification and detection datasets.
- [`experiments/pretraining/`](experiments/pretraining/) contains the large-scale self-supervised pretraining material.
- [`cnn/`](cnn/) contains the VGG16-based whistle presence detection and segmentation workflow. See the [CNN README](cnn/README.md) for setup, training, and inference.
- [`datasets_figures/`](datasets_figures/) builds the dataset overview figures used in the manuscript.

## Running the benchmark

The benchmark trains logistic-regression heads on frozen embeddings for each model family, inverse regularization value, cross-validation split, and task (classification or detection). The sweep includes:

`mfcc`, `spectrogram`, `spectral_features`, `dolph2vec`, `biolingual`, `aves_bio`, `aves_core`

Install Python 3.12 and the benchmark dependencies:

```bash
cd experiments/benchmark
pip install -r requirements.txt
./run_all.sh
```

If you need a specific CUDA or CPU PyTorch build, install `torch` and `torchaudio` from the [PyTorch installation guide](https://pytorch.org/get-started/locally/) before installing the requirements.

`run_all.sh` runs the classification and detection benchmarks, then generates micro-averaged precision–recall curves. Outputs are written under `experiments/benchmark/results/`. For CNN-specific training and evaluation, use [`cnn/`](cnn/) instead of the benchmark driver.

## Authors

Faadil Mustun, Chiara Semenzin, Roberto Dessi, Pablo Robin Guerrero, Pierre Orhan, Alexis Emanuelli, Emanuele Rossi, Yair Lakretz, Gonzalo de Polavieja, and German Sumbre.
