# OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations

This repository accompanies the **OpenWhistle** paper, accepted as a **Spotlight at the NeurIPS 2026 Datasets & Evaluations Track**. It provides code to access the data and reproduce the article experiments: running the benchmark across multiple models and pretraining a Wav2Vec2.0 model on OpenWhistle. It also includes the code for whistle presence detection and segmentation in the data processing pipeline, as well as scripts to reproduce the figures from the manuscript.

- **Paper:** [OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations](https://arxiv.org/abs/2609.34839)
- **Hugging Face collection:** [[NeurIPS'26] OpenWhistle](https://huggingface.co/collections/dolphinteam/neurips26-openwhistle-6a2ac4d02f951a035d8329dd)

<p align="center">
  <img src="Annotation_Pipeline.png" alt="OpenWhistle annotation and processing pipeline" width="720">
</p>


## Hugging Face datasets

| Resource | Hugging Face link |
|----------|-------------------|
| **Pretraining corpus** (long-form unlabeled / weakly processed audio and segments for self-supervised and large-scale use) | [dolphinteam/OpenWhistle-Pretraining](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Pretraining) |
| **Classification benchmark** (expert-annotated clips and labels) | [dolphinteam/OpenWhistle-Classification-Finetuning](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Classification-Finetuning) |
| **Detection benchmark** (expert-annotated windows for detection) | [dolphinteam/OpenWhistle-Detection-Finetuning](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Detection-Finetuning) |
| **CNN training set** (audio windows, spectrograms, and binary whistle/noise labels) | [dolphinteam/OpenWhistle-CNN](https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN) |

The OpenWhistle Hugging Face datasets are released under **CC-BY 4.0**.

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
├── LICENSE-DATA            # CC-BY 4.0 terms for the OpenWhistle datasets
└── README.md
```

- **`experiments/benchmark/`** reproduces the paper benchmark on the classification and detection Hugging Face datasets. The main entry point is [`experiments/benchmark/run_all.sh`](experiments/benchmark/run_all.sh), described below.
- **`experiments/pretraining/`** contains the large-scale self-supervised pretraining material.
- **`cnn/`** contains the VGG16-based whistle presence detection and segmentation branch, integrated with the public CNN dataset and checkpoints on the Hub. See [`cnn/README.md`](cnn/README.md) for setup, training, and inference.
- **`datasets_figures/`** builds the dataset overview figures used in the manuscript, including longitudinal coverage, class distributions, and SNR summaries. See [`datasets_figures/README.md`](datasets_figures/README.md) for plotting instructions.

## Running the benchmark

**Install:** Python 3.12. From `experiments/benchmark/`, run `pip install -r requirements.txt`. If you need a specific CUDA (or CPU) PyTorch build, install `torch` and `torchaudio` from the [PyTorch install page](https://pytorch.org/get-started/locally/) first, then install the requirements file.

The benchmark trains **logistic-regression heads on frozen embeddings** for every combination of **model family**, **inverse regularization** \(× cross-validation\), and **task** (classification vs. detection). Models included in the sweep:

`mfcc`, `spectrogram`, `spectral_features`, `dolph2vec`, `biolingual`, `aves_bio`, `aves_core`

From the repository root:

```bash
cd experiments/benchmark
./run_all.sh
```

This runs `train_lr_splits.py` for each model on both the classification and detection setups, then generates micro-averaged precision–recall curves via `plot_pr_micro.py`. Outputs are written under `experiments/benchmark/results/` (including `results/` text summaries and `results/pr_curves/` for plots and CSVs). Set `PYTHONPATH` as in `run_all.sh` if you invoke the Python modules manually.

For CNN-specific training and evaluation (whistle presence / segmentation branch), use **`cnn/`** and its README rather than this benchmark driver.

## Citation

If you use OpenWhistle, its datasets, models, or this repository, please cite:

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

## Authors

Faadil Mustun, Chiara Semenzin, Roberto Dessi, Pablo Robin Guerrero, Pierre Orhan, Alexis Emanuelli, Emanuele Rossi, Yair Lakretz, Gonzalo de Polavieja, and German Sumbre.
