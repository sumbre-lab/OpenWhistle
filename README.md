# OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations

This repository accompanies the **OpenWhistle** paper. It provides code to access data and reproduce the article experiments: running the benchmark across multiple models and pretraining a Wav2Vec2.0 model on OpenWhistle. It also includes the code for whistle presence detection and segmentation in the data processing pipeline, as well as scripts to reproduce the figures from the manuscript.

<p align="center">
  <img src="Annotation_Pipeline.png" alt="OpenWhistle annotation and processing pipeline" width="720">
</p>


## Hugging Face datasets

| Resource | Hugging Face link | Review samples |
|----------|-------------------|----------------|
| **Pretraining corpus** (long-form unlabeled / weakly processed audio and segments for self-supervised and large-scale use) | [OpenWhistleNeurIPS26/OpenWhistle-Pretraining](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-Pretraining) | [review-sample](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-Pretraining/viewer/review-sample/train) |
| **Classification benchmark** (expert-annotated clips and labels) | [OpenWhistleNeurIPS26/OpenWhistle-Classification-Finetuning](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-Classification-Finetuning) | [balanced-review-sample](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-Classification-Finetuning/viewer/balanced-review-sample/train) |
| **Detection benchmark** (expert-annotated windows for detection) | [OpenWhistleNeurIPS26/OpenWhistle-Detection-Finetuning](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-Detection-Finetuning) | n/a |
| **CNN training set** (audio windows, spectrograms, and binary whistle/noise labels) | [OpenWhistleNeurIPS26/OpenWhistle-CNN](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-CNN) | [review-sample](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-CNN/viewer/review-sample/train) |

The OpenWhistle Hugging Face datasets are released under **CC-BY 4.0**.

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

If you use OpenWhistle or this repository, please cite the paper when it is available.
