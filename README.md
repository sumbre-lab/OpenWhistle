# OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations

This repository accompanies the **OpenWhistle** paper. It provides code to access data and reproduce the article experiments: running the benchmark across multiple models and pretraining a Wav2Vec2.0 model on OpenWhistle. It also includes the code for whistle presence detection and segmentation in the data processing pipeline, as well as scripts to reproduce the figures from the manuscript.

<p align="center">
  <img src="Annotation_Pipeline.png" alt="OpenWhistle annotation and processing pipeline" width="720">
</p>


## Hugging Face datasets

| Resource | Hugging Face link |
|----------|-------------------|
| **Pretraining corpus** (long-form unlabeled / weakly processed audio and segments for self-supervised and large-scale use) | [OpenWhistleNeurIPS26/OpenWhistle-Pretraining](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-Pretraining) |
| **Classification benchmark** (expert-annotated clips and labels) | [OpenWhistleNeurIPS26/OpenWhistle-Classification-Finetuning](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-Classification-Finetuning) |
| **Detection benchmark** (expert-annotated windows for detection) | [OpenWhistleNeurIPS26/OpenWhistle-Detection-Finetuning](https://huggingface.co/datasets/OpenWhistleNeurIPS26/OpenWhistle-Detection-Finetuning) |

## Repository layout

- **`experiments/`** — code to **reproduce the paper’s experiments** (notably the tabular and audio benchmark). The main entry point is the benchmark driver under `experiments/benchmark/` (see below). Additional material for large-scale pretraining (e.g. self-supervised audio) lives under `experiments/pretraining/`.
- **`cnn/`** — **binary whistle presence detection** and **whistle segmentation** in the overall pipeline: VGG16-based training and inference on spectrograms, integrated with the public CNN training data and checkpoints on the Hub. See [`cnn/README.md`](cnn/README.md) for environment setup, training, and inference.
- **`datasets_figures/`** — scripts to build the **dataset overview figures** used in the paper (longitudinal extent, temporal coverage, class distributions, SNR summaries, etc.). See [`datasets_figures/README.md`](datasets_figures/README.md) for how to run the plot scripts.

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
