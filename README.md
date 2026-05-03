# OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations

This repository accompanies the **OpenWhistle** paper. It provides code to process data, train the CNN used in the pipeline, run the full benchmark over multiple models, and reproduce the dataset figures from the manuscript.

## Annotation and processing pipeline

The following diagram summarizes how raw hydrophone audio moves through detection, segmentation, and expert annotation for the published Hugging Face releases.

![OpenWhistle annotation and processing pipeline](Annotation_Pipeline.png)


## Hugging Face datasets

| Resource | Hugging Face link |
|----------|-------------------|
| **Pretraining corpus** (long-form unlabeled / weakly processed audio and segments for self-supervised and large-scale use) | [dolphinteam/OpenWhistle-1.0-Pretraining](https://huggingface.co/datasets/dolphinteam/OpenWhistle-1.0-Pretraining) |
| **Classification benchmark** (expert-annotated clips and labels) | [dolphinteam/OpenWhistle-1.0-Classification-Finetuning](https://huggingface.co/datasets/dolphinteam/OpenWhistle-1.0-Classification-Finetuning) |
| **Detection benchmark** (expert-annotated windows for detection) | [dolphinteam/OpenWhistle-1.0-Detection-Finetuning](https://huggingface.co/datasets/dolphinteam/OpenWhistle-1.0-Detection-Finetuning) |

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
chmod +x run_all.sh   # once, if needed
./run_all.sh
```

This runs `train_lr_splits.py` for each model on both the classification and detection setups, then generates micro-averaged precision–recall curves via `plot_pr_micro.py`. Outputs are written under `experiments/benchmark/results/` (including `results/` text summaries and `results/pr_curves/` for plots and CSVs). Set `PYTHONPATH` as in `run_all.sh` if you invoke the Python modules manually.

For CNN-specific training and evaluation (whistle presence / segmentation branch), use **`cnn/`** and its README rather than this benchmark driver.

## Citation

If you use OpenWhistle or this repository, please cite the paper when it is available.
