# OpenWhistle: End-to-End Fine-Tuning

End-to-end adaptation of the six neural encoders from the frozen-probe
benchmark: AVES-core, BioLingual, AVES-bio, OpenWhistle Wav2Vec2.0,
OpenWhistle AVES from scratch, and continual AVES-bio. Each is evaluated on
classification `all` (10 classes), classification `balanced` (6 classes),
and detection `default` (7 binary labels).

[Results](#results) · [Installation](#installation) · [Fine-tuning](#fine-tuning) ·
[Protocol](#training-and-evaluation-protocol) · [Export](#results-and-export)

## Results

Scores are percentages: the full-test metric ± the standard deviation over
1,000 test-clip bootstrap resamples. Each evaluation uses one training seed.
Best point estimates in each column are bold.

| Encoder | Classification (`all`) | Classification (`balanced`) | Detection |
|---|---:|---:|---:|
| AVES-core | 58.41 ± 2.24 | 78.89 ± 1.92 | 77.51 ± 2.17 |
| BioLingual | 57.28 ± 2.03 | 80.35 ± 1.76 | 73.53 ± 1.90 |
| AVES-bio | 61.42 ± 2.59 | 80.13 ± 1.82 | 76.86 ± 2.18 |
| AVES-bio (from scratch) | **64.79 ± 3.50** | **85.55 ± 1.58** | **81.44 ± 2.03** |
| Wav2Vec2.0 | 55.78 ± 1.88 | 84.14 ± 1.64 | 81.32 ± 2.02 |
| AVES-bio (continual) | 62.19 ± 2.57 | 82.17 ± 1.65 | 78.41 ± 1.90 |

[Full-precision CSV](results/results.csv) contains one row per model and dataset
configuration, using the same identifiers and score columns as linear probing.
The model directories under [results/](results/) retain 18 portable `results.json`
reports with effective training settings, histories and detailed metrics.
The directory name `classification/` denotes `all`; `balanced/` holds the other
classification configuration. Local paths, null fields and unused CLI defaults
are omitted. Checkpoints and prediction/bootstrap arrays remain local run outputs.

## Installation

Use Python 3.13 and compatible PyTorch/TorchAudio builds for the machine's CUDA
or CPU setup. The dependencies match the frozen benchmark:

```bash
python -m pip install -r experiments/finetuning/requirements.txt
```

Training requires the audio datasets and pretrained models; exporting
saved results does not. Use the complete repository checkout: this directory
imports the shared loaders, encoders and metrics from `experiments/benchmark/src`.

The recorded campaign used Python 3.13.9, PyTorch 2.12.1+cu130,
TorchAudio 2.11.0+cu130 and CUDA 13.0. Other recorded versions are NumPy 2.3.5,
SciPy 1.16.3, scikit-learn 1.7.2, Transformers 5.6.1, Datasets 4.8.4,
esp-aves 1.0.0 and huggingface-hub 1.23.0. The requirements pin the shared
Python dependencies; PyTorch/TorchAudio builds must suit the target machine.
CUDA is recommended for training the full encoders. The launcher runs one job
at a time. GPU memory depends on the model, batch size and clip lengths.
The retained reports do not include wall-clock durations or peak GPU memory.

## Fine-tuning

From the repository root:

```bash
python experiments/finetuning/src/run_finetuning.py \
  --output_dir runs/finetuning/reproduction
```

The Python launcher runs 18 jobs sequentially. It fixes seed 42 and 1,000
bootstrap resamples, matching the benchmark. Add `--resume` to skip compatible
completed evaluations. An interrupted job restarts
from pretrained initialization; its partial files move to `interrupted/`.
Resume requires matching recipes, model initialization and software environment.
It also requires the run's saved predictions; after removing them, start a
new training run with a separate `--output_dir` to regenerate the evaluation.

Use `--dry_run` to inspect commands without downloads or writes,
`--prepare_only` to prepare the run manifest, or `--output_dir` for a separate
experiment. Without `--output_dir`, a new directory is created at
`runs/finetuning/run-YYYYMMDD-HHMMSS/` at the repository root. Local run files
are ignored by Git; `results/` contains the published results.
The launcher disables tokenizer parallelism.

## Training and evaluation protocol

Each model/configuration uses one training seed, 42, and two phases:

1. Train on train only and select the epoch by validation macro-F1 for
   classification or validation mAP for detection, with early stopping.
2. Restart the original pretrained encoder and a fresh head with the same seed.
   Reset AdamW, the scheduler and AMP scaler. Refit on train + validation for
   exactly the selected number of epochs, without further validation selection.
   Reload `final.pt` and evaluate once on test.

Classification uses cross-entropy; detection uses binary cross-entropy with
logits. Neither loss applies class or positive weighting. Training uses FP16
AMP; validation and test use batch size 1 with AMP disabled. CUDA permits TF32.
BioLingual's unused text tower is frozen; its full audio tower and projection
are trained. Clips are not truncated.

The linear learning-rate schedule retains its original maximum epoch horizon
in both phases. Step counts account for each phase's loader size; the selected
epoch changes the stopping point, not the schedule horizon.

| Model | Classification batch × accumulation; max epochs / patience | Detection batch × accumulation; max epochs / patience |
|---|---|---|
| AVES-core | 32 × 1; 50 / 15 | 8 × 1; 30 / 5 |
| BioLingual | 32 × 1; 50 / 15 | 8 × 1; 30 / 10 |
| AVES-bio | 32 × 1; 50 / 15 | 64 × 1; 30 / 5 |
| Wav2Vec2.0 | 32 × 2; 50 / 15 | 8 × 1; 30 / 15 |
| AVES from scratch | 32 × 1; 50 / 15 | 64 × 1; 30 / 5 |
| AVES continual | 4 × 8; 50 / 15 | 4 × 16; 30 / 5 |

`all` and `balanced` share the classification recipe. AdamW uses encoder LR
1e-5, head LR 1e-4, weight decay 0.01, dropout 0.1, and 10% warmup. BioLingual's
classification encoder LR is 5e-6. Gradient clipping uses norm 1.0.

Models, labels and evaluation configurations are shared with linear probing.
Fine-tuning recipes are defined in [finetuning_config.py](src/finetuning_config.py).
Model and dataset revisions are pinned in
[`benchmark_revisions.py`](../benchmark/src/benchmark_revisions.py) and
[`aves_models.py`](../benchmark/src/aves_models.py).
Run manifests omit revision dictionaries; model and dataset loading uses the
pinned revisions defined in these modules.
Wav2Vec2.0 uses `dolphinteam/OpenWhistle-Wav2Vec2.0`, with the same pinned
checkpoint revision as the evaluated model. See the
[benchmark README](../benchmark/README.md).
Scratch AVES uses the native 44.1 kHz Stage-2 stride-320 export with verified
weight/config hashes. Continual AVES concatenates three 16 kHz band views,
matching the benchmark. Original AVES encoders use 44.1 kHz; BioLingual uses
its native CLAP frontend.

Fine-tuning changes the encoder, classifier optimization and regularization
relative to the standardized logistic-regression probe. Their comparison does
not isolate encoder freezing as the only changed factor.

## Results and export

Each new model/configuration run produces `best.pt` for selection,
`final.pt` for test evaluation, `results.json`, ordered test predictions and
bootstrap scores. The directory name `classification/` denotes `all`;
`balanced/` and `detection/` hold the other two configurations.
The published results retain only `results.json` reports and `results.csv`.
CSV scores and uncertainties are fractions; multiply by 100 for percentages.
Run manifests, software metadata and binary artifacts stay under
`runs/finetuning/` at the repository root.

The point estimate is the full-test macro-F1 or mAP. Each ± is the sample
standard deviation of 1,000 test-clip bootstrap scores with shared resampling
indices across models (`default_rng(42)`, `ddof=1`). It measures test-sample
uncertainty, not variability across training seeds.
It does not account for dependence between clips from the same recording session.
Training uses a fixed accumulation divisor even in a final incomplete group;
this retains the optimization used for the published scores.

The launcher validates each report before resuming or collecting its results,
including label order, validation selection and the train + validation refit.
CSV/report consistency and available prediction arrays are checked during export.

After a local campaign, rebuild `results.csv` from its saved predictions:

```bash
python experiments/finetuning/src/run_finetuning.py \
  --output_dir runs/finetuning/reproduction --results_only
```

Export a verified compact release into a new directory:

```bash
python experiments/finetuning/src/export_finetuning_results.py \
  --run_dir runs/finetuning/reproduction \
  --output_dir runs/finetuning/exported-results
```

The exporter checks the CSV against the reports and, when prediction arrays
are available, recomputes the metrics and bootstrap uncertainties. It copies
only the summary CSV and 18 detailed reports and refuses to overwrite an
existing release. The published summaries contain no prediction or bootstrap
arrays, so their uncertainties cannot be recomputed without rerunning training.

Checkpoint writes use a temporary file on the same filesystem and replace the
previous file only after successful serialization and flush. Allow space for
both the previous checkpoint and its replacement, about 360 MiB each for the
Wav2Vec2/HuBERT/AVES encoders.
