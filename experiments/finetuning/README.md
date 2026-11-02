# Supervised encoder fine-tuning

End-to-end adaptation of the six neural encoders from the frozen-probe
benchmark: AVES-core, BioLingual, AVES-bio, OpenWhistle Wav2Vec2.0,
OpenWhistle AVES from scratch, and continual AVES-bio. Each is evaluated on
classification `all` (10 classes), classification `balanced` (6 classes),
and detection `default` (7 binary labels).

The [published results](results/manuscript/results.csv) contain scores and
uncertainties for all 18 evaluations. Each model/configuration directory also
contains a portable `results.json` with training histories and detailed metrics.
Reports omit local output paths, null fields and unused CLI arguments.
The top-level `model_id` and optional `model_revision` identify the encoder;
`training_args` contains its training and evaluation settings.
Checkpoints, prediction arrays and bootstrap samples have been removed from the retained
release. New training outputs stay in ignored `runs/` directories.

## Installation

Use Python 3.13 and compatible PyTorch/TorchAudio builds for the machine's CUDA
or CPU setup. The dependencies match the frozen benchmark:

```bash
python -m pip install -r experiments/finetuning/requirements.txt
```

Training requires the audio datasets and pretrained models; verification of
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

## Reproduce or resume

From the repository root:

```bash
bash experiments/finetuning/scripts/run_finetuning.sh \
  --output_dir experiments/finetuning/runs/reproduction \
  --seed 42 --num_bootstrap 1000 --resume
```

The launcher sets thread counts and invokes
[`run_finetuning.py`](src/run_finetuning.py). It runs 18 jobs
sequentially and skips compatible completed jobs. An interrupted job restarts
from pretrained initialization; its partial files move to `interrupted/`.
Resume requires matching recipes, model initialization and software environment.
It also requires the run's saved predictions; after removing them, start a
new training run with a separate `--output_dir` to regenerate the evaluation.

Use `--dry_run` to inspect commands without downloads or writes,
`--prepare_only` to prepare the run manifest, or `--output_dir` for a separate
experiment. The default output is `experiments/finetuning/runs/manuscript/`.

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

## Results and verification

Each new model/configuration run produces `best.pt` for selection,
`final.pt` for test evaluation, `results.json`, ordered test predictions and
bootstrap scores. The directory name `classification/` denotes `all`;
`balanced/` and `detection/` hold the other two configurations.
The published results retain only `results.json` reports and `results.csv`.
CSV scores and uncertainties are fractions; multiply by 100 for percentages.
Run manifests, software metadata, generated tables and binary artifacts stay
in the local `runs/` directory.

The point estimate is the full-test macro-F1 or mAP. Each ± is the sample
standard deviation of 1,000 test-clip bootstrap scores with shared resampling
indices across models (`default_rng(42)`, `ddof=1`). It measures test-sample
uncertainty, not variability across training seeds.
It does not account for dependence between clips from the same recording session.
Training uses a fixed accumulation divisor even in a final incomplete group;
this retains the optimization used for the published scores.

Check CSV/report consistency without training or downloads:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python experiments/finetuning/src/verify_finetuning_results.py
```

After a local campaign, rebuild `results.csv` from its saved predictions:

```bash
bash experiments/finetuning/scripts/run_finetuning.sh \
  --output_dir experiments/finetuning/runs/reproduction --tables_only
```

Export a verified compact release into a new directory:

```bash
python experiments/finetuning/src/export_finetuning_results.py \
  --run_dir experiments/finetuning/runs/reproduction \
  --output_dir experiments/finetuning/runs/exported-results
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

## Resources and licensing

The pinned revisions in the source code are publicly accessible without an
access token. Datasets are downloaded on the first training run; the AVES
core/bio weights are cached from the official Earth Species Project storage.

The OpenWhistle [classification dataset](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Classification-Finetuning),
[detection dataset](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Detection-Finetuning)
and [Wav2Vec2.0 weights](https://huggingface.co/dolphinteam/OpenWhistle-Wav2Vec2.0)
are published under CC BY 4.0. See the repository's [existing notice](../../LICENSE-DATA).
The original software is licensed under [MIT](../../LICENSE).
Third-party dependencies, vendored code and pretrained checkpoints retain their
own licenses: [AVES](https://github.com/earthspecies/aves),
[BioLingual](https://huggingface.co/davidrrobinson/biolingual),
[OpenWhistle AVES from scratch](https://huggingface.co/dolphinteam/AVES-OpenWhistle-Stage-2-320-100pct)
and [continual AVES](https://huggingface.co/dolphinteam/AVES-bio-OpenWhistle-100pct-44kHz).

This is the public/camera-ready release. For an anonymous review submission,
identifying repository and Hub links would need a separate anonymized package.
See the [NeurIPS code/data guidelines](https://neurips.cc/public/guides/CodeSubmissionPolicy)
and the [2026 Evaluations & Datasets requirements](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets).
