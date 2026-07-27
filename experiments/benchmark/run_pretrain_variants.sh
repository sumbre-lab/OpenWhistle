#!/usr/bin/env bash
# Trains split-based LR for the AVES-bio / Wav2Vec2.0 pretraining-size
# ablation models (see the NOT ANONYMIZED block in src/conf.py) on the
# classification task, across both the "balanced" and "all" dataset
# configs. Always cds to this script's directory (experiments/benchmark);
# outputs under ./results/results_classification.txt .
#
# NOT ANONYMIZED: relies on the "dolphinteam" HF repo IDs in src/conf.py.
# Do not merge or push this script's results/logs to `main`.
#
# Invoke: ./run_pretrain_variants.sh  or  bash path/to/experiments/benchmark/run_pretrain_variants.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"

mkdir -p "${ROOT}/results"

models=(
  aves_bio_pretrain_10pct
  aves_bio_pretrain_50pct
  aves_bio_pretrain_100pct
  aves_bio_pretrain_10pct_44khz
  aves_bio_pretrain_50pct_44khz
  aves_bio_pretrain_100pct_44khz
  wav2vec2_pretrain_10pct
  wav2vec2_pretrain_50pct
  wav2vec2_pretrain_transformers
)
inverse_regs=(0.1 1.0 10.0)
classification_configs=(balanced all)

for model in "${models[@]}"; do
  for classification_config in "${classification_configs[@]}"; do
    echo "train_lr_splits: model=${model} inverse_regs=${inverse_regs[*]} dataset=classification classification_config=${classification_config}"
    python src/train_lr_splits.py \
      --model "$model" \
      --inverse_regs "${inverse_regs[@]}" \
      --dataset_name classification \
      --classification_config "$classification_config"
    echo "===================================================="
  done
done

echo "Finished. Training logs: ${ROOT}/results/results_classification.txt"
