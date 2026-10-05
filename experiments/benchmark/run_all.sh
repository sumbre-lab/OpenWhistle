#!/usr/bin/env bash
# Trains split-based LR then plot_pr_micro for each dataset. Always cds to this
# script's directory (experiments/benchmark); outputs under ./results/ .
# Invoke: ./run_all.sh  or  bash path/to/experiments/benchmark/run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"

mkdir -p "${ROOT}/results/pr_curves"

models=(mfcc spectrogram spectral_features dolph2vec biolingual aves_bio aves_core)
inverse_regs=(0.1 1.0 10.0)
datasets=(classification detection)

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "train_lr_splits: model=${model} inverse_regs=${inverse_regs[*]} dataset=${dataset}"
    python src/train_lr_splits.py \
      --model "$model" \
      --inverse_regs "${inverse_regs[@]}" \
      --dataset_name "$dataset"
    echo "===================================================="
  done
done

for dataset in "${datasets[@]}"; do
  echo "plot_pr_micro: dataset=${dataset}"
  python src/plot_pr_micro.py \
    --dataset_name "$dataset" \
    --models "${models[@]}" \
    --inverse_regs "${inverse_regs[@]}" \
    --out_dir "${ROOT}/results/pr_curves" \
    --out_name "pr_micro_${dataset}.png"
done

echo "Finished. Training logs: ${ROOT}/results/results_classification.txt and ${ROOT}/results/results_detection.txt"
echo "PR plots and CSVs: ${ROOT}/results/pr_curves/"
