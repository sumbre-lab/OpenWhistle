#!/usr/bin/env bash
# Trains the open-set (known vs. unseen whistle type) LR probe for each model.
# Always cds to this script's directory (experiments/benchmark); outputs under
# ./results/results_openset.txt .
# Invoke: ./run_openset.sh  or  bash path/to/experiments/benchmark/run_openset.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"

mkdir -p "${ROOT}/results"

models=(dolph2vec biolingual aves_bio)
inverse_regs=(0.1 1.0 10.0)
held_out_classes=(SW_Neo SW_Nikita SW_Nana SW_Yosefa NSW_1)

for model in "${models[@]}"; do
  for held_out_class in "${held_out_classes[@]}"; do
    echo "train_lr_openset: model=${model} inverse_regs=${inverse_regs[*]} held_out_class=${held_out_class}"
    python src/train_lr_openset.py \
      --model "$model" \
      --inverse_regs "${inverse_regs[@]}" \
      --held_out_class "$held_out_class"
    echo "===================================================="
  done
done

echo "Finished. Training logs: ${ROOT}/results/results_openset.txt"
