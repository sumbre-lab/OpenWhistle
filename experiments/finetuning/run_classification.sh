#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
seeds=(42 43 44)

for seed in "${seeds[@]}"; do
  python "${script_dir}/src/finetune_classification.py" \
    --backbone aves_bio \
    --dataset_config balanced \
    --output_dir "${script_dir}/results/aves_bio/seed_${seed}" \
    --seed "${seed}" \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --head_learning_rate 1e-4

  python "${script_dir}/src/finetune_classification.py" \
    --backbone biolingual \
    --dataset_config balanced \
    --output_dir "${script_dir}/results/biolingual/seed_${seed}" \
    --seed "${seed}" \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --head_learning_rate 1e-4
done
