#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_DIR"

for config in balanced unbalanced all; do
  if [[ "${config}" == "balanced" ]]; then
    batch_size=8
  else
    batch_size=32
  fi

  for seed in 42 43 44; do
    output_dir="experiments/finetuning/results/aves_core/${config}/seed_${seed}"

    if [[ -f "${output_dir}/results.json" ]]; then
      echo "Déjà terminé : AVES-Core / classification / ${config} / seed ${seed}"
      continue
    fi

    echo "Lancement : AVES-Core / classification / ${config} / seed ${seed}"

    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    python experiments/finetuning/src/finetune_classification.py \
      --backbone aves_core \
      --dataset_config "${config}" \
      --output_dir "${output_dir}" \
      --seed "${seed}" \
      --epochs 30 \
      --patience 5 \
      --batch_size "${batch_size}" \
      --gradient_accumulation_steps 1 \
      --num_workers 8 \
      --learning_rate 1e-5 \
      --head_learning_rate 1e-4
  done
done

for seed in 42 43 44; do
  output_dir="experiments/finetuning/results/aves_core/detection_map/seed_${seed}"

  if [[ -f "${output_dir}/results.json" ]]; then
    echo "Déjà terminé : AVES-Core / detection / seed ${seed}"
    continue
  fi

  echo "Lancement : AVES-Core / detection / seed ${seed}"

  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  python experiments/finetuning/src/finetune_detection_map.py \
    --backbone aves_core \
    --output_dir "${output_dir}" \
    --seed "${seed}" \
    --epochs 30 \
    --patience 5 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_workers 8 \
    --learning_rate 1e-5 \
    --head_learning_rate 1e-4
done
