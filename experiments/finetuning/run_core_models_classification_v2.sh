#!/usr/bin/env bash

set -euo pipefail

for backbone in aves_bio aves_core biolingual; do
  if [[ "${backbone}" == "biolingual" ]]; then
    learning_rate=5e-6
  else
    learning_rate=1e-5
  fi

  for config in balanced unbalanced all; do
    for seed in 42 43 44; do
      output_dir="experiments/finetuning/results/classification_v2/${backbone}/${config}/seed_${seed}"

      if [[ -f "${output_dir}/results.json" ]]; then
        echo "Déjà terminé : ${backbone} / ${config} / seed ${seed}"
        continue
      fi

      echo "Lancement : ${backbone} / ${config} / seed ${seed}"

      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      TOKENIZERS_PARALLELISM=false \
      python experiments/finetuning/src/finetune_classification.py \
        --backbone "${backbone}" \
        --dataset_config "${config}" \
        --output_dir "${output_dir}" \
        --seed "${seed}" \
        --epochs 50 \
        --patience 15 \
        --batch_size 32 \
        --gradient_accumulation_steps 1 \
        --num_workers 8 \
        --learning_rate "${learning_rate}" \
        --head_learning_rate 1e-4
    done
  done
done
