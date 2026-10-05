#!/usr/bin/env bash

set -euo pipefail

for config in balanced unbalanced all; do
  if [[ "${config}" == "balanced" ]]; then
    output_root="experiments/finetuning/results/hf_collection"
  else
    output_root="experiments/finetuning/results/hf_collection_${config}"
  fi

  echo "HF collection classification : ${config}"
  python experiments/finetuning/src/run_hf_collection_finetuning.py \
    --dataset_config "${config}" \
    --output_root "${output_root}" \
    --seeds 42 43 44
done

echo "HF collection detection"
python experiments/finetuning/src/run_hf_collection_detection.py \
  --output_root experiments/finetuning/results/hf_collection_detection \
  --seeds 42 43 44
