#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_DIR="$(cd "$BENCHMARK_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="$BENCHMARK_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPOSITORY_DIR"

"$PYTHON_BIN" -u experiments/benchmark/src/run_hf_collections.py \
  --datasets classification detection \
  --classification_configs balanced unbalanced all \
  --inverse_regs 0.1 1.0 10.0 \
  --embedding_batch_size 64 \
  --mixed_precision \
  --amp_dtype float16 \
  --audio_workers 8 \
  --fast_cuda \
  "$@"

"$PYTHON_BIN" experiments/benchmark/src/build_rebuttal_table.py
