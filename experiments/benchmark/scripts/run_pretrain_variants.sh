#!/usr/bin/env bash
# Reproduce the pretraining-size ablation using the shared checkpoint registry
# and the current masked-padding/BTB3 embedding pipeline.
set -euo pipefail
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON_BIN:-python}" "$BENCHMARK_DIR/src/run_hf_collections.py" \
  --datasets classification --classification_configs balanced all \
  --inverse_regs 0.1 1.0 10.0 "$@"
