#!/usr/bin/env python3
"""Run supervised fine-tuning for every HF_COLLECTION_MODELS checkpoint."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SRC = SCRIPT_DIR.parents[1] / "benchmark" / "src"
sys.path.insert(0, str(BENCHMARK_SRC))

from hf_collection_models import HF_COLLECTION_MODELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument(
        "--dataset_config",
        choices=("balanced", "unbalanced", "all"),
        default="balanced",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=SCRIPT_DIR.parent / "results" / "hf_collection",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional exact model IDs; default runs the complete collection.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--standard_batch_size", type=int, default=8)
    parser.add_argument("--btb3_batch_size", type=int, default=4)
    parser.add_argument("--effective_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id)


def main() -> None:
    args = parse_args()
    models = list(HF_COLLECTION_MODELS)
    if args.models is not None:
        requested = set(args.models)
        known = {model["model_id"] for model in models}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown model IDs: {sorted(unknown)}")
        models = [model for model in models if model["model_id"] in requested]

    train_script = SCRIPT_DIR / "finetune_hf_classification.py"
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")

    total = len(models) * len(args.seeds)
    run_index = 0
    for model_spec in models:
        feature_mode = model_spec.get("feature_mode", "standard")
        is_btb3 = feature_mode.startswith("btb3_")
        batch_size = (
            args.btb3_batch_size if is_btb3 else args.standard_batch_size
        )
        if args.effective_batch_size % batch_size:
            raise ValueError(
                f"Effective batch size {args.effective_batch_size} must be "
                f"divisible by physical batch size {batch_size}."
            )
        accumulation = args.effective_batch_size // batch_size

        for seed in args.seeds:
            run_index += 1
            output_dir = (
                args.output_root
                / safe_name(model_spec["model_id"])
                / f"seed_{seed}"
            )
            result_path = output_dir / "results.json"
            if result_path.exists() and not args.force:
                print(f"[{run_index}/{total}] Skip completed {result_path}")
                continue

            command = [
                sys.executable,
                str(train_script),
                "--model_id",
                model_spec["model_id"],
                "--feature_mode",
                feature_mode,
                "--dataset_config",
                args.dataset_config,
                "--output_dir",
                str(output_dir),
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch_size",
                str(batch_size),
                "--gradient_accumulation_steps",
                str(accumulation),
                "--num_workers",
                str(args.num_workers),
                "--learning_rate",
                str(args.learning_rate),
                "--head_learning_rate",
                str(args.head_learning_rate),
            ]
            if model_spec.get("sample_rate") is not None:
                command.extend(
                    ("--sample_rate", str(model_spec["sample_rate"]))
                )

            print(
                f"[{run_index}/{total}] {model_spec['label']} seed={seed} "
                f"batch={batch_size} accumulation={accumulation}"
            )
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
