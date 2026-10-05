#!/usr/bin/env python3
"""Run every Hugging Face collection checkpoint on both tasks."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from hf_collection_models import HF_COLLECTION_MODELS
from pipeline_config import EMBEDDING_PIPELINE_VERSION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("classification", "detection"),
        default=("classification", "detection"),
    )
    parser.add_argument(
        "--classification_configs",
        nargs="+",
        choices=("balanced", "unbalanced", "all"),
        default=("balanced", "unbalanced", "all"),
    )
    parser.add_argument("--inverse_regs", nargs="+", type=float, default=(0.1, 1.0, 10.0))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp_dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--fast_cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--audio_workers", type=int, default=8)
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "hf_collections_benchmark.csv",
    )
    parser.add_argument(
        "--model_ids",
        nargs="+",
        default=None,
        help="Optional subset of exact model IDs.",
    )
    parser.add_argument(
        "--embedding_cache_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "embedding_cache",
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed pairs.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def completed_pairs(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as result_file:
        completed = set()
        for row in csv.DictReader(result_file):
            dataset_name = row["dataset_name"]
            expected_metric = "Macro-F1" if dataset_name == "classification" else "mAP"
            if row.get("metric") != expected_metric:
                continue
            if (
                row.get("embedding_pipeline_version")
                != EMBEDDING_PIPELINE_VERSION
            ):
                continue
            completed.add(
                (
                    row["model_id"],
                    dataset_name,
                    row.get("dataset_config", ""),
                )
            )
        return completed


def main():
    args = parse_args()
    selected_ids = set(args.model_ids) if args.model_ids else None
    models = [
        model
        for model in HF_COLLECTION_MODELS
        if selected_ids is None or model["model_id"] in selected_ids
    ]
    if selected_ids is not None:
        unknown = selected_ids - {model["model_id"] for model in models}
        if unknown:
            raise ValueError(f"Unknown model IDs: {sorted(unknown)}")

    train_script = Path(__file__).resolve().parent / "train_lr_splits.py"
    completed = completed_pairs(args.results_csv)
    evaluations = []
    if "classification" in args.datasets:
        evaluations.extend(
            ("classification", config)
            for config in args.classification_configs
        )
    if "detection" in args.datasets:
        evaluations.append(("detection", "default"))

    total = len(models) * len(evaluations)
    index = 0
    for model in models:
        for dataset_name, dataset_config in evaluations:
            index += 1
            pair = (model["model_id"], dataset_name, dataset_config)
            if pair in completed and not args.force:
                print(
                    f"[{index}/{total}] Skipping completed "
                    f"{pair[0]} / {pair[1]}/{pair[2]}"
                )
                continue

            command = [
                sys.executable,
                str(train_script),
                "--model",
                "hf",
                "--hf_model_id",
                model["model_id"],
                "--model_label",
                model["label"],
                "--dataset_name",
                dataset_name,
                "--dataset_config",
                dataset_config,
                "--inverse_regs",
                *(str(value) for value in args.inverse_regs),
                "--seed",
                str(args.seed),
                "--num_seeds",
                str(args.num_seeds),
                "--num_bootstrap",
                str(args.num_bootstrap),
                "--embedding_batch_size",
                str(args.embedding_batch_size),
                "--amp_dtype",
                args.amp_dtype,
                "--audio_workers",
                str(args.audio_workers),
                "--embedding_cache_dir",
                str(args.embedding_cache_dir),
                "--results_csv",
                str(args.results_csv),
            ]
            if model.get("sample_rate") is not None:
                command.extend(
                    ("--hf_sample_rate", str(model["sample_rate"]))
                )
            if model.get("feature_mode") is not None:
                command.extend(
                    ("--hf_feature_mode", str(model["feature_mode"]))
                )
            if args.mixed_precision:
                command.append("--mixed_precision")
            if args.fast_cuda:
                command.append("--fast_cuda")
            print(
                f"[{index}/{total}] Running {model['model_id']} / "
                f"{dataset_name}/{dataset_config}"
            )
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)

    print(f"Structured results: {args.results_csv}")


if __name__ == "__main__":
    main()
