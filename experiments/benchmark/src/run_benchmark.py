"""Evaluate nine frozen representations and build classification and detection tables."""

import argparse
import csv
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from aves_models import AVES_SCRATCH_MODELS
from benchmark_config import (
    CACHE_FINGERPRINT,
    EVALUATIONS,
    EVALUATION_SETTINGS,
    RUNS_DIR,
)
from benchmark_revisions import DATASET_REVISIONS, MODEL_REVISIONS
from benchmark_models import BENCHMARK_MODELS, result_model_id

ROOT = Path(__file__).resolve().parent.parent


def environment_record():
    """Capture the interpreter and numerical stack used by evaluation subprocesses."""
    import torch
    import torchaudio

    packages = (
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "torchaudio",
        "transformers",
        "datasets",
        "esp-aves",
        "huggingface-hub",
        "librosa",
        "soundfile",
        "tqdm",
        "safetensors",
        "tokenizers",
        "numba",
        "pyarrow",
    )
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in packages},
        torch_version=torch.__version__,
        torchaudio_version=torchaudio.__version__,
        cuda=torch.version.cuda,
    )


def make_command(args, model, task, config):
    command = [
        sys.executable,
        "-u",
        str(ROOT / "src/train_lr_splits.py"),
        "--model",
        model["backend"],
        "--model_label",
        model["label"],
        "--dataset_name",
        task,
        "--dataset_config",
        config,
        "--embedding_batch_size",
        str(EVALUATION_SETTINGS["batch_size"]),
        "--batch_native_models",
        "--strict_embedding_batches",
        "--no-mixed_precision",
        "--inverse_regs",
        *map(str, EVALUATION_SETTINGS["inverse_regs"]),
        "--seed",
        str(EVALUATION_SETTINGS["seed"]),
        "--num_seeds",
        str(EVALUATION_SETTINGS["num_seeds"]),
        "--num_bootstrap",
        str(EVALUATION_SETTINGS["num_bootstrap"]),
        "--embedding_cache_dir",
        str(args.embedding_cache_dir),
        "--predictions_dir",
        str(args.output_dir / "predictions"),
        "--results_csv",
        str(args.output_dir / "results.csv"),
    ]
    if "model_id" in model:
        command += ["--hf_model_id", model["model_id"]]
    if "feature_mode" in model:
        command += ["--hf_feature_mode", model["feature_mode"]]
    if "sample_rate" in model:
        command += ["--hf_sample_rate", str(model["sample_rate"])]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Local run directory (default: runs/benchmark/run-YYYYMMDD-HHMMSS).",
    )
    parser.add_argument(
        "--embedding_cache_dir",
        type=Path,
        default=RUNS_DIR / "embedding_cache",
        help="Shared local embedding cache (default: runs/benchmark/embedding_cache at the repository root).",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue a compatible local run."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print 27 evaluation commands and one table command without running them.",
    )
    args = parser.parse_args()
    args.output_dir = (
        (
            args.output_dir
            or RUNS_DIR
            / (
                "run-"
                + datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y%m%d-%H%M%S")
            )
        )
        .expanduser()
        .resolve()
    )
    args.embedding_cache_dir = args.embedding_cache_dir.expanduser().resolve()
    models = [dict(model) for model in BENCHMARK_MODELS]
    jobs = [(model, task, config) for model in models for task, config in EVALUATIONS]
    commands = [make_command(args, *job) for job in jobs]
    table_command = [
        sys.executable,
        str(ROOT / "src/build_tables.py"),
        "--run_dir",
        str(args.output_dir),
    ]
    if args.dry_run:
        for command in commands + [table_command]:
            print(shlex.join(command))
        return
    manifest = dict(
        cache_fingerprint=CACHE_FINGERPRINT,
        **EVALUATION_SETTINGS,
        models=models,
        commands=commands,
        embedding_cache_dir=str(args.embedding_cache_dir),
    )
    model_ids = {result_model_id(model) for model in models} | {
        "davidrrobinson/biolingual"
    }
    revisions = {
        **MODEL_REVISIONS,
        **{spec["model_id"]: spec["revision"] for spec in AVES_SCRATCH_MODELS},
    }
    manifest["model_revisions"] = {
        model_id: revision
        for model_id, revision in revisions.items()
        if model_id in model_ids
    }
    manifest["dataset_revisions"] = DATASET_REVISIONS
    manifest_path = args.output_dir / "run_config.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            parser.error(
                "Existing run requires --resume with exactly the same settings."
            )
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(
            "Use a new/empty output directory; existing files have no matching run manifest."
        )
    environment_path = args.output_dir / "environment.json"
    current_environment = environment_record()
    if (
        environment_path.exists()
        and json.loads(environment_path.read_text()) != current_environment
    ):
        parser.error(
            "The numerical environment changed; use a new output directory to avoid mixing evaluations."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    environment_path.write_text(json.dumps(current_environment, indent=2) + "\n")
    results = args.output_dir / "results.csv"
    completed = set()
    if args.resume and results.exists():
        with results.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("strict_embedding_batches") == "True"
                    and int(row["embedding_batch_size"]) == manifest["batch_size"]
                    and row["mixed_precision"] == "False"
                    and int(row["num_seeds"]) == manifest["num_seeds"]
                    and int(row["num_bootstrap"]) == manifest["num_bootstrap"]
                    and int(row.get("seed") or -1) == manifest["seed"]
                    and row["metric"]
                    == (
                        "Macro-F1" if row["dataset_name"] == "classification" else "mAP"
                    )
                ):
                    completed.add(
                        (row["model_id"], row["dataset_name"], row["dataset_config"])
                    )
    print(
        f"Run directory: {args.output_dir}; {len(jobs)} evaluations; FP32; "
        f"{manifest['num_seeds']} bootstrap seeds × {manifest['num_bootstrap']} resamples "
        f"= {manifest['num_seeds'] * manifest['num_bootstrap']} total",
        flush=True,
    )
    for index, ((model, task, config), command) in enumerate(zip(jobs, commands), 1):
        key = (result_model_id(model), task, config)
        if key in completed:
            print(
                f"[{index}/{len(jobs)}] Already complete: {model['label']} {task}/{config}",
                flush=True,
            )
            continue
        print(f"[{index}/{len(jobs)}] {model['label']} {task}/{config}", flush=True)
        subprocess.run(command, check=True)
    subprocess.run(table_command, check=True)
    print(f"Finished: {results}", flush=True)


if __name__ == "__main__":
    main()
