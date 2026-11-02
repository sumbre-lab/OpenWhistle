"""Fine-tune six encoders on all/balanced classification and detection."""
import argparse
import csv
from datetime import datetime, timezone
import os
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from finetuning_config import (  # noqa: E402
    ROOT, RUNS_DIR, MODELS, MODEL_REVISIONS, RESULT_MODEL_IDS,
    HF_MODELS, TASKS, TASK_DATASETS, SEED, NUM_BOOTSTRAP, CSV_FIELDS, recipe, run_manifest,
)
from run_benchmark import environment_record  # noqa: E402
from hf_datasets import CLASSIFICATION_LABEL_NAMES, DETECTION_ONE_HOT_COLUMNS  # noqa: E402


def make_command(output, model, task, seed):
    key, _, _, model_id = model
    prefix = "finetune_hf" if key in HF_MODELS else "finetune"
    suffix = "_detection_map.py" if key in HF_MODELS else "_detection.py"
    script = prefix + (suffix if task == "detection" else "_classification.py")
    command = [sys.executable, "-u", str(ROOT / "src" / script),
               "--output_dir", str(output / key / task / f"seed_{seed}"), "--seed", str(seed)]
    if key in (*HF_MODELS, "biolingual", "aves_scratch"):
        command += ["--model_id", model_id]
    if key not in HF_MODELS:
        command += ["--backbone", key]
    if task != "detection":
        command += ["--dataset_config", "balanced" if task == "balanced" else "all"]
    for option, value in recipe(key, task).items():
        command += [f"--{option}"] if value is True else [f"--{option}", str(value)]
    return command


def validate_report_details(report, task):
    """Check split labels, validation selection, refit history and classification counts."""
    metric = "map" if task == "detection" else "macro_f1"
    expected_labels = (DETECTION_ONE_HOT_COLUMNS if task == "detection" else
                       CLASSIFICATION_LABEL_NAMES["balanced" if task == "balanced" else "all"])
    if report.get("label_names") != list(expected_labels):
        raise ValueError("Report label order differs from the pinned dataset.")
    history = report.get("history", [])
    if (not history or len(history) > report["training_args"]["epochs"]
            or [entry["epoch"] for entry in history] != list(range(1, len(history) + 1))):
        raise ValueError("Missing or invalid validation history.")
    scores = np.asarray([entry["validation"][metric] for entry in history])
    best_epoch = report["best_epoch"]
    if (not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or int(scores.argmax()) + 1 != best_epoch
            or not np.isclose(scores[best_epoch - 1], report[f"validation_{metric}"], rtol=0, atol=1e-10)
            or [entry["epoch"] for entry in report["refit_history"]] != list(range(1, best_epoch + 1))):
        raise ValueError("Selected epoch or refit history disagrees with validation selection.")
    losses = [entry["train_loss"] for entry in history + report["refit_history"]]
    if not np.isfinite(losses).all() or (np.asarray(losses) < 0).any():
        raise ValueError("Invalid training losses.")
    if task != "detection":
        cm = np.asarray(report.get("test_confusion_matrix", []))
        count = len(expected_labels)
        if (cm.shape != (count, count) or not np.issubdtype(cm.dtype, np.integer)
                or (cm < 0).any() or not cm.sum()):
            raise ValueError("Invalid test confusion matrix.")
        denominator = cm.sum(axis=0) + cm.sum(axis=1)
        f1 = np.divide(2 * cm.diagonal(), denominator,
                       out=np.zeros(count, dtype=float), where=denominator != 0).mean()
        if (not np.isclose(f1, report["test"]["macro_f1"], rtol=0, atol=1e-7)
                or not np.isclose(cm.trace() / cm.sum(), report["test"]["accuracy"], rtol=0, atol=1e-7)):
            raise ValueError("Test metrics disagree with the confusion matrix.")


def load_result(path, model, task, seed, *, require_checkpoint=True, require_predictions=True):
    key, _, _, model_id = model
    with path.open() as handle:
        result = json.load(handle)
    metric = "map" if task == "detection" else "macro_f1"
    expected_config = {"classification": "all", "balanced": "balanced", "detection": "default"}[task]
    if (result.get("protocol") != "supervised_end_to_end_finetuning"
            or result.get("model_id") != model_id or result.get("seed") != seed
            or result.get("dataset_config") != expected_config):
        raise ValueError(f"Incompatible model, seed or task in {path}")
    value = float(result["test"][metric])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid test metric in {path}")
    settings = {**recipe(key, task), "seed": seed}
    if (result.get("model_revision") != MODEL_REVISIONS.get(model_id)
            or result.get("training_args", {}).get("class_weight", "none") != "none"
            or result.get("class_weighting") is not None
            or result.get("final_training", {}).get("class_weighting") is not None
            or result.get("training_args", {}).get("max_duration_seconds") is not None
            or any(result.get("training_args", {}).get(key) != value for key, value in settings.items())):
        raise ValueError(f"Incompatible model revision or training/evaluation settings in {path}")
    final = result.get("final_training", {})
    best_epoch = result.get("best_epoch")
    if (not isinstance(best_epoch, int) or not 1 <= best_epoch <= settings["epochs"]
            or final.get("split") != "train+validation" or final.get("checkpoint") != "final.pt"
            or final.get("epochs") != result.get("best_epoch")
            or final.get("schedule_horizon_epochs") != settings["epochs"]
            or len(result.get("refit_history", [])) != result.get("best_epoch")
            or (require_checkpoint and not path.with_name("final.pt").is_file())):
        raise ValueError(f"Missing or incompatible train+validation refit in {path}")
    if require_predictions and not path.with_name("test_predictions.npz").is_file():
        raise ValueError(f"Missing saved test predictions for {path}")
    validate_report_details(result, task)
    return result


def write_results(output, seed, num_bootstrap):
    from metrics import bootstrap_metric_scores, score_predictions

    summary = []
    reference_targets = {}
    for model in MODELS:
        key, label, _, model_id = model
        for task in TASKS:
            metric_task = "detection" if task == "detection" else "classification"
            path = output / key / task / f"seed_{seed}/results.json"
            entry = load_result(path, model, task, seed, require_checkpoint=False)
            prediction_path = path.with_name("test_predictions.npz")
            with np.load(prediction_path, allow_pickle=False) as predictions:
                y_true, y_pred = predictions["y_true"], predictions["y_pred"]
            if (y_true.shape != y_pred.shape or len(y_true) == 0
                    or not np.isfinite(y_pred).all()
                    or (metric_task == "classification" and y_true.ndim != 1)
                    or (task == "detection" and (y_true.ndim != 2 or y_true.shape[1] != 7))):
                raise ValueError(f"Invalid test predictions in {prediction_path}")
            if task in reference_targets and not np.array_equal(y_true, reference_targets[task]):
                raise ValueError(f"Test target order differs across models: {prediction_path}")
            reference_targets[task] = y_true
            metric = "map" if task == "detection" else "macro_f1"
            score = score_predictions(metric_task, y_true, y_pred)
            if not np.isclose(score, entry["test"][metric], rtol=0, atol=1e-7):
                raise ValueError(f"Saved predictions disagree with test score in {path}")
            scores = bootstrap_metric_scores(metric_task, y_true, y_pred, num_bootstrap, seed)
            dataset_name, dataset_config = TASK_DATASETS[task]
            summary.append(dict(
                model_label=label, model_id=RESULT_MODEL_IDS[key], dataset_name=dataset_name,
                dataset_config=dataset_config, metric="mAP" if task == "detection" else "Macro-F1",
                validation_score=entry[f"validation_{metric}"], best_epoch=entry["best_epoch"],
                test_mean=score, test_std=float(np.std(scores, ddof=1)),
                num_training_seeds=1, num_bootstrap=num_bootstrap, seed=seed))
            np.savez_compressed(path.with_name("bootstrap_scores.npz"), scores=np.asarray(scores))
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved results: {output / 'results.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Local run directory (default: runs/finetuning/run-YYYYMMDD-HHMMSS).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results_only", action="store_true")
    parser.add_argument("--prepare_only", action="store_true",
                        help="Write and validate the manifest without training or aggregating results.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    output = (args.output_dir or RUNS_DIR / (
        "run-" + datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y%m%d-%H%M%S")
    )).expanduser().resolve()
    jobs = [(model, task, make_command(output, model, task, SEED))
            for model in MODELS for task in TASKS]
    if args.dry_run:
        for _, _, command in jobs:
            print(shlex.join(command))
        print(f"One training seed ({SEED}); validation/test batch 1, FP32; "
              f"select epoch on validation, refit on train+validation; {NUM_BOOTSTRAP} test bootstrap resamples.")
        print(f"Then generate {output / 'results.csv'}")
        return
    manifest = run_manifest()
    manifest_path = output / "run_config.json"
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not (args.resume or args.results_only) or previous != manifest:
            parser.error("Existing output requires --resume and exactly matching settings.")
    elif args.results_only:
        parser.error("--results_only requires an existing run.")
    elif output.exists() and any(output.iterdir()):
        parser.error("Use a new or empty output directory.")
    environment = environment_record()
    env_path = output / "environment.json"
    if env_path.exists() and not args.results_only and json.loads(env_path.read_text()) != environment:
        parser.error("The training environment changed; use a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not env_path.exists():
        env_path.write_text(json.dumps(environment, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared run: {output}", flush=True)
        return
    if not args.results_only:
        for model, task, command in jobs:
            key, label, _, _ = model
            path = output / key / task / f"seed_{SEED}/results.json"
            if path.is_file():
                load_result(path, model, task, SEED, require_checkpoint=False)
                print(f"Already complete: {label} {task}, seed {SEED}", flush=True)
                continue
            if path.parent.exists() and any(path.parent.iterdir()):
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archived = output / "interrupted" / key / task / f"seed_{SEED}-{stamp}"
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path.parent), str(archived))
                print(f"Archived interrupted run: {archived}", flush=True)
            print(f"Training {label} {task}, seed {SEED}", flush=True)
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True)
    write_results(output, SEED, NUM_BOOTSTRAP)


if __name__ == "__main__":
    main()
