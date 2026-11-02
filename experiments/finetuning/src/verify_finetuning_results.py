"""Check fine-tuning summaries; recompute metrics when prediction arrays exist."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_finetuning import ROOT, MODELS, TASKS, load_result, recipe
from metrics import bootstrap_metric_scores, score_predictions
from hf_datasets import CLASSIFICATION_LABEL_NAMES, DETECTION_ONE_HOT_COLUMNS


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


def validate_manifest(manifest):
    expected = json.loads(json.dumps(dict(
        protocol="supervised_end_to_end_finetuning", models=MODELS,
        seed=42, num_training_seeds=1, num_bootstrap=1000,
        recipes={model[0]: {task: recipe(model[0], task) for task in TASKS} for model in MODELS},
        cuda_tf32=True)))
    if manifest != expected:
        raise ValueError("The release must match the retained six-model, three-task, seed-42 protocol.")


def verify(run_dir):
    root = Path(run_dir).resolve()
    with (root / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = {model[3] for model in MODELS}
    if len(rows) != len(MODELS) or {row["model_id"] for row in rows} != expected_ids:
        raise ValueError("Missing or duplicate models in the published table.")
    prediction_files = [root / model[0] / task / "seed_42/test_predictions.npz"
                        for model in MODELS for task in TASKS]
    bootstrap_files = [path.with_name("bootstrap_scores.npz") for path in prediction_files]
    presence = [path.is_file() for path in prediction_files + bootstrap_files]
    if any(presence) and not all(presence):
        raise ValueError("Prediction and bootstrap arrays must be complete or entirely omitted.")
    has_arrays = all(presence)
    reference = {}
    for model in MODELS:
        row = next(row for row in rows if row["model_id"] == model[3])
        if row.get("method") != model[1] or row.get("pretraining") != model[2]:
            raise ValueError("Model labels or pretraining descriptions disagree with the model registry.")
        if any(int(row[key]) != value for key, value in
               (("seed", 42), ("num_training_seeds", 1), ("num_bootstrap", 1000))):
            raise ValueError("Incompatible uncertainty metadata in the published table.")
        for task in TASKS:
            path = root / model[0] / task / "seed_42/results.json"
            report = load_result(path, model, task, 42, require_checkpoint=False,
                                 require_predictions=has_arrays)
            validate_report_details(report, task)
            metric = "map" if task == "detection" else "macro_f1"
            stored_bootstrap = report.get("test_bootstrap")
            if stored_bootstrap is not None and (
                    any(stored_bootstrap.get(key) != value for key, value in
                        (("num_resamples", 1000), ("seed", 42), ("ddof", 1), ("metric", metric)))
                    or not np.isclose(stored_bootstrap["std"], float(row[f"{task}_std"]), rtol=0, atol=1e-12)):
                raise ValueError(f"Bootstrap metadata disagrees with the CSV: {path}")
            if not has_arrays:
                score, std = float(row[f"{task}_score"]), float(row[f"{task}_std"])
                if (not np.isclose(score, report["test"][metric], rtol=0, atol=1e-7)
                        or not np.isfinite(std) or std < 0):
                    raise ValueError(f"Invalid reported summary: {path}")
                print(f"Checked summary {model[1]} {task}: {100 * score:.2f} ± {100 * std:.2f}", flush=True)
                continue
            metric_task = "detection" if task == "detection" else "classification"
            with np.load(path.with_name("test_predictions.npz"), allow_pickle=False) as data:
                y, predictions = data["y_true"], data["y_pred"]
            if (y.shape != predictions.shape or not len(y) or not np.isfinite(predictions).all()
                    or (task != "detection" and y.ndim != 1)
                    or (task == "detection" and (y.ndim != 2 or y.shape[1] != 7))):
                raise ValueError(f"Invalid predictions: {path}")
            if task in reference and not np.array_equal(y, reference[task]):
                raise ValueError(f"Different test target order: {path}")
            reference[task] = y
            score = score_predictions(metric_task, y, predictions)
            metric = "map" if task == "detection" else "macro_f1"
            samples = bootstrap_metric_scores(metric_task, y, predictions, 1000, 42)
            std = float(np.std(samples, ddof=1))
            with np.load(path.with_name("bootstrap_scores.npz"), allow_pickle=False) as data:
                if data["scores"].shape != (1000,) or not np.allclose(data["scores"], samples, rtol=0, atol=1e-12):
                    raise ValueError(f"Saved bootstrap scores disagree: {path}")
            if (not np.isclose(score, report["test"][metric], rtol=0, atol=1e-7)
                    or not np.isclose(score, float(row[f"{task}_score"]), rtol=0, atol=1e-7)
                    or not np.isclose(std, float(row[f"{task}_std"]), rtol=0, atol=1e-12)):
                raise ValueError(f"Published score or uncertainty disagrees: {path}")
            print(f"Verified {model[1]} {task}: {100 * score:.2f} ± {100 * std:.2f}", flush=True)
    if has_arrays:
        print("Verified all 18 fine-tuning evaluations.", flush=True)
    else:
        print("Checked the CSV against 18 summary reports. Predictions and bootstrap arrays "
              "are omitted; metrics and uncertainties were not recomputed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=ROOT / "results/manuscript")
    args = parser.parse_args()
    verify(args.run_dir)


if __name__ == "__main__":
    main()
