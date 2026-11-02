"""Export one fine-tuning CSV and the 18 detailed evaluation reports."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from finetuning_config import (
    PUBLISHED_RESULTS_DIR, MODELS, RESULT_MODEL_IDS, TASKS, TASK_DATASETS,
    SEED, NUM_BOOTSTRAP, CSV_FIELDS, run_manifest,
)
from report_io import write_report
from run_finetuning import load_result
from metrics import bootstrap_metric_scores, score_predictions


def validate_results(run_dir):
    """Validate CSV/reports and recompute scores when local arrays are retained."""
    root = Path(run_dir).resolve()
    with (root / "results.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("Unexpected fine-tuning CSV columns.")
        rows = list(reader)
    expected = {(RESULT_MODEL_IDS[model[0]], *TASK_DATASETS[task]) for model in MODELS for task in TASKS}
    keys = [(row.get("model_id"), row.get("dataset_name"), row.get("dataset_config")) for row in rows]
    if len(rows) != len(expected) or set(keys) != expected:
        raise ValueError("Missing or duplicate evaluations in the published results.")
    by_evaluation = dict(zip(keys, rows))
    prediction_files = [root / model[0] / task / "seed_42/test_predictions.npz"
                        for model in MODELS for task in TASKS]
    bootstrap_files = [path.with_name("bootstrap_scores.npz") for path in prediction_files]
    presence = [path.is_file() for path in prediction_files + bootstrap_files]
    if any(presence) and not all(presence):
        raise ValueError("Prediction and bootstrap arrays must be complete or entirely omitted.")
    has_arrays = all(presence)
    reference = {}
    for model in MODELS:
        for task in TASKS:
            row = by_evaluation[(RESULT_MODEL_IDS[model[0]], *TASK_DATASETS[task])]
            if row.get("model_label") != model[1]:
                raise ValueError("Model labels disagree with the benchmark registry.")
            if any(int(row[key]) != value for key, value in
                   (("seed", SEED), ("num_training_seeds", 1), ("num_bootstrap", NUM_BOOTSTRAP))):
                raise ValueError("Incompatible uncertainty metadata in the published results.")
            path = root / model[0] / task / "seed_42/results.json"
            report = load_result(path, model, task, 42, require_checkpoint=False,
                                 require_predictions=has_arrays)
            metric = "map" if task == "detection" else "macro_f1"
            if (row.get("metric") != ("mAP" if task == "detection" else "Macro-F1")
                    or int(row["best_epoch"]) != report["best_epoch"]
                    or not np.isclose(float(row["validation_score"]), report[f"validation_{metric}"], rtol=0, atol=1e-10)):
                raise ValueError(f"Evaluation metadata disagrees with the report: {path}")
            stored_bootstrap = report.get("test_bootstrap")
            if stored_bootstrap is not None and (
                    any(stored_bootstrap.get(key) != value for key, value in
                        (("num_resamples", 1000), ("seed", 42), ("ddof", 1), ("metric", metric)))
                    or not np.isclose(stored_bootstrap["std"], float(row["test_std"]), rtol=0, atol=1e-12)):
                raise ValueError(f"Bootstrap metadata disagrees with the CSV: {path}")
            if not has_arrays:
                score, std = float(row["test_mean"]), float(row["test_std"])
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
                    or not np.isclose(score, float(row["test_mean"]), rtol=0, atol=1e-7)
                    or not np.isclose(std, float(row["test_std"]), rtol=0, atol=1e-12)):
                raise ValueError(f"Published score or uncertainty disagrees: {path}")
            print(f"Verified {model[1]} {task}: {100 * score:.2f} ± {100 * std:.2f}", flush=True)
    if has_arrays:
        print("Verified all 18 fine-tuning evaluations.", flush=True)
    else:
        print("Checked the CSV against 18 summary reports. Predictions and bootstrap arrays "
              "are omitted; metrics and uncertainties were not recomputed.", flush=True)


def export(run_dir, output_dir):
    source, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing releases are not overwritten.")
    manifest = json.loads((source / "run_config.json").read_text())
    if manifest != run_manifest():
        raise ValueError("The release must match the retained six-model, three-task, seed-42 protocol.")
    validate_results(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finetuning-release-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shutil.copy2(source / "results.csv", staging / "results.csv")
        for model in MODELS:
            for task in TASKS:
                relative = Path(model[0]) / task / "seed_42"
                (staging / relative).mkdir(parents=True)
                report = json.loads((source / relative / "results.json").read_text())
                write_report(staging / relative / "results.json", report)
        validate_results(staging)
        staging.rename(output)
    print(f"Exported verified fine-tuning release: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=PUBLISHED_RESULTS_DIR)
    args = parser.parse_args()
    export(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
