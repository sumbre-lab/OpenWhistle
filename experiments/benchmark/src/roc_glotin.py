#!/usr/bin/env python3
"""ROC for frozen-backbone OpenWhistle detection probes.

Uses the benchmark's cached embeddings, chooses logistic-regression C by
validation mAP, refits on train+validation, and evaluates seven binary heads
on the held-out test set. Run from the repository root:
    python experiments/benchmark/src/roc_glotin.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from hf_datasets import (CLASSIFICATION_DATASET_ID, DETECTION_DATASET_ID,
                         DETECTION_ONE_HOT_COLUMNS)
from metrics import MeanAveragePrecision
from pipeline_config import EMBEDDING_PIPELINE_VERSION

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "dolphinteam/OpenWhistle-Wav2Vec2.0"
C_GRID = (0.1, 1.0, 10.0)
SEED = 42
CLASSIFICATION_LABEL_NAMES = (
    "NSW_3", "NSW_2", "NSW_1", "SW_Dana", "SW_Luna",
    "SW_Nana", "SW_Neo", "SW_Nikita", "SW_Shy", "SW_Yosefa",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-name", choices=("detection", "classification"),
                        default="detection")
    parser.add_argument("--classification-configs", nargs="+",
                        choices=("balanced", "unbalanced", "all"),
                        default=("all", "unbalanced"))
    parser.add_argument("--results-csv", type=Path,
                        default=ROOT / "results/hf_collections_benchmark.csv")
    parser.add_argument("--embedding-cache-dir", type=Path,
                        default=ROOT / "results/embedding_cache")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_result(path: Path, model_id: str, dataset_name: str,
                dataset_config: str, metric: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["model_id"] == model_id
                and row["dataset_name"] == dataset_name
                and row["dataset_config"] == dataset_config
                and row["metric"] == metric
                and row["embedding_pipeline_version"] == EMBEDDING_PIPELINE_VERSION]
    if len(rows) != 1:
        raise ValueError(f"Expected one matching benchmark row, found {len(rows)} in {path}")
    return rows[0]


def load_splits(cache_dir: Path, model_id: str, feature_mode: str,
                dataset_name: str = "detection", dataset_config: str = "default"):
    cache_id = model_id if feature_mode == "standard" else f"{model_id}__{feature_mode}"
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "__", cache_id)
    root = cache_dir / EMBEDDING_PIPELINE_VERSION / safe_id / dataset_name / dataset_config
    splits = {}
    for name in ("train", "validation", "test"):
        path = root / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; generate benchmark embeddings first")
        with np.load(path, allow_pickle=False) as data:
            x, y = np.asarray(data["x"]), np.asarray(data["y"])
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError(f"Invalid embeddings or labels in {path}")
        if dataset_name == "detection":
            valid_labels = (y.ndim == 2 and y.shape[1] == len(DETECTION_ONE_HOT_COLUMNS)
                            and np.isin(y, (0, 1)).all())
        else:
            valid_labels = (y.ndim == 1 and np.isin(
                y, np.arange(len(CLASSIFICATION_LABEL_NAMES))).all())
        if not valid_labels:
            raise ValueError(f"Invalid labels in {path}")
        splits[name] = (x, y.astype(np.int64, copy=False))
    if len({x.shape[1] for x, _ in splits.values()}) != 1:
        raise ValueError("Embedding dimensions differ across splits")
    return splits


def fit_scores(x_train, y_train, x_eval, c: float, normalize: bool):
    if normalize:
        scaler = StandardScaler()
        x_train, x_eval = scaler.fit_transform(x_train), scaler.transform(x_eval)
    head = LogisticRegression(max_iter=20000, random_state=SEED, C=c)
    clf = MultiOutputClassifier(head).fit(x_train, y_train)
    columns = []
    for estimator, probabilities in zip(clf.estimators_, clf.predict_proba(x_eval)):
        positive_index = np.flatnonzero(estimator.classes_ == 1)
        if positive_index.size != 1:
            raise ValueError("A detection head is missing its positive class")
        columns.append(probabilities[:, positive_index[0]])
    return np.column_stack(columns)


def map_score(y, scores) -> float:
    metric = MeanAveragePrecision()
    metric.update(scores, y)
    return float(metric.get_primary_metric())


def run_detection(args):
    result = load_result(args.results_csv, args.model_id,
                         "detection", "default", "mAP")
    splits = load_splits(args.embedding_cache_dir, args.model_id, result["feature_mode"])
    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]
    x_test, y_test = splits["test"]
    normalize = result["normalize_data"] == "True"

    validation_scores = []
    for c in C_GRID:
        scores = fit_scores(x_train, y_train, x_val, c, normalize)
        validation_scores.append({"c": c, "map": map_score(y_val, scores)})
    selected = max(validation_scores, key=lambda value: value["map"])
    if (selected["c"] != float(result["best_c"])
            or abs(selected["map"] - float(result["validation_score"])) > 1e-3):
        raise ValueError("Validation result differs from the existing benchmark")

    x_final = np.concatenate((x_train, x_val))
    y_final = np.concatenate((y_train, y_val))
    scores = fit_scores(x_final, y_final, x_test, selected["c"], normalize)
    test_map = map_score(y_test, scores)
    if abs(test_map - float(result["test_mean"])) > 1e-3:
        raise ValueError(f"Test mAP {test_map:.6f} differs from benchmark {result['test_mean']}")

    curves, points = [], []
    sources = [("micro", y_test.ravel(), scores.ravel())]
    sources += [(label, y_test[:, i], scores[:, i])
                for i, label in enumerate(DETECTION_ONE_HOT_COLUMNS)]
    for label, truth, probability in sources:
        positives = int(truth.sum())
        negatives = int(len(truth) - positives)
        if not positives or not negatives:
            raise ValueError(f"ROC undefined for {label}: only one test class")
        fpr, tpr, thresholds = roc_curve(truth, probability, pos_label=1)
        curves.append({"label": label, "n_positive": positives,
                       "n_negative": negatives, "roc_auc": float(auc(fpr, tpr)),
                       "fpr": fpr, "tpr": tpr})
        points.extend({"label": label, "point_index": i, "threshold": float(th),
                       "false_positive_rate": float(fp), "true_positive_rate": float(tp)}
                      for i, (th, fp, tp) in enumerate(zip(thresholds, fpr, tpr)))

    micro_auc = curves[0]["roc_auc"]
    macro_auc = float(np.mean([curve["roc_auc"] for curve in curves[1:]]))
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "__", args.model_id)
    out_dir = args.out_dir or ROOT / "results/roc_curves" / safe_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "roc_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "label", "point_index", "threshold", "false_positive_rate",
            "true_positive_rate"))
        writer.writeheader()
        writer.writerows(points)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot((0, 1), (0, 1), "--", color="0.55", linewidth=1, label="Chance")
    for curve in curves:
        label = curve["label"]
        ax.plot(curve["fpr"], curve["tpr"],
                linewidth=2.5 if label == "micro" else 1.4,
                label=f'{label} (AUC={curve["roc_auc"]:.3f})')
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate",
           title=f"{args.model_id} — detection ROC on held-out test")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_detection.png", dpi=200)
    fig.savefig(out_dir / "roc_detection.pdf")
    plt.close(fig)

    summary = {
        "model_id": args.model_id, "dataset_id": DETECTION_DATASET_ID,
        "dataset_config": "default", "split": "test",
        "task": "seven-label whistle-type detection",
        "score": "positive-class probability from each logistic detection head",
        "embedding_pipeline_version": EMBEDDING_PIPELINE_VERSION,
        "feature_mode": result["feature_mode"],
        "standardize_embeddings": normalize, "seed": SEED,
        "c_selection_metric": "validation mAP", "c_grid": list(C_GRID),
        "best_c": selected["c"], "validation_scores": validation_scores,
        "test_map": test_map,
        "n_examples": {name: len(y) for name, (_, y) in splits.items()},
        "micro_roc_auc": micro_auc, "macro_roc_auc": macro_auc,
        "per_label": [{key: curve[key] for key in (
            "label", "n_positive", "n_negative", "roc_auc")}
            for curve in curves[1:]],
    }
    with (out_dir / "roc_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Best C: {selected['c']}; validation mAP: {selected['map']:.6f}")
    print(f"Test mAP: {test_map:.6f}")
    print(f"Micro ROC-AUC: {micro_auc:.6f}; macro ROC-AUC: {macro_auc:.6f}")
    print(f"Outputs: {out_dir}")


def run_classification(args, config: str):
    result = load_result(args.results_csv, args.model_id,
                         "classification", config, "Macro-F1")
    splits = load_splits(args.embedding_cache_dir, args.model_id,
                         result["feature_mode"], "classification", config)
    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]
    x_test, y_test = splits["test"]
    normalize = result["normalize_data"] == "True"
    classes = np.arange(len(CLASSIFICATION_LABEL_NAMES))
    if not all(np.array_equal(np.unique(y), classes) for _, y in splits.values()):
        raise ValueError(f"Missing classes in classification/{config} split")

    def fit(x_fit, y_fit, x_eval, c):
        if normalize:
            scaler = StandardScaler()
            x_fit, x_eval = scaler.fit_transform(x_fit), scaler.transform(x_eval)
        clf = LogisticRegression(max_iter=20000, random_state=SEED, C=c)
        clf.fit(x_fit, y_fit)
        if not np.array_equal(clf.classes_, classes):
            raise ValueError("Classifier probability columns do not match class labels")
        return clf.predict(x_eval), clf.predict_proba(x_eval)

    validation_scores = []
    for c in C_GRID:
        predictions, _ = fit(x_train, y_train, x_val, c)
        validation_scores.append({
            "c": c,
            "macro_f1": float(f1_score(y_val, predictions, average="macro",
                                       zero_division=0)),
        })
    selected = max(validation_scores, key=lambda value: value["macro_f1"])
    if (selected["c"] != float(result["best_c"])
            or abs(selected["macro_f1"] - float(result["validation_score"])) > 1e-3):
        raise ValueError(f"Validation result differs from benchmark for {config}")

    x_final = np.concatenate((x_train, x_val))
    y_final = np.concatenate((y_train, y_val))
    predictions, scores = fit(x_final, y_final, x_test, selected["c"])
    test_macro_f1 = float(f1_score(y_test, predictions, average="macro",
                                   zero_division=0))
    if abs(test_macro_f1 - float(result["test_mean"])) > 1e-3:
        raise ValueError(
            f"Test macro-F1 {test_macro_f1:.6f} differs from benchmark "
            f'{result["test_mean"]} for {config}'
        )

    binary_truth = (y_test[:, None] == classes[None, :]).astype(np.int64)
    sources = [("micro", binary_truth.ravel(), scores.ravel())]
    sources += [(label, binary_truth[:, i], scores[:, i])
                for i, label in enumerate(CLASSIFICATION_LABEL_NAMES)]
    curves, points = [], []
    for label, truth, probability in sources:
        positives = int(truth.sum())
        negatives = int(len(truth) - positives)
        if not positives or not negatives:
            raise ValueError(f"ROC undefined for {label}: only one test class")
        fpr, tpr, thresholds = roc_curve(truth, probability, pos_label=1)
        curves.append({"label": label, "n_positive": positives,
                       "n_negative": negatives, "roc_auc": float(auc(fpr, tpr)),
                       "fpr": fpr, "tpr": tpr})
        points.extend({"label": label, "point_index": i, "threshold": float(th),
                       "false_positive_rate": float(fp), "true_positive_rate": float(tp)}
                      for i, (th, fp, tp) in enumerate(zip(thresholds, fpr, tpr)))

    micro_auc = curves[0]["roc_auc"]
    macro_auc = float(np.mean([curve["roc_auc"] for curve in curves[1:]]))
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "__", args.model_id)
    base_dir = args.out_dir or ROOT / "results/roc_curves" / safe_id
    out_dir = base_dir / f"classification_{config}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "roc_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "label", "point_index", "threshold", "false_positive_rate",
            "true_positive_rate"))
        writer.writeheader()
        writer.writerows(points)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot((0, 1), (0, 1), "--", color="0.55", linewidth=1, label="Chance")
    for curve in curves:
        label = curve["label"]
        ax.plot(curve["fpr"], curve["tpr"],
                linewidth=2.5 if label == "micro" else 1.3,
                label=f'{label} (AUC={curve["roc_auc"]:.3f})')
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate",
           title=f"Whistle classification ROC: {config} test")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_classification.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "roc_classification.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model_id": args.model_id,
        "dataset_id": CLASSIFICATION_DATASET_ID,
        "dataset_config": config, "split": "test",
        "task": "ten-class one-vs-rest whistle classification",
        "score": "class probability from multinomial logistic regression",
        "embedding_pipeline_version": EMBEDDING_PIPELINE_VERSION,
        "feature_mode": result["feature_mode"],
        "standardize_embeddings": normalize, "seed": SEED,
        "c_selection_metric": "validation macro-F1", "c_grid": list(C_GRID),
        "best_c": selected["c"], "validation_scores": validation_scores,
        "test_macro_f1": test_macro_f1,
        "n_examples": {name: len(y) for name, (_, y) in splits.items()},
        "micro_roc_auc": micro_auc, "macro_roc_auc": macro_auc,
        "per_label": [{key: curve[key] for key in (
            "label", "n_positive", "n_negative", "roc_auc")}
            for curve in curves[1:]],
    }
    with (out_dir / "roc_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"{config}: best C={selected['c']}; "
          f"test macro-F1={test_macro_f1:.6f}; "
          f"micro AUC={micro_auc:.6f}; macro AUC={macro_auc:.6f}")
    print(f"Outputs: {out_dir}")


def main():
    args = arguments()
    if args.dataset_name == "detection":
        run_detection(args)
    else:
        for config in args.classification_configs:
            run_classification(args, config)


if __name__ == "__main__":
    main()
