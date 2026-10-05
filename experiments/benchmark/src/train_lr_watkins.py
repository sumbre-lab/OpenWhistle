"""Re-evaluate cached Watkins embeddings with validation macro-F1 selection.

Preserves the historical Watkins probe except for its selection metric.
No audio extraction, download, or test-driven hyperparameter selection.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

CS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, .1, .3, 1., 3., 10., 30., 100.]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(cache):
    frame = pd.read_csv(cache / "embedding_index.csv")
    required = {"audio_path", "label", "split"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("Index requires nonempty audio_path, label and split columns")
    frame["split"] = frame["split"].replace({"validation": "valid", "val": "valid"})
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise ValueError("Expected nonempty train, valid and test splits only")
    if frame["audio_path"].duplicated().any():
        raise ValueError("Duplicate audio paths: possible split leakage")
    # Portable IDs preserve the class directory and filename in these caches.
    frame["sample_id"] = frame["audio_path"].map(lambda p: "/".join(Path(p).parts[-2:]))
    if frame["sample_id"].duplicated().any():
        raise ValueError("Sample IDs are not unique")
    labels = set(frame["label"])
    for split in ("train", "valid", "test"):
        if set(frame.loc[frame["split"] == split, "label"]) != labels:
            raise ValueError(f"Every class must appear in {split}")
    embeddings = np.load(cache / "embeddings.npy", allow_pickle=False)
    if embeddings.ndim != 2 or len(embeddings) != len(frame) or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be a finite matrix aligned with the index")
    return frame, embeddings


def select_candidate(sweep, metric="valid_macro_f1"):
    # Stable tie rule: first C in the ascending grid.
    return max(sweep, key=lambda row: row[metric])


def bootstrap_macro_f1(y, predictions, labels, count, seed):
    rng = np.random.default_rng(seed)
    values = np.empty(count)
    for i in range(count):
        indices = rng.integers(0, len(y), size=len(y))
        values[i] = f1_score(y[indices], predictions[indices], labels=labels,
                             average="macro", zero_division=0)
    return values


def evaluate(cache, output, bootstrap_count, seed):
    frame, embeddings = load_cache(cache)
    encoder = LabelEncoder().fit(frame["label"])
    y = encoder.transform(frame["label"])
    labels = np.arange(len(encoder.classes_))
    masks = {split: frame["split"].eq(split).to_numpy() for split in ("train", "valid", "test")}
    scaler = StandardScaler().fit(embeddings[masks["train"]])
    x_train = scaler.transform(embeddings[masks["train"]])
    x_valid = scaler.transform(embeddings[masks["valid"]])
    sweep = []
    for c in CS:
        classifier = LogisticRegression(C=c, max_iter=20000, solver="lbfgs",
                                        class_weight="balanced", random_state=seed)
        classifier.fit(x_train, y[masks["train"]])
        predictions = classifier.predict(x_valid)
        sweep.append({"C": c, "valid_accuracy": float(accuracy_score(y[masks["valid"]], predictions)),
                      "valid_macro_f1": float(f1_score(y[masks["valid"]], predictions, labels=labels,
                                                       average="macro", zero_division=0)),
                      "classifier": classifier})
        print(f"{cache.name}: C={c:g}, validation macro-F1={sweep[-1]['valid_macro_f1']:.6f}", flush=True)
    best = select_candidate(sweep)
    historical = select_candidate(sweep, "valid_accuracy")
    x_test = scaler.transform(embeddings[masks["test"]])
    y_test = y[masks["test"]]
    predictions = best["classifier"].predict(x_test)
    historical_predictions = historical["classifier"].predict(x_test)
    samples = bootstrap_macro_f1(y_test, predictions, labels, bootstrap_count, seed)
    metrics = {
        "dataset": "watkins", "model_name": cache.name, "primary_metric": "macro_f1",
        "selection_metric": "valid_macro_f1", "best_C": best["C"],
        "valid_macro_f1": best["valid_macro_f1"], "valid_accuracy": best["valid_accuracy"],
        "test_macro_f1": float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "n_total": len(frame), "n_classes": len(labels),
        **{f"n_{s}": int(mask.sum()) for s, mask in masks.items()},
        "bootstrap": {"count": bootstrap_count, "seed": seed, "sampling": "test clips, with replacement, unstratified",
                      "class_universe": encoder.classes_.tolist(), "zero_division": 0, "std_ddof": 0,
                      "mean": float(samples.mean()), "std": float(samples.std(ddof=0)),
                      "percentile_ci_95": np.percentile(samples, [2.5, 97.5]).tolist()},
        "historical_accuracy_selection_recomputed": {
            "best_C": historical["C"], "test_accuracy": float(accuracy_score(y_test, historical_predictions)),
            "test_macro_f1": float(f1_score(y_test, historical_predictions, labels=labels, average="macro", zero_division=0))},
    }
    historical_file = cache / "metrics.json"
    if historical_file.exists():
        recorded = json.loads(historical_file.read_text())
        differences = {}
        for key in ("best_C", "test_accuracy", "test_macro_f1"):
            actual = metrics["historical_accuracy_selection_recomputed"][key]
            if not np.isclose(actual, recorded[key], rtol=0, atol=1e-10):
                differences[key] = {"recomputed": actual, "recorded": recorded[key]}
        metrics["historical_comparison"] = {"exact_match": not differences, "differences": differences}
        if differences:
            print(f"Historical reconstruction differs for {cache.name}: {differences}. "
                  "These are new probe results; do not describe them as an exact historical reproduction.", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame([{k: v for k, v in row.items() if k != "classifier"} for row in sweep]).to_csv(output / "c_sweep.csv", index=False)
    test = frame.loc[masks["test"], ["sample_id", "label", "split"]].copy()
    test["prediction"] = encoder.inverse_transform(predictions)
    test["historical_accuracy_selected_prediction"] = encoder.inverse_transform(historical_predictions)
    test.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame({"macro_f1": samples}).to_csv(output / "bootstrap.csv", index=False)
    manifest = frame[["sample_id", "label", "split"]]
    manifest.to_csv(output / "split_manifest.csv", index=False)
    provenance = {
        "inputs_sha256": {name: sha256(cache / name) for name in ("embeddings.npy", "embedding_index.csv")},
        "runner_sha256": sha256(Path(__file__)), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "thread_pools": threadpool_info(),
        "protocol": {"C_grid": CS, "solver": "lbfgs", "max_iter": 20000, "class_weight": "balanced",
                     "random_state": seed, "tol": 1e-4,
                     "standard_scaler_fit": "train", "probe_fit": "train",
                     "refit_train_valid": False, "tie_break": "first C in ascending grid",
                     "embedding_extraction": "external historical caches; not rerun by this script"},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if historical_file.exists():
        (output / "historical_metrics.json").write_text(historical_file.read_text())
    return metrics, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["dolphinteam_OpenWhistle_Wav2Vec2.0", "aves_bio_rerun"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.num_bootstrap < 2 or args.threads < 1:
        parser.error("Need at least 2 bootstrap samples and 1 thread")
    # Validate all models and their exact row alignment before fitting.
    reference = None
    for model in args.models:
        frame, _ = load_cache(args.cache_root / model)
        manifest = frame[["sample_id", "label", "split"]]
        if reference is not None and not reference.equals(manifest):
            raise ValueError("Model caches have different sample order, labels or splits")
        reference = manifest
    results = []
    with threadpool_limits(limits=args.threads):
        for model in args.models:
            metrics, _ = evaluate(args.cache_root / model, args.output_dir / model, args.num_bootstrap, args.seed)
            results.append({"model": model, "best_C": metrics["best_C"], "test_macro_f1_pct": 100 * metrics["test_macro_f1"],
                            "bootstrap_std_pct": 100 * metrics["bootstrap"]["std"], "test_accuracy_pct": 100 * metrics["test_accuracy"]})
    pd.DataFrame(results).to_csv(args.output_dir / "summary.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
