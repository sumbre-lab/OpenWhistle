import argparse
from pathlib import Path

import numpy as np
from hf_datasets import load_classification_splits
from plot_pr_micro import get_audio_model
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from train_lr_splits import (
    embed_split,
    evaluate_classifier,
    fit_classifier,
    normalize_train_eval,
    set_seed,
)

DATASET_NAME = "classification"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--inverse_reg", default=1.0, type=float)
    parser.add_argument(
        "--inverse_regs",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Candidate C values. The best C is selected on the validation split. "
            "Defaults to --inverse_reg for backward compatibility."
        ),
    )
    parser.add_argument(
        "--num_seeds",
        default=10,
        type=int,
        help="Number of final logistic-regression seeds evaluated on the test split.",
    )
    parser.add_argument(
        "--num_bootstrap",
        default=1000,
        type=int,
        help=(
            "Number of bootstrap resamples of the test set used to estimate "
            "the reported standard deviation."
        ),
    )

    parser.add_argument(
        "--no_normalize_data",
        dest="normalize_data",
        action="store_false",
        default=True,
        help="Disable feature standardization before logistic regression.",
    )

    parser.add_argument(
        "--model",
        choices=[
            "dolph2vec",
            "aves_core",
            "aves_bio",
            "biolingual",
            "mfcc",
            "spectrogram",
            "spectral_features",
        ],
        default="dolph2vec",
    )

    parser.add_argument("--target_sample_rate", default=44100, type=int)

    parser.add_argument(
        "--held_out_class",
        default="SW_Luna",
        help="Class name excluded from training/validation and treated as unknown.",
    )
    parser.add_argument(
        "--known_acceptance_rate",
        default=0.95,
        type=float,
        help=(
            "Fraction of known validation samples that should be accepted (not "
            "rejected as unknown) when calibrating the rejection threshold."
        ),
    )

    return parser.parse_args()


def split_known_unknown(x, y, held_out_id: int):
    known_mask = y != held_out_id
    return x[known_mask], y[known_mask], x[~known_mask], y[~known_mask]


def compute_openset_predictions(clf, x_eval, threshold: float, held_out_id: int):
    raw_pred = clf.predict(x_eval)
    unknown_score = 1.0 - clf.predict_proba(x_eval).max(axis=1)
    final_pred = np.where(unknown_score > threshold, held_out_id, raw_pred)
    return raw_pred, unknown_score, final_pred


def openset_metrics(y_true, raw_pred, unknown_score, final_pred, held_out_id, label_universe):
    is_unknown_true = (y_true == held_out_id).astype(int)
    known_mask = ~is_unknown_true.astype(bool)

    metrics = {
        "auroc": float(roc_auc_score(is_unknown_true, unknown_score)),
        "auprc": float(average_precision_score(is_unknown_true, unknown_score)),
        "open_set_f1": float(
            f1_score(y_true, final_pred, labels=label_universe, average="macro")
        ),
    }
    if known_mask.any():
        metrics["closed_set_accuracy"] = float(
            accuracy_score(y_true[known_mask], raw_pred[known_mask])
        )
    else:
        metrics["closed_set_accuracy"] = float("nan")
    return metrics


def bootstrap_openset_metrics(
    y_true,
    raw_pred,
    unknown_score,
    final_pred,
    held_out_id,
    label_universe,
    num_bootstrap: int,
    seed: int,
):
    if num_bootstrap <= 0:
        return {"auroc": [], "auprc": [], "open_set_f1": [], "closed_set_accuracy": []}

    rng = np.random.default_rng(seed)
    n_examples = len(y_true)
    is_unknown_true = (y_true == held_out_id).astype(int)

    bootstrap_scores = {"auroc": [], "auprc": [], "open_set_f1": [], "closed_set_accuracy": []}
    for _ in range(num_bootstrap):
        indices = rng.integers(0, n_examples, size=n_examples)
        if len(np.unique(is_unknown_true[indices])) < 2:
            continue
        scores = openset_metrics(
            y_true[indices],
            raw_pred[indices],
            unknown_score[indices],
            final_pred[indices],
            held_out_id,
            label_universe,
        )
        for key, value in scores.items():
            bootstrap_scores[key].append(value)
    return bootstrap_scores


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def main():
    args = get_args()
    set_seed(args.seed)
    inverse_regs = (
        args.inverse_regs if args.inverse_regs is not None else [args.inverse_reg]
    )

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model = get_audio_model(args.model, args.target_sample_rate)

    dataset, label_feature = load_classification_splits()
    if args.held_out_class not in label_feature.names:
        raise ValueError(
            f"Held-out class {args.held_out_class!r} not found in classification "
            f"label names: {label_feature.names}"
        )
    held_out_id = label_feature.str2int(args.held_out_class)

    x_train, y_train = embed_split(dataset["train"], "train", DATASET_NAME, model)
    x_validation, y_validation = embed_split(
        dataset["validation"], "validation", DATASET_NAME, model
    )
    x_test, y_test = embed_split(dataset["test"], "test", DATASET_NAME, model)

    x_train_known, y_train_known, _, _ = split_known_unknown(
        x_train, y_train, held_out_id
    )
    x_validation_known, y_validation_known, _, _ = split_known_unknown(
        x_validation, y_validation, held_out_id
    )
    num_unknown_test = int(np.sum(y_test == held_out_id))
    if num_unknown_test == 0:
        raise ValueError(
            f"No {args.held_out_class!r} samples found in the test split; cannot "
            "evaluate open-set detection."
        )

    print(
        f"train: {len(y_train_known)} known / {len(y_train) - len(y_train_known)} unknown, "
        f"validation: {len(y_validation_known)} known / "
        f"{len(y_validation) - len(y_validation_known)} unknown, "
        f"test: {len(y_test) - num_unknown_test} known / {num_unknown_test} unknown"
    )

    x_train_for_validation, x_validation_for_selection = normalize_train_eval(
        x_train_known, x_validation_known, args.normalize_data
    )
    validation_scores = []
    calibration_clfs = {}
    for c in inverse_regs:
        clf = fit_classifier(
            DATASET_NAME, args.seed, c, x_train_for_validation, y_train_known
        )
        score = evaluate_classifier(
            DATASET_NAME, clf, x_validation_for_selection, y_validation_known
        )
        validation_scores.append((c, score))
        calibration_clfs[c] = clf

    best_c, best_validation_score = max(validation_scores, key=lambda item: item[1])
    calibration_clf = calibration_clfs[best_c]

    calibration_unknown_score = 1.0 - calibration_clf.predict_proba(
        x_validation_for_selection
    ).max(axis=1)
    threshold = float(
        np.quantile(calibration_unknown_score, args.known_acceptance_rate)
    )

    x_final_train = np.concatenate([x_train_known, x_validation_known], axis=0)
    y_final_train = np.concatenate([y_train_known, y_validation_known], axis=0)
    x_final_train, x_test_for_evaluation = normalize_train_eval(
        x_final_train, x_test, args.normalize_data
    )

    final_metrics = {"auroc": [], "auprc": [], "open_set_f1": [], "closed_set_accuracy": []}
    final_bootstrap_metrics = {
        "auroc": [],
        "auprc": [],
        "open_set_f1": [],
        "closed_set_accuracy": [],
    }
    seeds = [args.seed + offset for offset in range(args.num_seeds)]
    for seed in seeds:
        set_seed(seed)
        clf = fit_classifier(DATASET_NAME, seed, best_c, x_final_train, y_final_train)
        label_universe = sorted(set(clf.classes_.tolist()) | {held_out_id})

        raw_pred, unknown_score, final_pred = compute_openset_predictions(
            clf, x_test_for_evaluation, threshold, held_out_id
        )
        scores = openset_metrics(
            y_test, raw_pred, unknown_score, final_pred, held_out_id, label_universe
        )
        for key, value in scores.items():
            final_metrics[key].append(value)

        bootstrap_scores = bootstrap_openset_metrics(
            y_test,
            raw_pred,
            unknown_score,
            final_pred,
            held_out_id,
            label_universe,
            args.num_bootstrap,
            seed,
        )
        for key, values in bootstrap_scores.items():
            final_bootstrap_metrics[key].extend(values)

    reported = {}
    for key in final_metrics:
        seed_mean, seed_std = mean_std(final_metrics[key])
        if final_bootstrap_metrics[key]:
            _, bootstrap_std = mean_std(final_bootstrap_metrics[key])
            std = bootstrap_std
        else:
            std = seed_std
        reported[key] = (seed_mean, std)

    auroc_mean, auroc_std = reported["auroc"]
    auprc_mean, auprc_std = reported["auprc"]
    f1_mean, f1_std = reported["open_set_f1"]
    acc_mean, acc_std = reported["closed_set_accuracy"]

    result_text = (
        f"Model {args.model} open-set (known vs unknown={args.held_out_class}) "
        f"on dataset {DATASET_NAME}:\n"
        f"Best C = {best_c} "
        f"(validation known-class Accuracy = {best_validation_score:.4f})\n"
        f"Rejection threshold (known-acceptance rate={args.known_acceptance_rate}) "
        f"= {threshold:.4f}\n"
        f"Open-Set Test AUROC (known vs unknown): {auroc_mean:.4f} ± {auroc_std:.4f}\n"
        f"Open-Set Test AUPRC (unknown detection): {auprc_mean:.4f} ± {auprc_std:.4f}\n"
        f"Open-Set Test Macro F1 (K known + unknown): {f1_mean:.4f} ± {f1_std:.4f}\n"
        f"Closed-Set Test Accuracy (known classes only): {acc_mean:.4f} ± {acc_std:.4f}"
    )
    print(result_text)

    with open(results_dir / "results_openset.txt", "a") as f:
        print(result_text, file=f)


if __name__ == "__main__":
    main()
