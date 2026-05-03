#!/usr/bin/env python3
"""
Build a single figure: micro-averaged precision–recall (OvR, pooled) per model.
Uses the same split protocol as train_lr_splits.py (no k-fold):

- Train / validation / test from load_*_splits().
- C is chosen by validation accuracy (classification) or validation mAP (detection)
  over --inverse_regs, with the scaler fit on training data only for that stage.
- Final logistic regression is fit on train+validation (with scaler fit on that union),
  then micro PR is computed on the test split probability estimates.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hf_datasets import load_classification_splits, load_detection_splits
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from conf import (
    dolph2vec_base,
    dolph2vec_config_path,
    get_aves_paths,
    get_aves_sample_rate,
)
from models import (
    MFCC,
    Aves,
    BioLingual,
    Dolph2Vec,
    SpectralFeatures,
    Spectrogram,
)
from train_lr_splits import (
    detection_probabilities,
    embed_split,
    evaluate_classifier,
    fit_classifier,
    normalize_train_eval,
    set_seed,
)


def color_for_model(model_name: str) -> str:
    """Fixed colors for common backbones; unknown models use gray."""
    n = model_name.lower()
    if "dolph2vec" in n:
        return "#c62828"  # red
    if n == "spectrogram":
        return "#ef6c00"  # orange
    if n == "mfcc":
        return "#1565c0"  # blue
    if "spectral" in n:
        return "#6a1b9a"  # purple
    return "#757575"


def chance_level_precision(y: np.ndarray, is_multilabel: bool) -> float:
    """
    Horizontal baseline for micro PR: prevalence of positives after flattening.
    Multilabel: mean of all binary labels.
    Multiclass (micro OvR): one positive per row over K classes → 1/K.
    """
    if is_multilabel:
        return float(np.mean(y))
    n_classes = len(np.unique(y))
    if n_classes < 2:
        return 1.0
    return 1.0 / float(n_classes)


def get_audio_model(model_name: str, target_sample_rate: int):
    name2model = {
        "aves_core": Aves,
        "aves_bio": Aves,
        "aves_ow": Aves,
        "biolingual": BioLingual,
        "dolph2vec": Dolph2Vec,
        "mfcc": MFCC,
        "spectrogram": Spectrogram,
        "spectral_features": SpectralFeatures,
    }
    if model_name == "aves_bio":
        amodel_path, aconfig = get_aves_paths("bio")
        target_sample_rate = get_aves_sample_rate("bio")
    elif model_name == "aves_core":
        amodel_path, aconfig = get_aves_paths("core")
        target_sample_rate = get_aves_sample_rate("core")
    elif model_name == "aves_ow":
        amodel_path, aconfig = get_aves_paths("ow")
        target_sample_rate = get_aves_sample_rate("ow")
    else:
        amodel_path, aconfig = "", ""
    model_args = dict(
        sample_rate=target_sample_rate,
        dolph2vec_config_path=dolph2vec_config_path,
        dolph2vec_model_path=dolph2vec_base,
    )
    model_args["aves_model_path"] = amodel_path
    model_args["aves_config_path"] = aconfig
    return name2model[model_name](**model_args)


def micro_pr_curve(y_true: np.ndarray, y_score: np.ndarray, classes: np.ndarray):
    """Micro-averaged PR curve (OvR binarized targets, scores flattened)."""
    y_b = label_binarize(y_true, classes=classes)
    if y_b.shape[1] == 1:
        y_b = np.hstack([1 - y_b, y_b])
        if y_score.shape[1] == 1:
            y_score = np.hstack([1 - y_score, y_score])
    precision, recall, _ = precision_recall_curve(y_b.ravel(), y_score.ravel())
    ap = average_precision_score(y_true, y_score, average="micro")
    return precision, recall, ap


def micro_pr_curve_multilabel(y_true: np.ndarray, y_score: np.ndarray):
    """Micro-averaged PR for multilabel (flatten all label dimensions)."""
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    ap = average_precision_score(y_true, y_score, average="micro")
    return precision, recall, ap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="Embedding / numpy seed.")
    p.add_argument(
        "--lr_seed",
        default=None,
        type=int,
        help=(
            "Random state for the final logistic regression fit used for the PR curve "
            "(same role as the first seed in train_lr_splits --num_seeds). "
            "Defaults to --seed."
        ),
    )
    p.add_argument(
        "--no_normalize_data",
        dest="normalize_data",
        action="store_false",
        default=True,
        help="Disable feature standardization before logistic regression.",
    )
    p.add_argument(
        "--dataset_name",
        default="classification",
        choices=[
            "classification",
            "detection",
        ],
        help="detection uses multilabel micro PR; classification uses single-label multiclass micro PR.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["mfcc", "spectrogram", "spectral_features", "dolph2vec"],
    )
    p.add_argument(
        "--inverse_regs",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help="Candidate C values; best is chosen on the validation split (accuracy or mAP).",
    )
    p.add_argument("--target_sample_rate", type=int, default=44100)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Default: results/pr_curves under the experiments/benchmark folder.",
    )
    p.add_argument(
        "--out_name",
        default="pr_micro_models.png",
        help="Output filename under out_dir.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    lr_seed = args.lr_seed if args.lr_seed is not None else args.seed
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    benchmark_root = script_dir.parent
    out_dir = args.out_dir or (benchmark_root / "results" / "pr_curves")
    out_dir.mkdir(parents=True, exist_ok=True)

    is_detection = args.dataset_name == "detection"
    if is_detection:
        dataset, _ = load_detection_splits()
    else:
        dataset, _ = load_classification_splits()

    task = "multilabel_detection" if is_detection else "multiclass_micro_ovr"

    fig, ax = plt.subplots(figsize=(8, 6.5))

    curve_records: list[dict] = []
    chance_prec: float | None = None
    for model_name in args.models:
        set_seed(args.seed)
        model = get_audio_model(model_name, args.target_sample_rate)

        x_train, y_train = embed_split(
            dataset["train"], "train", args.dataset_name, model
        )
        x_validation, y_validation = embed_split(
            dataset["validation"], "validation", args.dataset_name, model
        )
        x_test, y_test = embed_split(dataset["test"], "test", args.dataset_name, model)

        if chance_prec is None:
            if is_detection:
                y_all = np.vstack([y_train, y_validation, y_test])
            else:
                y_all = np.concatenate([y_train, y_validation, y_test])
            chance_prec = chance_level_precision(y_all, is_detection)

        x_train_for_validation, x_validation_for_selection = normalize_train_eval(
            x_train, x_validation, args.normalize_data
        )
        validation_scores = []
        for c in args.inverse_regs:
            clf = fit_classifier(
                args.dataset_name, args.seed, c, x_train_for_validation, y_train
            )
            score = evaluate_classifier(
                args.dataset_name, clf, x_validation_for_selection, y_validation
            )
            validation_scores.append((c, score))
        best_c, _ = max(validation_scores, key=lambda item: item[1])

        x_final_train = np.concatenate([x_train, x_validation], axis=0)
        y_final_train = np.concatenate([y_train, y_validation], axis=0)
        x_final_train, x_test_for_evaluation = normalize_train_eval(
            x_final_train, x_test, args.normalize_data
        )

        set_seed(lr_seed)
        clf = fit_classifier(
            args.dataset_name, lr_seed, best_c, x_final_train, y_final_train
        )

        if is_detection:
            test_score = detection_probabilities(clf, x_test_for_evaluation)
            precision, recall, ap = micro_pr_curve_multilabel(y_test, test_score)
        else:
            test_score = clf.predict_proba(x_test_for_evaluation)
            classes = np.sort(clf.classes_)
            precision, recall, ap = micro_pr_curve(y_test, test_score, classes)

        for i in range(len(recall)):
            curve_records.append(
                {
                    "dataset_name": args.dataset_name,
                    "task": task,
                    "seed": args.seed,
                    "lr_seed": lr_seed,
                    "eval_split": "test",
                    "normalize_data": args.normalize_data,
                    "dolph2vec_model": dolph2vec_base,
                    "model": model_name,
                    "point_index": i,
                    "recall": float(recall[i]),
                    "precision": float(precision[i]),
                    "best_inverse_reg_C": float(best_c),
                }
            )

        ax.plot(
            recall,
            precision,
            label=f"{model_name} (AP={ap:.3f})",
            linewidth=2,
            color=color_for_model(model_name),
            zorder=3,
        )

    assert chance_prec is not None
    for i, r in enumerate((0.0, 1.0)):
        curve_records.append(
            {
                "dataset_name": args.dataset_name,
                "task": task,
                "seed": args.seed,
                "lr_seed": lr_seed,
                "eval_split": "test",
                "normalize_data": args.normalize_data,
                "dolph2vec_model": dolph2vec_base,
                "model": "Chance",
                "point_index": i,
                "recall": r,
                "precision": float(chance_prec),
                "best_inverse_reg_C": float("nan"),
            }
        )

    ax.axhline(
        chance_prec,
        color="#616161",
        linestyle="--",
        linewidth=1.5,
        label=f"Chance (prevalence={chance_prec:.3f})",
        zorder=1,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        frameon=False,
        borderaxespad=0,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / args.out_name
    csv_path = out_dir / f"{out_path.stem}_curves.csv"
    pd.DataFrame(curve_records).to_csv(csv_path, index=False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
