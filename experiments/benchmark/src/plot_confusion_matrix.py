#!/usr/bin/env python3
"""
Plot a confusion matrix using the same evaluation setup as train_lr_splits.py:

- Same dataset splits, backbone embeddings, and optional feature standardization.
- Inverse regularization C chosen on the validation split (grid from --inverse_reg /
  --inverse_regs).
- Final logistic regression fit on train+validation, evaluated on the test split.

For ``classification``, a single multiclass confusion matrix is shown.
For ``detection`` (multilabel), sklearn's per-label binary confusion matrices are
plotted in a grid (decision rule matches ``MultiOutputClassifier.predict``, i.e.
default 0.5 probability threshold per label).

Raw counts are also written to a CSV (same basename as the figure by default):
classification uses rows/columns of class names; detection uses one row per label
with ``true_i_pred_j`` counts for each 2×2 block.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from conf import (
    dolph2vec_base,
    dolph2vec_config_path,
    get_aves_paths,
    get_aves_sample_rate,
)
from hf_datasets import load_classification_splits, load_detection_splits
from models import MFCC, Aves, BioLingual, Dolph2Vec, SpectralFeatures, Spectrogram
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    multilabel_confusion_matrix,
)
from train_lr_splits import (
    embed_split,
    evaluate_classifier,
    fit_classifier,
    normalize_train_eval,
    set_seed,
)

LR_SEED_HELP = (
    "Random state for the final logistic regression fit used for the plot "
    "(same role as the first seed in train_lr_splits --num_seeds)."
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Embedding / numpy seed.")
    parser.add_argument(
        "--lr_seed",
        default=None,
        type=int,
        help=f"{LR_SEED_HELP} Defaults to --seed.",
    )

    parser.add_argument("--inverse_reg", default=1.0, type=float)
    parser.add_argument(
        "--inverse_regs",
        nargs="+",
        type=float,
        default=None,
        help="Candidate C values; best C is selected on the validation split.",
    )

    parser.add_argument(
        "--no_normalize_data",
        dest="normalize_data",
        action="store_false",
        default=True,
        help="Disable feature standardization before logistic regression.",
    )

    parser.add_argument(
        "--dataset_name",
        choices=["classification", "detection"],
        required=True,
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
        "--output",
        type=Path,
        default=None,
        help="Output image path (.png / .pdf). Default: benchmark/results/confusion_<dataset>_<model>.png",
    )
    parser.add_argument(
        "--normalize",
        choices=["true", "pred", "all", "none"],
        default="none",
        help=(
            "Normalization for the classification matrix only (sklearn "
            "ConfusionMatrixDisplay). Detection uses raw counts per label."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path for raw count matrix CSV. Default: same basename as the figure with .csv.",
    )
    return parser.parse_args()


def _build_model(args):
    name2model = {
        "aves_core": Aves,
        "aves_bio": Aves,
        "biolingual": BioLingual,
        "dolph2vec": Dolph2Vec,
        "mfcc": MFCC,
        "spectrogram": Spectrogram,
        "spectral_features": SpectralFeatures,
    }

    actual_sample_rate = args.target_sample_rate
    if args.model == "aves_bio":
        aves_model_path, aves_config_path = get_aves_paths("bio")
        actual_sample_rate = get_aves_sample_rate("bio")
    elif args.model == "aves_core":
        aves_model_path, aves_config_path = get_aves_paths("core")
        actual_sample_rate = get_aves_sample_rate("core")
    else:
        aves_model_path, aves_config_path = "", ""

    model_args = dict(
        sample_rate=actual_sample_rate,
        dolph2vec_config_path=dolph2vec_config_path,
        dolph2vec_model_path=dolph2vec_base,
        aves_model_path=aves_model_path,
        aves_config_path=aves_config_path,
    )
    return name2model[args.model](**model_args), actual_sample_rate


def write_confusion_matrix_csv(
    dataset_name: str,
    csv_path: Path,
    label_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Write sklearn-style count matrices (not the optional plot normalization)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(label_names)))
    if dataset_name == "classification":
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        df = pd.DataFrame(cm, index=label_names, columns=label_names)
        df.index.name = "true_label"
        df.to_csv(csv_path)
    else:
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        rows = []
        for i, name in enumerate(label_names):
            mat = mcm[i]
            rows.append(
                {
                    "label": name,
                    "true_0_pred_0": int(mat[0, 0]),
                    "true_0_pred_1": int(mat[0, 1]),
                    "true_1_pred_0": int(mat[1, 0]),
                    "true_1_pred_1": int(mat[1, 1]),
                }
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    args = get_args()
    lr_seed = args.lr_seed if args.lr_seed is not None else args.seed
    set_seed(args.seed)

    inverse_regs = (
        args.inverse_regs if args.inverse_regs is not None else [args.inverse_reg]
    )

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        out_path = results_dir / f"confusion_{args.dataset_name}_{args.model}.png"
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv) if args.csv is not None else out_path.with_suffix(".csv")

    model, _ = _build_model(args)

    if args.dataset_name == "detection":
        dataset, label_names = load_detection_splits()
    else:
        dataset, label_feature = load_classification_splits()
        label_names = list(label_feature.names)

    x_train, y_train = embed_split(dataset["train"], "train", args.dataset_name, model)
    x_validation, y_validation = embed_split(
        dataset["validation"], "validation", args.dataset_name, model
    )
    x_test, y_test = embed_split(dataset["test"], "test", args.dataset_name, model)

    x_train_for_validation, x_validation_for_selection = normalize_train_eval(
        x_train, x_validation, args.normalize_data
    )
    validation_scores = []
    for c in inverse_regs:
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
    y_pred = clf.predict(x_test_for_evaluation)

    normalize = None if args.normalize == "none" else args.normalize

    if args.dataset_name == "classification":
        figsize = (max(8, len(label_names)), max(6, len(label_names) * 0.6))
        fig, ax = plt.subplots(figsize=figsize)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=label_names,
            normalize=normalize,
            cmap="Blues",
            colorbar=False,
            include_values=True,
            values_format=".2f" if normalize is not None else "d",
            ax=ax,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
    else:
        # Multilabel: one 2x2 matrix per output (same as sklearn multilabel_confusion_matrix).
        mcm = multilabel_confusion_matrix(y_test, y_pred)
        n_labels = mcm.shape[0]
        ncols = min(4, n_labels)
        nrows = int(np.ceil(n_labels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.5 * ncols, 3.2 * nrows),
            squeeze=False,
        )
        for i in range(nrows * ncols):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            if i < n_labels:
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=mcm[i], display_labels=[0, 1]
                )
                disp.plot(ax=ax, include_values=True, cmap="Blues", colorbar=False, values_format="d")
                ax.set_title(label_names[i], fontsize=10)
            else:
                ax.axis("off")
        fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_confusion_matrix_csv(
        args.dataset_name, csv_path, label_names, y_test, y_pred
    )
    print(f"Saved confusion matrix figure to {out_path.resolve()}")
    print(f"Saved confusion matrix counts to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
