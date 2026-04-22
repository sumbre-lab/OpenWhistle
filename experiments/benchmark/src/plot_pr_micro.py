#!/usr/bin/env python3
"""
Build a single figure: micro-averaged precision–recall (OvR, pooled) per model.
For classification, C is chosen by highest mean k-fold accuracy over --inverse_regs.
For multilabel detection, C is chosen by highest mean k-fold micro AP.
Curves use out-of-fold probability estimates with that C.
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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


def set_seed(seed: int = 42):
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_audio_model(model_name: str, target_sample_rate: int):
    name2model = {
        "aves_core": Aves,
        "aves_bio": Aves,
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


def load_embeddings(
    model_name: str,
    dataset_name: str,
    target_sample_rate: int,
    seed: int,
):
    set_seed(seed)
    model = get_audio_model(model_name, target_sample_rate)

    dataset_name2path = {
        "classification": "data/classification/balanced/all.csv",
        "dolphin_reef_balanced": "data/dolphin_reef/balanced/all.csv",
        "dolphin_reef_unbalanced": "data/dolphin_reef/unbalanced/all.csv",
        "detection": "data/detection/all.csv",
    }
    data_path = dataset_name2path[dataset_name]
    df = pd.read_csv(data_path)
    detection_label_cols = None
    if dataset_name == "detection":
        _detection_meta = {"path", "name", "original_path"}
        detection_label_cols = [c for c in df.columns if c not in _detection_meta]

    df["path"] = df["path"].str.replace(
        "/lustre/fsn1/projects/rech/vzf/uqe97pu/raw_data/all_categories/",
        "/media/DOLPHIN/HF_DolphinReef-labeled/",
        regex=False,
    )

    embeddings = []
    labels = []
    for _, row in tqdm(df.iterrows(), desc=f"embed {model_name}", total=len(df)):
        path = row["path"]
        if dataset_name == "detection":
            label = row[detection_label_cols].values.astype(int)
        else:
            label = row["label"]
        try:
            embedding = model(path)
            embeddings.append(embedding.cpu())
            labels.append(label)
        except Exception as e:
            print(f"error processing {path}: {e}")

    x = np.array(embeddings)
    y = np.asarray(labels)
    if dataset_name == "detection":
        if y.ndim != 2:
            raise ValueError(
                f"detection expects multilabel y with shape (n, n_labels); got {y.shape}"
            )
    elif y.ndim != 1:
        raise ValueError(
            "plot_pr_micro expects single-label classification (1D labels) for this dataset. "
            f"Got label array shape {y.shape} for dataset {dataset_name}."
        )
    return x, y


LR_MAX_ITER = 10000


def mean_cv_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    kf: StratifiedKFold,
    c: float,
    seed: int,
) -> float:
    accs = []
    for train_idx, val_idx in kf.split(x, y):
        clf = LogisticRegression(max_iter=LR_MAX_ITER, random_state=seed, C=c)
        clf.fit(x[train_idx], y[train_idx])
        accs.append(accuracy_score(y[val_idx], clf.predict(x[val_idx])))
    return float(np.mean(accs))


def oof_predict_proba(
    x: np.ndarray,
    y: np.ndarray,
    kf: StratifiedKFold,
    c: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probability matrix aligned to sorted global class list."""
    classes_sorted = np.sort(np.unique(y))
    n_classes = len(classes_sorted)
    n_samples = len(y)
    oof = np.zeros((n_samples, n_classes))
    for train_idx, val_idx in kf.split(x, y):
        clf = LogisticRegression(max_iter=LR_MAX_ITER, random_state=seed, C=c)
        clf.fit(x[train_idx], y[train_idx])
        proba = clf.predict_proba(x[val_idx])
        for j, c_lab in enumerate(clf.classes_):
            gi = int(np.searchsorted(classes_sorted, c_lab))
            oof[val_idx, gi] = proba[:, j]
    return oof, classes_sorted


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


def mean_cv_micro_ap_detection(
    x: np.ndarray,
    y: np.ndarray,
    kf: MultilabelStratifiedKFold,
    c: float,
    seed: int,
) -> float:
    """Mean validation micro AP across folds (multilabel detection)."""
    aps = []
    for train_idx, val_idx in kf.split(x, y):
        base_clf = LogisticRegression(max_iter=LR_MAX_ITER, random_state=seed, C=c)
        clf = MultiOutputClassifier(base_clf)
        clf.fit(x[train_idx], y[train_idx])
        y_score = clf.predict_proba(x[val_idx])
        y_score = np.array([np.asarray(p)[:, 1] for p in y_score]).T
        aps.append(
            average_precision_score(y[val_idx], y_score, average="micro")
        )
    return float(np.mean(aps))


def oof_predict_proba_multilabel(
    x: np.ndarray,
    y: np.ndarray,
    kf: MultilabelStratifiedKFold,
    c: float,
    seed: int,
) -> np.ndarray:
    n_samples, n_labels = y.shape
    oof = np.zeros((n_samples, n_labels))
    for train_idx, val_idx in kf.split(x, y):
        base_clf = LogisticRegression(max_iter=LR_MAX_ITER, random_state=seed, C=c)
        clf = MultiOutputClassifier(base_clf)
        clf.fit(x[train_idx], y[train_idx])
        y_score = clf.predict_proba(x[val_idx])
        y_score = np.array([np.asarray(p)[:, 1] for p in y_score]).T
        oof[val_idx] = y_score
    return oof


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument(
        "--no_normalize",
        dest="normalize_data",
        action="store_false",
        help="Disable StandardScaler on embeddings before logistic regression (default: scale).",
    )
    p.set_defaults(normalize_data=True)
    p.add_argument(
        "--dataset_name",
        default="dolphin_reef_balanced",
        choices=[
            "classification",
            "dolphin_reef_balanced",
            "dolphin_reef_unbalanced",
            "detection",
        ],
        help="detection uses multilabel micro PR; others use single-label multiclass micro PR.",
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
        help="Candidate C values; best is chosen by mean CV accuracy per model.",
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
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    benchmark_root = script_dir.parent
    out_dir = args.out_dir or (benchmark_root / "results" / "pr_curves")
    out_dir.mkdir(parents=True, exist_ok=True)

    is_detection = args.dataset_name == "detection"
    if is_detection:
        kf = MultilabelStratifiedKFold(
            n_splits=args.kfold, shuffle=True, random_state=args.seed
        )
    else:
        kf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    task = "multilabel_detection" if is_detection else "multiclass_micro_ovr"

    fig, ax = plt.subplots(figsize=(8, 6.5))

    curve_records: list[dict] = []
    chance_prec: float | None = None
    for model_name in args.models:
        x, y = load_embeddings(
            model_name,
            args.dataset_name,
            args.target_sample_rate,
            args.seed,
        )
        if chance_prec is None:
            chance_prec = chance_level_precision(y, is_detection)
        if args.normalize_data:
            x = StandardScaler().fit_transform(x)

        best_c = None
        best_metric = -1.0
        for c in args.inverse_regs:
            if is_detection:
                metric = mean_cv_micro_ap_detection(x, y, kf, c, args.seed)
            else:
                metric = mean_cv_accuracy(x, y, kf, c, args.seed)
            if metric > best_metric:
                best_metric = metric
                best_c = c

        assert best_c is not None
        if is_detection:
            oof_score = oof_predict_proba_multilabel(x, y, kf, best_c, args.seed)
            precision, recall, ap = micro_pr_curve_multilabel(y, oof_score)
        else:
            oof_score, classes = oof_predict_proba(x, y, kf, best_c, args.seed)
            precision, recall, ap = micro_pr_curve(y, oof_score, classes)

        for i in range(len(recall)):
            curve_records.append(
                {
                    "dataset_name": args.dataset_name,
                    "task": task,
                    "seed": args.seed,
                    "kfold": args.kfold,
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
                "kfold": args.kfold,
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
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / args.out_name
    csv_path = out_dir / f"{out_path.stem}_curves.csv"
    pd.DataFrame(curve_records).to_csv(csv_path, index=False)

    fig.savefig(out_path, dpi=150)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
