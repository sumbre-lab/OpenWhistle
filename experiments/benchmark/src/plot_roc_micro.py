#!/usr/bin/env python3
"""Compare the seven baseline models' micro-averaged detection ROC curves.

Reads the held-out test predictions exported by roc_glotin.py; no embedding
extraction or classifier fitting is repeated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc


ROOT = Path(__file__).resolve().parent.parent
# Table 2 order in https://arxiv.org/html/2609.34839v1
MODELS = (
    ("spectral_features", "Spectral features", "#6a1b9a"),
    ("mfcc", "MFCCs", "#1565c0"),
    ("spectrogram", "Mean spectrogram", "#ef6c00"),
    ("aves_core", "AVES-core", "#2e7d32"),
    ("biolingual", "BioLingual", "#00838f"),
    ("aves_bio", "AVES-bio", "#f9a825"),
    ("dolph2vec", "Wav2Vec2.0 (OpenWhistle)", "#c62828"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roc-dir", type=Path,
                        default=ROOT / "results/roc_curves")
    parser.add_argument("--out-name", default="roc_micro_detection_models")
    args = parser.parse_args()

    curves = []
    dataset_id = None
    n_examples = None
    for model_id, display_name, color in MODELS:
        model_dir = args.roc_dir / model_id
        with (model_dir / "roc_summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if (summary["model_id"] != model_id
                or summary["dataset_config"] != "default"
                or summary["split"] != "test"
                or summary["task"] != "seven-label whistle-type detection"):
            raise ValueError(f"Incompatible ROC summary for {model_id}")
        if dataset_id is None:
            dataset_id = summary["dataset_id"]
            n_examples = summary["n_examples"]
        elif (summary["dataset_id"] != dataset_id
              or summary["n_examples"] != n_examples):
            raise ValueError(f"Dataset/split mismatch for {model_id}")

        with (model_dir / "roc_points.csv").open(newline="", encoding="utf-8") as handle:
            micro = [row for row in csv.DictReader(handle) if row["label"] == "micro"]
        fpr = np.array([float(row["false_positive_rate"]) for row in micro])
        tpr = np.array([float(row["true_positive_rate"]) for row in micro])
        if (len(fpr) < 2 or not np.isfinite(fpr).all() or not np.isfinite(tpr).all()
                or np.any(np.diff(fpr) < 0) or np.any(np.diff(tpr) < 0)
                or (fpr[0], tpr[0]) != (0.0, 0.0)
                or (fpr[-1], tpr[-1]) != (1.0, 1.0)):
            raise ValueError(f"Invalid micro ROC points for {model_id}")
        roc_auc = float(auc(fpr, tpr))
        if not np.isclose(roc_auc, summary["micro_roc_auc"], atol=1e-8):
            raise ValueError(f"Micro ROC-AUC mismatch for {model_id}")
        curves.append((display_name, color, fpr, tpr, roc_auc))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot((0, 1), (0, 1), "--", color="0.55", linewidth=1.2, label="Chance")
    for name, color, fpr, tpr, roc_auc in curves:
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate",
           title=f"Micro-averaged ROC — detection test (n={n_examples['test']})")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    output = args.roc_dir / args.out_name
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output.with_suffix('.png')} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
