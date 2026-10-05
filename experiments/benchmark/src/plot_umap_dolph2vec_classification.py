#!/usr/bin/env python3
"""
UMAP scatter plot of Dolph2Vec embeddings for the OpenWhistle classification
dataset loaded from Hugging Face (default config: ``all``). By default, rows
from ``train``, ``validation``, and ``test`` are concatenated; override with
``--splits`` if a config exposes different split names.

Use your ``dolph2vec`` conda environment (or any env with PyTorch, transformers,
datasets, etc.). Add UMAP if needed: ``pip install umap-learn``.

From ``experiments/benchmark``::

  conda activate dolph2vec
  export PYTHONPATH="$(pwd)/src"
  python src/plot_umap_dolph2vec_classification.py --out_path results/umap_dolph2vec_all.png
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import torch
from hf_datasets import load_classification_examples
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from conf import dolph2vec_base, dolph2vec_config_path
from models import Dolph2Vec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config_name",
        type=str,
        default="all",
        help="HF dataset config (passed as ``name=`` to ``load_dataset``).",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,validation,test",
        help="Comma-separated HF split names to merge, in order.",
    )
    p.add_argument(
        "--out_path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results"
        / "umap_dolph2vec_classification.png",
        help="Output PNG path.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, randomly subsample this many rows (stratified by label when possible).",
    )
    p.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply StandardScaler to embeddings before UMAP.",
    )
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="cosine")
    return p.parse_args()


def _maybe_subsample(
    x: np.ndarray,
    y: np.ndarray,
    label_names: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    if max_samples is None or n <= max_samples:
        return x, y, label_names
    rng = np.random.default_rng(seed)
    # Stratified subsample per class
    unique = np.unique(y)
    indices: list[int] = []
    per_class = max(1, max_samples // len(unique))
    for c in unique:
        pool = np.where(y == c)[0]
        take = min(len(pool), per_class)
        chosen = rng.choice(pool, size=take, replace=False)
        indices.extend(chosen.tolist())
    indices = np.array(indices, dtype=np.int64)
    if len(indices) > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)
    return x[indices], y[indices], label_names[indices]


def main():
    args = parse_args()
    set_seed(args.seed)

    try:
        import umap
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: install with ``pip install umap-learn``"
        ) from e

    split_tuple = tuple(
        s.strip() for s in args.splits.split(",") if s.strip()
    )
    if not split_tuple:
        raise SystemExit("--splits must list at least one split name.")
    dataset, label_feature = load_classification_examples(
        config_name=args.config_name,
        splits=split_tuple,
    )
    model = Dolph2Vec(
        dolph2vec_model_path=dolph2vec_base,
        dolph2vec_config_path=dolph2vec_config_path,
        sample_rate=44100,
    )

    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    for row in tqdm(dataset, desc="Dolph2Vec embed", total=len(dataset)):
        audio = row["audio"]
        try:
            emb = model(audio)
            vec = emb.detach().cpu().numpy()
            if vec.ndim > 1:
                vec = vec.reshape(-1)
            embeddings.append(vec.astype(np.float32, copy=False))
            labels.append(int(row["label"]))
        except Exception as ex:
            path = audio.get("path") if isinstance(audio, dict) else None
            print(f"skip {path or '<audio>'}: {ex}")

    if not embeddings:
        raise SystemExit("No embeddings produced.")

    x = np.stack(embeddings, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    names = np.array([label_feature.int2str(int(i)) for i in y])

    x, y, names = _maybe_subsample(x, y, names, args.max_samples, args.seed)

    if args.standardize:
        x = StandardScaler().fit_transform(x)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        verbose=True,
    )
    z = reducer.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))
    classes = np.unique(names)
    tab20 = colormaps["tab20"]
    for i, cname in enumerate(classes):
        m = names == cname
        ax.scatter(
            z[m, 0],
            z[m, 1],
            s=8,
            alpha=0.65,
            label=cname,
            color=tab20((i % 20) / 19.0),
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(
        f"Dolph2Vec + UMAP — {args.config_name} [{','.join(split_tuple)}] "
        f"({len(y)} whistles)\n{dolph2vec_base}"
    )
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        markerscale=2,
        framealpha=0.9,
    )
    fig.tight_layout()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_path.resolve()}")


if __name__ == "__main__":
    main()
