"""Scientific experiment settings, output paths and local cache identity."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from aves_models import AVES_SCRATCH_MODELS
from benchmark_revisions import DATASET_REVISIONS, MODEL_REVISIONS
from benchmark_models import BENCHMARK_MODELS

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
PUBLISHED_RESULTS_DIR = BENCHMARK_DIR / "results"
RUNS_DIR = BENCHMARK_DIR.parents[1] / "runs" / "benchmark"
CLASSIFICATION_CONFIGS = ("all", "balanced")
EVALUATIONS = tuple(("classification", config) for config in CLASSIFICATION_CONFIGS) + (
    ("detection", "default"),
)
EVALUATION_SETTINGS = dict(
    batch_size=1,
    precision="float32",
    seed=42,
    num_seeds=1,
    num_bootstrap=1000,
    strict_embedding_batches=True,
    batch_native_models=True,
    inverse_regs=[0.1, 1, 10],
)

# Fixed official split counts, used only to calculate the table's chance baselines.
PUBLISHED_SPLIT_STATISTICS = {
    "classification": {
        "all": {
            "train_class_counts": [31, 19, 460, 235, 2054, 365, 1567, 622, 57, 438],
            "test_class_counts": [7, 4, 99, 50, 440, 78, 336, 133, 12, 94],
        },
        "balanced": {
            "train_class_counts": [350, 350, 350, 350, 350, 350],
            "test_class_counts": [75, 75, 75, 75, 75, 75],
        },
    },
    "detection": {
        "default": {
            "test_examples": 840,
            "positives_per_label": [60, 60, 60, 60, 60, 60, 60],
        }
    },
}


def embedding_cache_fingerprint():
    """Invalidate local embeddings automatically when extraction inputs change."""
    digest = hashlib.sha256()
    for name in (
        "benchmark_config.py",
        "benchmark_revisions.py",
        "aves_models.py",
        "aves_artifacts.py",
        "hf_datasets.py",
        "models.py",
        "train_lr_splits.py",
    ):
        digest.update((BENCHMARK_DIR / "src" / name).read_bytes())
    for name in (
        "numpy",
        "scipy",
        "torch",
        "torchaudio",
        "transformers",
        "datasets",
        "esp-aves",
        "librosa",
        "numba",
        "soundfile",
    ):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "unavailable"
        digest.update(f"{name}={version}".encode())
    return digest.hexdigest()[:16]


CACHE_FINGERPRINT = embedding_cache_fingerprint()


def read_run_config(run_dir):
    """Load settings from a local run or use the published experiment definition."""
    run_dir = Path(run_dir).resolve()
    local = run_dir / "run_config.json"
    if local.is_file():
        return json.loads(local.read_text())
    if run_dir == PUBLISHED_RESULTS_DIR:
        revisions = {
            **MODEL_REVISIONS,
            **{model["model_id"]: model["revision"] for model in AVES_SCRATCH_MODELS},
        }
        return dict(
            **EVALUATION_SETTINGS,
            models=[dict(model) for model in BENCHMARK_MODELS],
            dataset_revisions=dict(DATASET_REVISIONS),
            model_revisions=revisions,
            split_statistics=PUBLISHED_SPLIT_STATISTICS,
        )
    raise FileNotFoundError(f"No run_config.json found in local run: {run_dir}")
