"""Paths and default Hugging Face dataset ids for figure data."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"


def repo_root() -> Path:
    """Repository root (parent of ``datasets_figures``)."""
    return PACKAGE_ROOT.parent


PRETRAINING_SEGMENTS_HF_ID = "dolphinteam/OpenWhistle-Pretraining"
CLASSIFICATION_HF_ID = "dolphinteam/OpenWhistle-Classification-Finetuning"
