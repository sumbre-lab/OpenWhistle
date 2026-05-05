from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PRETRAINING_ROOT = PACKAGE_ROOT.parent
ARTIFACTS_ROOT = Path(
    os.environ.get(
        "OPENWHISTLE_PRETRAINING_ARTIFACTS_DIR",
        PRETRAINING_ROOT / "artifacts",
    )
).resolve()


def artifact_dir(*parts: str, create: bool = False) -> Path:
    path = ARTIFACTS_ROOT.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def submitit_parameters(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value not in (None, "")}
