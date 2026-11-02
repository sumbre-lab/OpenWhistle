"""Fetch and verify the official native AVES artifacts in the local cache."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from aves_models import AVES_VARIANTS

AVES_CACHE_DIR = Path(
    os.environ.get(
        "OPENWHISTLE_AVES_CACHE_DIR", Path.home() / ".cache/openwhistle/aves"
    )
).expanduser()


def _matches_sha256(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == expected


def _download_file(url, destination, expected_sha256):
    """Publish a cache entry only after its complete download passes verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        print(f"Downloading {url} -> {destination}")
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if not _matches_sha256(temporary, expected_sha256):
            raise ValueError(f"AVES artifact hash differs: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_config(path):
    """Keep the verified source intact when TorchAudio metadata must be removed."""
    config = json.loads(path.read_text())
    if "sample_rate" not in config:
        return path
    config.pop("sample_rate")
    normalized = path.with_name(path.stem + ".normalized.json")
    normalized.write_text(json.dumps(config, indent=2) + "\n")
    return normalized


def get_aves_paths(variant: str, download: bool = True):
    if variant not in AVES_VARIANTS:
        raise ValueError(f"Unknown AVES variant: {variant}")
    spec = AVES_VARIANTS[variant]
    paths = {
        kind: AVES_CACHE_DIR / spec[kind + "_filename"] for kind in ("model", "config")
    }
    if download:
        for kind, path in paths.items():
            expected = spec[kind + "_sha256"]
            if not _matches_sha256(path, expected):
                _download_file(spec[kind + "_url"], path, expected)
        paths["config"] = _normalized_config(paths["config"])
    return str(paths["model"]), str(paths["config"])


def get_aves_sample_rate(variant: str):
    if variant not in AVES_VARIANTS:
        raise ValueError(f"Unknown AVES variant: {variant}")
    return AVES_VARIANTS[variant]["sample_rate"]
