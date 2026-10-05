import json
import os
from pathlib import Path
from urllib.request import urlopen


AVES_CACHE_DIR = Path(
    os.environ.get("OPENWHISTLE_AVES_CACHE_DIR", Path.home() / ".cache/openwhistle/aves")
).expanduser()

AVES_VARIANTS = {
    "core": {
        "model_filename": "aves-base-core.torchaudio.pt",
        "config_filename": "aves-base-core.torchaudio.model_config.json",
        "model_url": "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-core.torchaudio.pt",
        "config_url": "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-core.torchaudio.model_config.json",
        "sample_rate": 44100,
    },
    "bio": {
        "model_filename": "aves-base-bio.torchaudio.pt",
        "config_filename": "aves-base-bio.torchaudio.model_config.json",
        "model_url": "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt",
        "config_url": "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.model_config.json",
        "sample_rate": 44100,
    }
}


def _looks_like_html_file(path: Path) -> bool:
    with path.open("rb") as f:
        prefix = f.read(256).lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _is_valid_json_file(path: Path) -> bool:
    try:
        with path.open() as f:
            json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return True


def _is_valid_cached_file(path: Path, *, is_json: bool = False) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if _looks_like_html_file(path):
        return False
    if is_json:
        return _is_valid_json_file(path)
    return True


def _normalize_aves_config_file(path: Path):
    with path.open() as f:
        config = json.load(f)

    # The installed `aves` package passes this dict directly to
    # torchaudio.models.wav2vec2_model, which does not accept metadata keys.
    unsupported_keys = ("sample_rate",)
    normalized = {
        key: value for key, value in config.items() if key not in unsupported_keys
    }
    if normalized != config:
        path.write_text(json.dumps(normalized, indent=2) + "\n")


def _download_file(url: str, destination: Path, *, is_json: bool = False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    with urlopen(url) as response:
        destination.write_bytes(response.read())
    if not _is_valid_cached_file(destination, is_json=is_json):
        raise ValueError(f"Downloaded file is invalid: {destination}")


def _env_override_paths(variant: str):
    prefix = f"OPENWHISTLE_AVES_{variant.upper()}"
    model_path = os.environ.get(f"{prefix}_MODEL")
    config_path = os.environ.get(f"{prefix}_CONFIG")
    if model_path and config_path:
        return Path(model_path).expanduser(), Path(config_path).expanduser()
    return None, None


def get_aves_paths(variant: str, download: bool = True):
    if variant not in AVES_VARIANTS:
        raise ValueError(f"Unknown AVES variant: {variant}")

    env_model, env_config = _env_override_paths(variant)
    if env_model and env_config:
        return str(env_model), str(env_config)

    spec = AVES_VARIANTS[variant]
    model_path = AVES_CACHE_DIR / spec["model_filename"]
    config_path = AVES_CACHE_DIR / spec["config_filename"]

    if download:
        if not _is_valid_cached_file(model_path):
            _download_file(spec["model_url"], model_path)
        if not _is_valid_cached_file(config_path, is_json=True):
            _download_file(spec["config_url"], config_path, is_json=True)
        _normalize_aves_config_file(config_path)

    return str(model_path), str(config_path)


def get_aves_sample_rate(variant: str):
    if variant not in AVES_VARIANTS:
        raise ValueError(f"Unknown AVES variant: {variant}")
    return AVES_VARIANTS[variant]["sample_rate"]


dolph2vec_base = "dolphinteam/OpenWhistle-Wav2Vec2.0"
dolph2vec_config_path = dolph2vec_base
