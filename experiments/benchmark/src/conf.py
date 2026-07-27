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


dolph2vec_base = "OpenWhistleNeurIPS26/OpenWhistle-Wav2Vec2.0"
dolph2vec_config_path = dolph2vec_base

# =============================================================================
# WARNING: NOT ANONYMIZED. DO NOT MERGE OR PUSH THIS BLOCK TO `main`.
#
# The repo IDs below live under the "dolphinteam" HF org, which reveals
# author identity ahead of NeurIPS 2026 double-blind review. They are for
# local pretraining-size ablations on feature branches only. Before opening
# a PR into `main`, remove this block or repoint it at anonymized
# `OpenWhistleNeurIPS26` org copies.
# =============================================================================

# AVES-bio checkpoints exported as HF `HubertModel` (see
# models.AvesBioPretrain). These were pretrained at 16kHz (the "44kHz" name
# on the continuous-pretraining ones refers to a BTB3 band-shift transform
# applied to the pretraining audio, not the model's native input rate).
# We deliberately evaluate at AVES_BIO_PRETRAIN_SAMPLE_RATE (44.1kHz) below
# to match the rest of the benchmark's models, not the 16kHz training rate.
AVES_BIO_PRETRAIN_VARIANTS = {
    # different pretraining sizes
    "10pct": "dolphinteam/AVES-bio-OpenWhistle-10pct",
    "50pct": "dolphinteam/AVES-bio-OpenWhistle-50pct",
    "100pct": "dolphinteam/AVES-bio-OpenWhistle-100pct",
    # continuous-learning different pretraining sizes
    "10pct_44khz": "dolphinteam/AVES-bio-OpenWhistle-10pct-44kHz",
    "50pct_44khz": "dolphinteam/AVES-bio-OpenWhistle-50pct-44kHz",
    "100pct_44khz": "dolphinteam/AVES-bio-OpenWhistle-100pct-44kHz",
}
AVES_BIO_PRETRAIN_SAMPLE_RATE = 44100

# Wav2Vec2.0 checkpoints with different pretraining sizes, same HF
# `Wav2Vec2Model` loading path as `dolph2vec_base` above.
WAV2VEC2_PRETRAIN_VARIANTS = {
    "10pct": "dolphinteam/wav2vec2-44k-stride960-10pct-40k",
    "50pct": "dolphinteam/wav2vec2-44k-stride960-50pct-200k",
    "transformers": "dolphinteam/wav2vec2-44k-stride960-transformers",
}
WAV2VEC2_PRETRAIN_SAMPLE_RATE = 44100