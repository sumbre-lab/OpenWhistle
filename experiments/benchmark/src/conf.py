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
    },
}


def _download_file(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    with urlopen(url) as response:
        destination.write_bytes(response.read())


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
        if not model_path.exists():
            _download_file(spec["model_url"], model_path)
        if not config_path.exists():
            _download_file(spec["config_url"], config_path)

    return str(model_path), str(config_path)


def get_aves_sample_rate(variant: str):
    if variant not in AVES_VARIANTS:
        raise ValueError(f"Unknown AVES variant: {variant}")
    return AVES_VARIANTS[variant]["sample_rate"]


dolph2vec_config_path = "/users/zfne/mustun/Documents/GitHub/Dolph2Vec/dolph2vec-base/preprocessor_config.json"

dolph2vec_base = "dolphinteam/model-dolph2vec_type-base_data-DolphinChat_version-v0"
