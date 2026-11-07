"""Pinned native AVES exports and file-integrity specifications."""

AVES_SCRATCH_MODELS = (
    {
        "label": "AVES OpenWhistle Stage 2 - 320 - 100%",
        "model_id": "dolphinteam/AVES-OpenWhistle-Stage-2-320-100pct",
        "family": "AVES-Bio",
        "backend": "aves_hf",
        "sample_rate": 44100,
        "revision": "961e814923251f346f818440fb2e7883cd1331b1",
        "model_filename": "models/openwhistle-aves-base-stage2.torchaudio.pt",
        "config_filename": "config/openwhistle-aves-base-stage2.torchaudio.model_config.json",
        "weights_sha256": "aacb6b474be576c2340879a43e841a83574cd6b43f1d610fc56fcf877e74fe15",
        "config_sha256": "0cbf6fe27794cc5dcac298faef59f2c76c317602cabb457a65df2400a69c778f",
    },
)


AVES_VARIANTS = {
    "core": {
        "model_filename": "aves-base-core.torchaudio.pt",
        "config_filename": "aves-base-core.torchaudio.model_config.json",
        "model_url": "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-core.torchaudio.pt",
        "config_url": "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-core.torchaudio.model_config.json",
        "sample_rate": 44100,
        "model_sha256": "ccacdb52fadb338772698f6e1301092b18538289dc520e8d775026971fbf8ee3",
        "config_sha256": "5c6ad4a9469bfa0f4ca705eed0cb814241048546833cbd291a32a6082ed96dc9",
    },
    "bio": {
        "model_filename": "aves-base-bio.torchaudio.pt",
        "config_filename": "aves-base-bio.torchaudio.model_config.json",
        "model_url": "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt",
        "config_url": "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.model_config.json",
        "sample_rate": 44100,
        "model_sha256": "7a7dfaff2ea0b617cae1d82d7831e766be2a9ac00e37962a26f0a1b285be2530",
        "config_sha256": "5c6ad4a9469bfa0f4ca705eed0cb814241048546833cbd291a32a6082ed96dc9",
    },
}
