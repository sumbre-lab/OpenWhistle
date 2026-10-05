"""Hugging Face checkpoints evaluated for the OpenWhistle rebuttal."""

HF_COLLECTION_MODELS = (
    {
        "label": "AVES-Bio OpenWhistle 10%",
        "model_id": "dolphinteam/AVES-bio-OpenWhistle-10pct",
        "family": "AVES-Bio",
        "sample_rate": 44100,
    },
    {
        "label": "AVES-Bio OpenWhistle 50%",
        "model_id": "dolphinteam/AVES-bio-OpenWhistle-50pct",
        "family": "AVES-Bio",
        "sample_rate": 44100,
    },
    {
        "label": "AVES-Bio OpenWhistle 100%",
        "model_id": "dolphinteam/AVES-bio-OpenWhistle-100pct",
        "family": "AVES-Bio",
        "sample_rate": 44100,
    },
    {
        "label": "AVES-Bio OpenWhistle BTB3 10%",
        "model_id": "dolphinteam/AVES-bio-OpenWhistle-10pct-44kHz",
        "family": "AVES-Bio",
        "sample_rate": 16000,
        "feature_mode": "btb3_concat",
    },
    {
        "label": "AVES-Bio OpenWhistle BTB3 50%",
        "model_id": "dolphinteam/AVES-bio-OpenWhistle-50pct-44kHz",
        "family": "AVES-Bio",
        "sample_rate": 16000,
        "feature_mode": "btb3_concat",
    },
    {
        "label": "AVES-Bio OpenWhistle BTB3 100%",
        "model_id": "dolphinteam/AVES-bio-OpenWhistle-100pct-44kHz",
        "family": "AVES-Bio",
        "sample_rate": 16000,
        "feature_mode": "btb3_concat",
    },
    {
        "label": "Wav2Vec2.0 stride 960 10%",
        "model_id": "dolphinteam/wav2vec2-44k-stride960-10pct-40k",
        "family": "Wav2Vec2.0",
    },
    {
        "label": "Wav2Vec2.0 stride 960 50 %",
        "model_id": "dolphinteam/wav2vec2-44k-stride960-50pct-200k",
        "family": "Wav2Vec2.0",
    },
    {
        "label": "Wav2Vec2.0 stride 960 100%",
        "model_id": "dolphinteam/wav2vec2-44k-stride960-transformers",
        "family": "Wav2Vec2.0",
    },
    {
        "label": "Dolph2Vec",
        "model_id": "dolphinteam/OpenWhistle-Wav2Vec2.0",
        "family": "Wav2Vec2.0",
    }
)

# Native TorchAudio exports are frozen-probe backbones, not Transformers models.
# Keep them outside HF_COLLECTION_MODELS, which is also used by HF fine-tuning.
AVES_SCRATCH_MODELS = (
    {
        "label": "AVES OpenWhistle Stage 1 - 960 - 100%",
        "model_id": "dolphinteam/AVES-OpenWhistle-Stage-1-960-100pct",
        "family": "AVES-Bio",
        "backend": "aves_hf",
        "sample_rate": 44100,
        "revision": "4e2c4c4131c249ddef7f00bd3612d90c84a4029f",
        "model_filename": "models/openwhistle-aves-base-stage1-960.torchaudio.pt",
        "config_filename": "config/openwhistle-aves-base-stage1-960.torchaudio.model_config.json",
    },
    {
        "label": "AVES OpenWhistle Stage 2 - 320 - 100%",
        "model_id": "dolphinteam/AVES-OpenWhistle-Stage-2-320-100pct",
        "family": "AVES-Bio",
        "backend": "aves_hf",
        "sample_rate": 44100,
        "revision": "961e814923251f346f818440fb2e7883cd1331b1",
        "model_filename": "models/openwhistle-aves-base-stage2.torchaudio.pt",
        "config_filename": "config/openwhistle-aves-base-stage2.torchaudio.model_config.json",
    },
)

FROZEN_COLLECTION_MODELS = HF_COLLECTION_MODELS + AVES_SCRATCH_MODELS

# Exact checkpoint-to-row mapping for the rebuttal, not inferred from "100%".
_BY_ID = {model["model_id"]: model for model in FROZEN_COLLECTION_MODELS}
REBUTTAL_PRETRAINING_MODELS = tuple(
    {**_BY_ID[model_id], "label": label}
    for label, model_id in (
        ("AVES from scratch (Stage 2, stride 320)", "dolphinteam/AVES-OpenWhistle-Stage-2-320-100pct"),
        ("AVES continual (BTB3, 100%)", "dolphinteam/AVES-bio-OpenWhistle-100pct-44kHz"),
        ("Wav2Vec2 OpenWhistle 10%", "dolphinteam/wav2vec2-44k-stride960-10pct-40k"),
        ("Wav2Vec2 OpenWhistle 50%", "dolphinteam/wav2vec2-44k-stride960-50pct-200k"),
        ("Wav2Vec2 OpenWhistle 100%", "dolphinteam/OpenWhistle-Wav2Vec2.0"),
    )
)
