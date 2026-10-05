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
