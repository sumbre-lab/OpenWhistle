"""Definitions and display order of the nine benchmark representations."""

BENCHMARK_MODELS = (
    dict(
        key="spectral_features",
        label="Spectral features",
        backend="spectral_features",
        group="Classical acoustic features",
        pretraining="--",
    ),
    dict(
        key="mfcc",
        label="MFCCs",
        backend="mfcc",
        group="Classical acoustic features",
        pretraining="--",
    ),
    dict(
        key="spectrogram",
        label="Mean spectrogram",
        backend="spectrogram",
        group="Classical acoustic features",
        pretraining="--",
    ),
    dict(
        key="aves_core",
        label="AVES-core",
        backend="aves_core",
        group="Off-the-shelf pretrained models",
        pretraining="General audio (AudioSet, FSD50K)",
    ),
    dict(
        key="biolingual",
        label="BioLingual",
        backend="biolingual",
        group="Off-the-shelf pretrained models",
        pretraining="Audio-text (AnimalSpeak)",
    ),
    dict(
        key="aves_bio",
        label="AVES-bio",
        backend="aves_bio",
        group="Off-the-shelf pretrained models",
        pretraining="Animal vocalizations (AudioSet, VGGSound)",
    ),
    dict(
        key="aves_scratch",
        label="AVES-bio (from scratch)",
        latex_label=r"AVES-bio$^{\ddagger}$",
        backend="aves_hf",
        group="Pretrained on OpenWhistle (ours)",
        pretraining="OpenWhistle (from scratch)",
        model_id="dolphinteam/AVES-OpenWhistle-Stage-2-320-100pct",
    ),
    dict(
        key="wav2vec2",
        label="Wav2Vec2.0",
        backend="hf",
        group="Pretrained on OpenWhistle (ours)",
        pretraining="OpenWhistle (from scratch)",
        model_id="dolphinteam/OpenWhistle-Wav2Vec2.0",
        sample_rate=44100,
    ),
    dict(
        key="aves_continual",
        label="AVES-bio (continual)",
        latex_label=r"AVES-bio$^{\dagger}$",
        backend="hf",
        group="Continual pretraining on OpenWhistle (ours)",
        pretraining="AVES-bio checkpoint to OpenWhistle",
        model_id="dolphinteam/AVES-bio-OpenWhistle-100pct-44kHz",
        feature_mode="btb3_concat",
        sample_rate=16000,
    ),
)


def result_model_id(model):
    return model.get("model_id", model["backend"])
