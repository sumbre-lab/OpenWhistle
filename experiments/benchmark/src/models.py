import librosa
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from transformers import (
    ClapModel,
    ClapProcessor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)



def infer_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("| Using CUDA for computation.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("| Using Apple MPS for computation.")
    else:
        device = torch.device("cpu")
        print("| Using CPU for computation.")
    return device


def get_waveform(filename, target_sample_rate):
    try:
        waveform, sample_rate = sf.read(filename, always_2d=True)
    except Exception:
        waveform, sample_rate = librosa.load(filename, sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        else:
            waveform = np.asarray(waveform)
        waveform = waveform.T

    waveform = np.asarray(waveform, dtype=np.float32)

    # Convert to mono consistently, regardless of backend output shape.
    waveform = waveform.mean(axis=1)

    if sample_rate != target_sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=target_sample_rate
        )

    return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)


class BioLingual:
    def __init__(self, *args, **kwargs):
        self.processor = ClapProcessor.from_pretrained("davidrrobinson/biolingual")
        model = ClapModel.from_pretrained("davidrrobinson/biolingual")

        self.device = infer_device()

        self.model = model.to(self.device).eval()

    def __call__(self, file_path: str):
        waveform = get_waveform(file_path, 48000)
        waveform = waveform.squeeze().numpy()

        processed = self.processor(
            audio=waveform, return_tensors="pt", sampling_rate=48000
        )
        inputs = processed["input_features"].to(self.device)

        with torch.no_grad():
            outputs = self.model.get_audio_features(input_features=inputs)

        return outputs.pooler_output.squeeze(0)


class Aves:
    def __init__(
        self,
        aves_model_path: str,
        aves_config_path: str,
        sample_rate: int = 44100,
        *args,
        **kwargs,
    ):
        try:
            from aves import load_feature_extractor
        except ImportError as exc:
            raise ImportError(
                "AVES support requires a package that exposes "
                "`aves.load_feature_extractor`, but the installed `aves` "
                "module does not provide it."
            ) from exc

        if not Path(aves_model_path).exists():
            raise FileNotFoundError(f"AVES model file not found: {aves_model_path}")
        if not Path(aves_config_path).exists():
            raise FileNotFoundError(f"AVES config file not found: {aves_config_path}")

        self.feature_extractor = load_feature_extractor(
            config_path=aves_config_path,
            model_path=aves_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            for_inference=True,
        )
        self.sample_rate = sample_rate

    def __call__(self, file_path):
        waveform = get_waveform(file_path, self.sample_rate)
        return (
            self.feature_extractor.extract_features(waveform, layers=-1)
            .mean(dim=1)
            .squeeze()
        )

class Dolph2Vec:
    def __init__(self, dolph2vec_model_path: str, dolph2vec_config_path: str, sample_rate: int = 44100, *args, **kwargs):
        config_path = Path(dolph2vec_config_path)
        if config_path.exists():
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(
                str(config_path)
            )
        else:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                dolph2vec_config_path
            )
        self.model = Wav2Vec2Model.from_pretrained(dolph2vec_model_path)

        self.device = infer_device()
        self.model = self.model.to(self.device).eval()
        self.sample_rate = sample_rate

    def __call__(self, file_path):
        waveform = get_waveform(file_path, self.sample_rate).squeeze().numpy()

        features = self.feature_extractor(
            raw_speech=waveform,
            padding="longest",
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        )["input_values"].to(self.device)

        with torch.no_grad():
            res = self.model(features, output_hidden_states=True)

        return res.hidden_states[-1].mean(1).squeeze()


class Spectrogram:
    def __init__(self, sample_rate: int = 44100, *args, **kwargs):
        self.sample_rate = sample_rate

    def __call__(self, file_path):
        waveform = get_waveform(file_path, self.sample_rate)
        waveform = waveform.squeeze().numpy()

        # Compute the spectrogram
        # spectrogram = torchaudio.transforms.Spectrogram()(torch.tensor(waveform))
        # spectrogram = torch.mean(spectrogram, dim=0)

        S = librosa.feature.melspectrogram(y=waveform, sr=self.sample_rate, n_mels=128)
        feats = np.mean(
            librosa.power_to_db(S, ref=np.max), axis=1
        )  # Averaged spectrogram
        return torch.Tensor(feats)


class MFCC:
    def __init__(self, sample_rate: int = 44100, n_mfcc: int = 13, *args, **kwargs):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def __call__(self, file_path):
        waveform = get_waveform(file_path, self.sample_rate)
        waveform = waveform.squeeze().numpy()

        mfccs = librosa.feature.mfcc(y=waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        return torch.Tensor(np.mean(mfccs, axis=1))  # Averaged MFCCs


class SpectralFeatures:
    def __init__(self, sample_rate: int = 44100, *args, **kwargs):
        self.sample_rate = sample_rate

    def __call__(self, file_path):
        waveform = get_waveform(file_path, self.sample_rate)
        waveform = waveform.squeeze().numpy()

        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform, sr=self.sample_rate
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform, sr=self.sample_rate
        )
        spectral_contrast = librosa.feature.spectral_contrast(
            y=waveform, sr=self.sample_rate
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=waveform, sr=self.sample_rate
        )

        feats = np.concatenate(
            [
                np.mean(spectral_centroid, axis=1),
                np.mean(spectral_bandwidth, axis=1),
                np.mean(spectral_contrast, axis=1),
                np.mean(spectral_rolloff, axis=1),
            ]
        )

        return torch.Tensor(feats)
