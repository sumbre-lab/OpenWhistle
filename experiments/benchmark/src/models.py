import io
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from scipy.signal import butter, hilbert, resample_poly, sosfilt, sosfiltfilt
from typing import Any
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    ClapModel,
    ClapProcessor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)



def _load_waveform_from_file(filename):
    try:
        waveform, sample_rate = sf.read(filename, always_2d=True)
    except Exception:
        waveform, sample_rate = librosa.load(filename, sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        else:
            waveform = np.asarray(waveform)
        waveform = waveform.T
    return waveform, sample_rate


def infer_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("| Using CUDA for computation.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("| Using Apple MPS for computation.")
    else:
        device = torch.device("cpu")
        print("| Using CPU for computation.")
    return device


def load_waveform(audio_source: Any):
    if hasattr(audio_source, "get_all_samples"):
        decoded = audio_source.get_all_samples()
        waveform = decoded.data.detach().cpu().numpy().T
        sample_rate = int(decoded.sample_rate)
    elif isinstance(audio_source, dict):
        if audio_source.get("array") is not None:
            waveform = np.asarray(audio_source["array"], dtype=np.float32)
            sample_rate = int(audio_source["sampling_rate"])
            if waveform.ndim == 1:
                waveform = waveform[:, np.newaxis]
            elif waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
                waveform = waveform.T
        elif audio_source.get("path"):
            waveform, sample_rate = _load_waveform_from_file(audio_source["path"])
        elif audio_source.get("bytes") is not None:
            waveform, sample_rate = sf.read(
                io.BytesIO(audio_source["bytes"]),
                dtype="float32",
                always_2d=True,
            )
        else:
            raise ValueError(
                "Unsupported audio mapping: expected an 'array', 'path', or 'bytes'."
            )
    else:
        waveform, sample_rate = _load_waveform_from_file(audio_source)

    waveform = np.asarray(waveform, dtype=np.float32)

    # Convert to mono consistently, regardless of backend output shape.
    waveform = waveform.mean(axis=1)
    waveform = np.nan_to_num(waveform, copy=False)
    return waveform, int(sample_rate)


def get_waveform(audio_source: Any, target_sample_rate):
    waveform, sample_rate = load_waveform(audio_source)
    if sample_rate != target_sample_rate:
        waveform = (
            torchaudio.functional.resample(
                torch.from_numpy(waveform),
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )
            .cpu()
            .numpy()
        )

    return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)


BTB3_ENCODER_SAMPLE_RATE = 16000
BTB3_SAFE_UPPER_HZ = 21800.0


def _match_length(waveform: np.ndarray, length: int) -> np.ndarray:
    if waveform.shape[0] > length:
        waveform = waveform[:length]
    elif waveform.shape[0] < length:
        waveform = np.pad(waveform, (0, length - waveform.shape[0]))
    return waveform.astype(np.float32, copy=False)


def _resample_poly_exact(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == BTB3_ENCODER_SAMPLE_RATE:
        return waveform.astype(np.float32, copy=False)
    ratio = Fraction(BTB3_ENCODER_SAMPLE_RATE, sample_rate).limit_denominator(1000)
    output = resample_poly(
        waveform,
        ratio.numerator,
        ratio.denominator,
    ).astype(np.float32)
    target_length = int(
        round(
            waveform.shape[0]
            * BTB3_ENCODER_SAMPLE_RATE
            / float(sample_rate)
        )
    )
    return _match_length(output, target_length)


def _filter_sos(
    waveform: np.ndarray,
    sample_rate: int,
    cutoff: float | tuple[float, float],
    filter_type: str,
) -> np.ndarray:
    nyquist = 0.5 * float(sample_rate)
    if isinstance(cutoff, tuple):
        low_hz, high_hz = cutoff
        normalized_cutoff = [
            max(float(low_hz), 1.0) / nyquist,
            min(float(high_hz), nyquist * 0.98) / nyquist,
        ]
    else:
        normalized_cutoff = min(float(cutoff), nyquist * 0.98) / nyquist
    sos = butter(8, normalized_cutoff, btype=filter_type, output="sos")
    try:
        return sosfiltfilt(sos, waveform).astype(np.float32)
    except ValueError:
        return sosfilt(sos, waveform).astype(np.float32)


def make_btb3_views(
    waveform: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the BTB3 transform used for continued pretraining on Jean Zay."""
    waveform = np.asarray(waveform, dtype=np.float32)
    band0 = _resample_poly_exact(
        _filter_sos(waveform, sample_rate, 8000.0, "lowpass"),
        sample_rate,
    )

    shifted_views = []
    for low_hz, high_hz in (
        (8000.0, 16000.0),
        (16000.0, BTB3_SAFE_UPPER_HZ),
    ):
        band = _filter_sos(
            waveform,
            sample_rate,
            (low_hz, high_hz),
            "bandpass",
        )
        analytic = hilbert(band.astype(np.float64, copy=False))
        time = np.arange(band.shape[0], dtype=np.float64) / float(sample_rate)
        shifted = np.real(
            analytic * np.exp(-2j * np.pi * low_hz * time)
        ).astype(np.float32)
        shifted = _filter_sos(shifted, sample_rate, 8000.0, "lowpass")
        shifted_views.append(_resample_poly_exact(shifted, sample_rate))

    return (
        band0,
        _match_length(shifted_views[0], band0.shape[0]),
        _match_length(shifted_views[1], band0.shape[0]),
    )


class BioLingual:
    def __init__(self, *args, **kwargs):
        self.processor = ClapProcessor.from_pretrained("davidrrobinson/biolingual")
        model = ClapModel.from_pretrained("davidrrobinson/biolingual")

        self.device = infer_device()

        self.model = model.to(self.device).eval()

    def __call__(self, file_path: str):
        waveform = get_waveform(file_path, 48000)
        waveform = waveform.squeeze().numpy()

        processed = self.processor.feature_extractor(
            waveform, return_tensors="pt", sampling_rate=48000
        )
        inputs = processed["input_features"].to(self.device)

        with torch.no_grad():
            outputs = self.model.get_audio_features(input_features=inputs)

        features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        return features.squeeze(0)


class Aves:
    def __init__(
        self,
        aves_model_path: str,
        aves_config_path: str,
        sample_rate: int = 44100,
        hf_feature_mode: str = "standard",
        audio_workers: int = 0,
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
        self.feature_mode = hf_feature_mode
        self.audio_executor = (
            ThreadPoolExecutor(
                max_workers=audio_workers,
                thread_name_prefix="openwhistle-aves-audio",
            )
            if audio_workers > 0
            else None
        )

    def embed_batch(self, audio_sources):
        if not self.feature_mode.startswith("btb3_"):
            return torch.stack([self(audio_source) for audio_source in audio_sources])

        def prepare_views(audio_source):
            waveform, sample_rate = load_waveform(audio_source)
            return make_btb3_views(waveform, sample_rate)

        if self.audio_executor is None:
            source_views = [prepare_views(source) for source in audio_sources]
        else:
            source_views = list(self.audio_executor.map(prepare_views, audio_sources))

        flat_views = [
            torch.from_numpy(view)
            for views in source_views
            for view in views
        ]
        lengths = torch.tensor(
            [view.shape[0] for view in flat_views],
            dtype=torch.int64,
        )
        inputs = torch.nn.utils.rnn.pad_sequence(
            flat_views,
            batch_first=True,
        )
        device = next(self.feature_extractor.model.parameters()).device
        inputs = inputs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        with torch.inference_mode():
            layer_outputs, output_lengths = (
                self.feature_extractor.model.extract_features(inputs, lengths)
            )
        hidden = layer_outputs[-1]
        positions = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
        mask = positions < output_lengths.unsqueeze(1)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (
            (hidden * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        )
        pooled = pooled.reshape(len(audio_sources), 3, -1)
        if self.feature_mode == "btb3_concat":
            return pooled.reshape(len(audio_sources), -1)
        if self.feature_mode == "btb3_mean":
            return pooled.mean(dim=1)
        raise ValueError(f"Unknown AVES feature mode: {self.feature_mode}")

    def __call__(self, file_path):
        if self.feature_mode.startswith("btb3_"):
            waveform, sample_rate = load_waveform(file_path)
            views = make_btb3_views(waveform, sample_rate)
            band_embeddings = []
            for view in views:
                view_tensor = torch.from_numpy(view).unsqueeze(0)
                band_embeddings.append(
                    self.feature_extractor.extract_features(
                        view_tensor,
                        layers=-1,
                    )
                    .mean(dim=1)
                    .squeeze(0)
                )
            stacked = torch.stack(band_embeddings)
            if self.feature_mode == "btb3_concat":
                return stacked.reshape(-1)
            if self.feature_mode == "btb3_mean":
                return stacked.mean(dim=0)
            raise ValueError(f"Unknown AVES feature mode: {self.feature_mode}")

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


class HuggingFaceAudioBackbone:
    """Mean-pool the last hidden state of a Transformers audio backbone."""

    def __init__(
        self,
        hf_model_id: str,
        hf_sample_rate: int | None = None,
        mixed_precision: bool = False,
        amp_dtype: str = "float16",
        audio_workers: int = 0,
        hf_feature_mode: str = "standard",
        *args,
        **kwargs,
    ):
        if not hf_model_id:
            raise ValueError("hf_model_id is required for a Hugging Face backbone.")

        self.model_id = hf_model_id
        self.feature_mode = hf_feature_mode
        if self.feature_mode not in {
            "standard",
            "btb3_concat",
            "btb3_mean",
        }:
            raise ValueError(f"Unknown Hugging Face feature mode: {self.feature_mode}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_id)
        self.model = AutoModel.from_pretrained(hf_model_id)
        self.sample_rate = int(
            hf_sample_rate
            if hf_sample_rate is not None
            else getattr(self.feature_extractor, "sampling_rate", 44100)
        )
        if hf_sample_rate is not None:
            self.feature_extractor.sampling_rate = self.sample_rate
        self.device = infer_device()
        self.model = self.model.to(self.device).eval()
        self.mixed_precision = bool(mixed_precision and self.device.type == "cuda")
        self.amp_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[amp_dtype]
        self.audio_executor = (
            ThreadPoolExecutor(
                max_workers=audio_workers,
                thread_name_prefix="openwhistle-audio",
            )
            if audio_workers > 0
            else None
        )

    def _pool_hidden_state(self, hidden, attention_mask):
        if attention_mask is None:
            return hidden.mean(dim=1)

        # Convert the waveform-level mask to the feature-vector length when the
        # model exposes the standard Wav2Vec2/HuBERT helper.
        if hasattr(self.model, "_get_feature_vector_attention_mask"):
            attention_mask = self.model._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            )
        elif attention_mask.shape[1] != hidden.shape[1]:
            return hidden.mean(dim=1)

        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def embed_batch(self, audio_sources):
        def prepare_waveform(audio_source):
            if self.feature_mode.startswith("btb3_"):
                waveform, sample_rate = load_waveform(audio_source)
                return make_btb3_views(waveform, sample_rate)
            return get_waveform(
                audio_source,
                self.sample_rate,
            ).squeeze().numpy()

        if self.audio_executor is None:
            waveforms = [
                prepare_waveform(audio_source)
                for audio_source in audio_sources
            ]
        else:
            waveforms = list(
                self.audio_executor.map(prepare_waveform, audio_sources)
            )
        if self.feature_mode.startswith("btb3_"):
            waveforms = [
                view
                for source_views in waveforms
                for view in source_views
            ]
            encoder_sample_rate = BTB3_ENCODER_SAMPLE_RATE
        else:
            encoder_sample_rate = self.sample_rate

        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=encoder_sample_rate,
            padding=True,
            # Several Wav2Vec2 checkpoints (including Dolph2Vec) default to
            # return_attention_mask=False.  That is harmless for single-file
            # inference, but batched mean pooling would otherwise include the
            # right-padding frames and produce different embeddings depending
            # on the other files in the batch.
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.pin_memory().to(self.device, non_blocking=True)
            if self.device.type == "cuda"
            else value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.mixed_precision,
            ):
                outputs = self.model(**model_inputs)

        hidden = outputs.last_hidden_state
        attention_mask = model_inputs.get("attention_mask")
        pooled = self._pool_hidden_state(hidden, attention_mask).float()
        if self.feature_mode.startswith("btb3_"):
            pooled = pooled.reshape(len(audio_sources), 3, -1)
            if self.feature_mode == "btb3_concat":
                return pooled.reshape(len(audio_sources), -1)
            return pooled.mean(dim=1)
        return pooled

    def __call__(self, file_path):
        return self.embed_batch([file_path]).squeeze(0)


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
