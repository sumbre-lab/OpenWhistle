"""Noise-subtracted SNR around a tracked spectrogram ridge (no learned model)."""
from dataclasses import asdict, dataclass
from io import BytesIO

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import spectrogram


@dataclass(frozen=True)
class SNRConfig:
    low_hz: float = 3000.0
    high_hz: float = 25000.0
    nfft: int = 2048
    hop: int = 256
    ridge_half_width_hz: float = 300.0
    noise_offset_hz: float = 400.0
    noise_width_hz: float = 2000.0

    def to_dict(self):
        return asdict(self)


def decode_audio(payload):
    if isinstance(payload, dict) and payload.get('bytes') is not None:
        source = BytesIO(payload['bytes'])
    elif isinstance(payload, dict) and payload.get('path'):
        source = payload['path']
    else:
        raise ValueError('Expected a Hugging Face Audio(decode=False) bytes/path payload.')
    audio, fs = sf.read(source, dtype='float32', always_2d=True)
    return audio.mean(axis=1), int(fs)


def estimate_snr(audio, fs, config=SNRConfig()):
    """Return (dB, status, number of frames); pool energies across the full clip."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError('Audio must be a finite mono waveform.')
    if len(audio) < config.nfft:
        return float('nan'), 'too_short', 0
    frequencies, _, power = spectrogram(
        audio, fs=fs, window='blackman', nperseg=config.nfft,
        noverlap=config.nfft - config.hop, nfft=config.nfft,
        scaling='density', mode='psd',
    )
    selected = (frequencies >= config.low_hz) & (frequencies <= config.high_hz)
    band = power[selected]
    if len(band) < 3:
        return float('nan'), 'frequency_band_unavailable', 0
    enhanced = median_filter(np.maximum(band - np.median(band, axis=0), 0), size=(3, 3))
    indices = np.argmax(enhanced, axis=0)
    peaks = enhanced[indices, np.arange(band.shape[1])]
    valid = (peaks > 0) & (peaks >= np.percentile(peaks, 30))
    if not valid.any():
        return float('nan'), 'no_ridge_detected', 0
    times = np.arange(len(indices))
    indices = median_filter(np.interp(times, times[valid], indices[valid]), size=5)
    indices = np.rint(indices).astype(int)
    df = fs / config.nfft
    width = max(1, int(config.ridge_half_width_hz / df))
    offset = max(width + 1, int(config.noise_offset_hz / df))
    noise_width = max(1, int(config.noise_width_hz / df))
    cumulative = np.vstack([np.zeros((1, band.shape[1])), np.cumsum(band, axis=0, dtype=np.float64)])

    def energy(lo, hi):
        lo, hi = np.clip(lo, 0, len(band)), np.clip(hi, 0, len(band))
        return float(np.sum(cumulative[hi, times] - cumulative[lo, times])), int(np.sum(hi - lo))

    signal, n_signal = energy(indices - width, indices + width + 1)
    lower, n_lower = energy(indices - offset - noise_width, indices - offset)
    upper, n_upper = energy(indices + offset + 1, indices + offset + noise_width + 1)
    if not (n_lower + n_upper):
        return float('nan'), 'no_noise_bins', 0
    noise = (lower + upper) / (n_lower + n_upper) * n_signal
    if noise <= 0:
        return float('nan'), 'no_noise_energy', 0
    # Relative floor preserves amplitude invariance and caps undefined negative signal at -120 dB.
    ratio = max((signal - noise) / noise, 1e-12)
    return float(10 * np.log10(ratio)), 'ok', band.shape[1]
