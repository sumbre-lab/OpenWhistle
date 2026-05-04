import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from utils.model import (
    SpectrogramConfig,
    build_spectrogram_plan,
    build_vgg16_whistle_classifier,
    make_spectrogram_batch,
    normalize_uint8_batch_to_torch,
    resample_audio_if_needed,
    torchvision_image_normalization,
)
from utils.runtime_utils import get_device

def parse_optional_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {'', 'none', 'null', '0'}:
        return None
    return int(normalized)

def parse_float_tuple(value: object, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
        return tuple(float(part) for part in parts)
    return tuple(float(part) for part in value)

def read_file_list(path: Path) -> list[str]:
    with path.open('r', encoding='utf-8') as handle:
        return [line.strip() for line in handle if line.strip()]

@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_path: Path | None = None
    model_repo: str = 'OpenWhistleNeurIPS26/OpenWhistle-CNN-VGG16'
    model_filename: str = 'model_vgg_final_best.pt'
    recordings_dir: Path = Path('.')
    output_dir: Path = Path('cnn/runs/inference')
    batch_size: int = 64
    threshold: float = 0.5
    start_time: float = 0.0
    end_time: float | None = None
    save_positive_spectrograms: bool = False
    specific_files_path: Path | None = None
    recursive: bool = False
    limit: int = 0
    cpu_only: bool = False
    spectrogram_config: SpectrogramConfig = field(default_factory=SpectrogramConfig)

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> 'InferenceConfig':
        defaults = cls()
        parser = argparse.ArgumentParser(
            description='Run OpenWhistle CNN inference on WAV/FLAC recordings.'
        )
        parser.add_argument(
            '--checkpoint-path',
            type=Path,
            default=defaults.checkpoint_path,
            help='Local checkpoint override. Defaults to the Hugging Face model.',
        )
        parser.add_argument(
            '--model-repo',
            default=defaults.model_repo,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--model-filename',
            default=defaults.model_filename,
            help=argparse.SUPPRESS,
        )
        parser.add_argument('--recordings-dir', type=Path, required=True)
        parser.add_argument('--output-dir', type=Path, default=defaults.output_dir)
        parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
        parser.add_argument('--threshold', type=float, default=defaults.threshold)
        parser.add_argument('--start-time', type=float, default=defaults.start_time)
        parser.add_argument('--end-time', type=float, default=defaults.end_time)
        parser.add_argument(
            '--save-positive-spectrograms',
            action='store_true',
            default=defaults.save_positive_spectrograms,
        )
        parser.add_argument('--specific-files', type=Path, default=defaults.specific_files_path)
        parser.add_argument(
            '--recursive',
            action='store_true',
            default=defaults.recursive,
            help='Search WAV/FLAC files recursively under --recordings-dir.',
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=defaults.limit,
            help='Optional cap on the number of files to process; 0 means no cap.',
        )
        parser.add_argument('--cpu-only', action='store_true', default=defaults.cpu_only)
        parser.add_argument(
            '--target-fs',
            type=parse_optional_int,
            default=defaults.spectrogram_config.target_fs,
            help='Target sampling rate; use "none" or "0" to disable resampling.',
        )
        parser.add_argument(
            '--cut-low-frequency',
            type=float,
            default=defaults.spectrogram_config.cut_low_frequency,
        )
        parser.add_argument(
            '--cut-high-frequency',
            type=float,
            default=defaults.spectrogram_config.cut_high_frequency,
        )
        args = parser.parse_args(argv)
        spectrogram_config = SpectrogramConfig(
            cut_low_frequency=args.cut_low_frequency,
            cut_high_frequency=args.cut_high_frequency,
            target_fs=args.target_fs,
        )
        return cls(
            checkpoint_path=args.checkpoint_path,
            model_repo=args.model_repo,
            model_filename=args.model_filename,
            recordings_dir=args.recordings_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            threshold=args.threshold,
            start_time=args.start_time,
            end_time=args.end_time,
            save_positive_spectrograms=args.save_positive_spectrograms,
            specific_files_path=args.specific_files,
            recursive=args.recursive,
            limit=args.limit,
            cpu_only=args.cpu_only,
            spectrogram_config=spectrogram_config,
        )

    def validate(self) -> None:
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {self.checkpoint_path}')
        if not self.recordings_dir.exists():
            raise FileNotFoundError(f'Recordings folder not found: {self.recordings_dir}')
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive.')
        if not 0 <= self.threshold <= 1:
            raise ValueError('threshold must be between 0 and 1.')
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError('end_time must be greater than start_time.')
        if self.specific_files_path is not None and not self.specific_files_path.exists():
            raise FileNotFoundError(f'Specific file list not found: {self.specific_files_path}')
        if self.limit < 0:
            raise ValueError('limit must be zero or positive.')

    def resolved_checkpoint_path(self) -> Path:
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        checkpoint = hf_hub_download(
            repo_id=self.model_repo,
            filename=self.model_filename,
        )
        return Path(checkpoint)

@dataclass
class InferenceModel:
    model: torch.nn.Module
    device: torch.device
    image_size: tuple[int, int]
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    spectrogram_config: SpectrogramConfig

def load_audio(
    path: Path,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> tuple[int, np.ndarray]:
    with sf.SoundFile(path) as handle:
        fs = int(handle.samplerate)
        start_frame = int(start_time * fs)
        handle.seek(start_frame)
        frames = -1 if end_time is None else max(0, int(end_time * fs) - start_frame)
        audio = handle.read(frames=frames, always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return fs, np.asarray(audio, dtype=np.float32)

def load_inference_model(checkpoint_path: Path, cpu_only: bool) -> InferenceModel:
    device = torch.device('cpu') if cpu_only else get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_vgg16_whistle_classifier(
        pretrained_backbone=False,
        freeze_backbone=bool(checkpoint.get('freeze_backbone', False)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    img_size = int(checkpoint.get('img_size', 224))
    default_normalization = torchvision_image_normalization()
    normalization = checkpoint.get('normalization', {})
    mean = parse_float_tuple(normalization.get('mean'), default_normalization.mean)
    std = parse_float_tuple(normalization.get('std'), default_normalization.std)
    spectrogram_config = SpectrogramConfig.from_metadata(
        checkpoint.get('spectrogram_config')
    )
    spectrogram_config = SpectrogramConfig(
        image_size=(img_size, img_size),
        cut_low_frequency=spectrogram_config.cut_low_frequency,
        cut_high_frequency=spectrogram_config.cut_high_frequency,
        wlen=spectrogram_config.wlen,
        nfft=spectrogram_config.nfft,
        sliding_window=spectrogram_config.sliding_window,
        target_fs=spectrogram_config.target_fs,
    )
    return InferenceModel(
        model=model,
        device=device,
        image_size=(img_size, img_size),
        normalization_mean=mean,
        normalization_std=std,
        spectrogram_config=spectrogram_config,
    )

def audio_files(config: InferenceConfig) -> list[Path]:
    suffixes = {'.wav', '.flac'}
    if config.specific_files_path is not None:
        files = [
            config.recordings_dir / item
            for item in read_file_list(config.specific_files_path)
            if (config.recordings_dir / item).suffix.lower() in suffixes
        ]
    else:
        iterator = (
            config.recordings_dir.rglob('*')
            if config.recursive
            else config.recordings_dir.iterdir()
        )
        files = sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in suffixes
        )
    if config.limit > 0:
        files = files[: config.limit]
    return files

def relative_file_name(audio_path: Path, recordings_dir: Path) -> str:
    try:
        return audio_path.relative_to(recordings_dir).as_posix()
    except ValueError:
        return audio_path.name

def output_stem(audio_path: Path, recordings_dir: Path) -> str:
    relative_name = relative_file_name(audio_path, recordings_dir)
    stem_path = Path(relative_name).with_suffix('')
    if len(stem_path.parts) == 1:
        return audio_path.stem
    return re.sub(r'[^A-Za-z0-9._=-]+', '__', stem_path.as_posix()).strip('_')

def predict_batch(model: InferenceModel, images_uint8: np.ndarray) -> np.ndarray:
    inputs = normalize_uint8_batch_to_torch(
        images_uint8,
        mean=model.normalization_mean,
        std=model.normalization_std,
        device=model.device,
    )
    with torch.inference_mode():
        logits = model.model(inputs)
        return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

def write_prediction_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['file_name', 'initial_point', 'finish_point', 'confidence'],
        )
        writer.writeheader()
        writer.writerows(rows)

def save_positive_spectrogram(
    image_uint8: np.ndarray,
    start_time: float,
    end_time: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{start_time:.2f}-{end_time:.2f}.jpg'
    cv2.imwrite(str(path), image_uint8)

def process_file(
    audio_path: Path,
    config: InferenceConfig,
    inference_model: InferenceModel,
) -> list[dict[str, object]]:
    start_time = float(config.start_time)
    fs, audio = load_audio(audio_path, start_time, config.end_time)
    spectrogram_config = SpectrogramConfig(
        image_size=inference_model.image_size,
        cut_low_frequency=config.spectrogram_config.cut_low_frequency,
        cut_high_frequency=config.spectrogram_config.cut_high_frequency,
        wlen=inference_model.spectrogram_config.wlen,
        nfft=inference_model.spectrogram_config.nfft,
        sliding_window=inference_model.spectrogram_config.sliding_window,
        target_fs=config.spectrogram_config.target_fs,
    )
    fs, audio = resample_audio_if_needed(audio, fs, spectrogram_config.target_fs)
    samples_per_window = round(spectrogram_config.sliding_window * fs)
    total_windows = max(0, len(audio) // samples_per_window)
    spectrogram_plan = build_spectrogram_plan(fs, spectrogram_config)
    rows: list[dict[str, object]] = []
    file_name = relative_file_name(audio_path, config.recordings_dir)
    file_stem = output_stem(audio_path, config.recordings_dir)
    file_output_dir = config.output_dir / file_stem
    positives_dir = file_output_dir / 'positive'
    for batch_index in range(0, total_windows, config.batch_size):
        batch_start = batch_index * samples_per_window
        images, start_times = make_spectrogram_batch(
            audio,
            fs,
            batch_start,
            min(config.batch_size, total_windows - batch_index),
            spectrogram_config,
            spectrogram_plan,
        )
        if len(images) == 0:
            continue
        scores = predict_batch(inference_model, images)
        for index, score in enumerate(scores):
            if float(score) < config.threshold:
                continue
            start = round(start_time + float(start_times[index]), 2)
            end = round(start + spectrogram_config.sliding_window, 2)
            rows.append(
                {
                    'file_name': file_name,
                    'initial_point': start,
                    'finish_point': end,
                    'confidence': float(score),
                }
            )
            if config.save_positive_spectrograms:
                save_positive_spectrogram(images[index], start, end, positives_dir)
    prediction_path = file_output_dir / f'{file_stem}.wav_predictions.csv'
    write_prediction_csv(prediction_path, rows)
    return rows

def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['file_name', 'initial_point', 'finish_point', 'confidence'],
        )
        writer.writeheader()
        writer.writerows(rows)

class InferenceRun:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.model: InferenceModel | None = None
        self.files: list[Path] = []

    def setup(self) -> None:
        self.config.validate()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.files = audio_files(self.config)
        if not self.files:
            return
        checkpoint_path = self.config.resolved_checkpoint_path()
        print(f'Using checkpoint: {checkpoint_path}')
        self.model = load_inference_model(
            checkpoint_path,
            self.config.cpu_only,
        )

    def run(self) -> None:
        self.setup()
        if not self.files:
            print('No WAV/FLAC files found.')
            return
        if self.model is None:
            raise RuntimeError('Inference model was not initialized.')
        print(f'Using device: {self.model.device}')
        print(f'Found {len(self.files)} files.')
        rows = self.predict_files()
        self.write_outputs(rows)

    def predict_files(self) -> list[dict[str, object]]:
        if self.model is None:
            raise RuntimeError('Inference model was not initialized.')
        all_rows: list[dict[str, object]] = []
        for audio_path in tqdm(self.files, desc='files'):
            rows = process_file(audio_path, self.config, self.model)
            all_rows.extend(rows)
            print(f'{audio_path.name}: {len(rows)} detections')
        return all_rows

    def write_outputs(self, rows: list[dict[str, object]]) -> None:
        summary_path = self.config.output_dir / 'detections.csv'
        write_summary(summary_path, rows)
        print(f'Wrote {len(rows)} detections to {summary_path}')

def main(argv: list[str] | None = None) -> None:
    InferenceRun(InferenceConfig.from_args(argv)).run()

if __name__ == '__main__':
    main()
