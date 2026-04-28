import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import random
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from itertools import combinations
from io import BytesIO
from pathlib import Path
from typing import Literal, get_args

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Audio as HFAudio
from datasets import Dataset as HFDataset
from datasets import (
    DatasetDict,
    Image as HFImage,
    get_dataset_split_names,
    load_dataset,
    load_from_disk,
)
from PIL import Image as PILImage
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
import wandb

from modeling_components import (
    SpectrogramConfig,
    audio_payload_to_spectrogram_image,
    build_vgg16_whistle_classifier,
    normalize_uint8_image_to_tensor,
    torchvision_image_normalization,
)

InputSource = Literal['audio', 'spectrogram']

def input_source_choices() -> tuple[str, ...]:
    return get_args(InputSource)

def default_normalization_mean() -> tuple[float, float, float]:
    return torchvision_image_normalization().mean

def default_normalization_std() -> tuple[float, float, float]:
    return torchvision_image_normalization().std

def env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {'0', 'false', 'no'}

def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()

def env_optional_int(name: str, default: int | None) -> int | None:
    default_value = 'none' if default is None else str(default)
    raw_value = env_str(name, default_value).lower()
    if raw_value in {'', 'none', 'null'}:
        return None
    return int(raw_value)

def parse_optional_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {'', 'none', 'null'}:
        return None
    return int(normalized)

def format_float_tuple(values: tuple[float, ...]) -> str:
    return ','.join(str(value) for value in values)

def parse_float_tuple(value: str, expected_length: int = 3) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(',') if part.strip()]
    if len(parts) != expected_length:
        raise argparse.ArgumentTypeError(
            f'Expected {expected_length} comma-separated floats, got {value!r}.'
        )
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'Expected comma-separated floats, got {value!r}.'
        ) from exc

def add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    parser.add_argument(
        f'--{name}',
        dest=name.replace('-', '_'),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )

@dataclass(frozen=True)
class TrainConfig:
    img_size: int = 224
    batch_size: int = 4
    num_epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-5
    random_state: int = 7
    num_workers: int = 0
    dataset_source: str = 'dolphinteam/OpenWhistle-1.0-CNN'
    models_dir: str = 'artifacts/cnn/models'
    figs_dir: str = 'artifacts/cnn/figs'
    reports_dir: str = 'artifacts/cnn/reports'
    train_split: str = 'train'
    validation_split: str = 'validation'
    test_split: str = 'test'
    train_input_source: InputSource = 'spectrogram'
    normalization_mean: tuple[float, float, float] = field(
        default_factory=default_normalization_mean
    )
    normalization_std: tuple[float, float, float] = field(
        default_factory=default_normalization_std
    )
    wandb_enabled: bool = True
    wandb_project: str = 'dolphin-whistle-training'
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = 'online'
    wandb_console: str = 'redirect'
    use_amp: bool = True
    eval_only: bool = False
    spectrogram_cache_enabled: bool = True
    pretrained_backbone: bool = True
    freeze_backbone: bool = False
    spectrogram_config: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    min_learning_rate: float = 1e-6
    spectrogram_cache_dir: Path = Path('artifacts/cnn/spectrogram_cache')
    checkpoint_path: str | None = None

    @classmethod
    def defaults_from_env(cls) -> 'TrainConfig':
        img_size = int(env_str('TRAIN_IMG_SIZE', str(cls.img_size)))
        spectrogram_defaults = SpectrogramConfig()
        normalization_defaults = torchvision_image_normalization()
        return cls(
            img_size=img_size,
            batch_size=int(env_str('TRAIN_BATCH_SIZE', str(cls.batch_size))),
            num_epochs=int(env_str('TRAIN_NUM_EPOCHS', str(cls.num_epochs))),
            patience=int(env_str('TRAIN_PATIENCE', str(cls.patience))),
            learning_rate=float(env_str('TRAIN_LEARNING_RATE', str(cls.learning_rate))),
            random_state=int(env_str('TRAIN_RANDOM_STATE', str(cls.random_state))),
            num_workers=int(env_str('TRAIN_NUM_WORKERS', str(cls.num_workers))),
            dataset_source=env_str('TRAIN_DATASET_SOURCE', cls.dataset_source),
            models_dir=env_str('TRAIN_MODELS_DIR', cls.models_dir),
            figs_dir=env_str('TRAIN_FIGS_DIR', cls.figs_dir),
            reports_dir=env_str('TRAIN_REPORTS_DIR', cls.reports_dir),
            train_split=env_str('TRAIN_SPLIT', cls.train_split),
            validation_split=env_str('TRAIN_VALIDATION_SPLIT', cls.validation_split),
            test_split=env_str('TRAIN_TEST_SPLIT', cls.test_split),
            train_input_source=env_str(
                'TRAIN_INPUT_SOURCE',
                cls.train_input_source,
            ).lower(),
            normalization_mean=parse_float_tuple(
                env_str(
                    'TRAIN_NORMALIZATION_MEAN',
                    format_float_tuple(normalization_defaults.mean),
                )
            ),
            normalization_std=parse_float_tuple(
                env_str(
                    'TRAIN_NORMALIZATION_STD',
                    format_float_tuple(normalization_defaults.std),
                )
            ),
            wandb_enabled=env_bool('WANDB_ENABLED', cls.wandb_enabled),
            wandb_project=env_str('WANDB_PROJECT', cls.wandb_project),
            wandb_entity=os.environ.get('WANDB_ENTITY'),
            wandb_run_name=os.environ.get('WANDB_RUN_NAME'),
            wandb_mode=env_str('WANDB_MODE', cls.wandb_mode),
            wandb_console=env_str('WANDB_CONSOLE', cls.wandb_console),
            use_amp=env_bool('TRAIN_USE_AMP', cls.use_amp),
            eval_only=env_bool('TRAIN_EVAL_ONLY', cls.eval_only),
            spectrogram_cache_enabled=env_bool(
                'TRAIN_SPECTROGRAM_CACHE',
                cls.spectrogram_cache_enabled,
            ),
            pretrained_backbone=env_bool(
                'TRAIN_PRETRAINED_BACKBONE',
                cls.pretrained_backbone,
            ),
            freeze_backbone=env_bool('TRAIN_FREEZE_BACKBONE', cls.freeze_backbone),
            spectrogram_config=SpectrogramConfig(
                image_size=(img_size, img_size),
                cut_low_frequency=float(
                    env_str(
                        'TRAIN_SPECTROGRAM_CUT_LOW_FREQUENCY',
                        str(spectrogram_defaults.cut_low_frequency),
                    )
                ),
                cut_high_frequency=float(
                    env_str(
                        'TRAIN_SPECTROGRAM_CUT_HIGH_FREQUENCY',
                        str(spectrogram_defaults.cut_high_frequency),
                    )
                ),
                wlen=int(
                    env_str('TRAIN_SPECTROGRAM_WLEN', str(spectrogram_defaults.wlen))
                ),
                nfft=int(
                    env_str('TRAIN_SPECTROGRAM_NFFT', str(spectrogram_defaults.nfft))
                ),
                sliding_window=float(
                    env_str(
                        'TRAIN_SPECTROGRAM_SLIDING_WINDOW',
                        str(spectrogram_defaults.sliding_window),
                    )
                ),
                target_fs=env_optional_int(
                    'TRAIN_SPECTROGRAM_TARGET_FS',
                    spectrogram_defaults.target_fs,
                ),
            ),
            lr_scheduler_factor=float(
                env_str('TRAIN_LR_SCHEDULER_FACTOR', str(cls.lr_scheduler_factor))
            ),
            lr_scheduler_patience=int(
                env_str('TRAIN_LR_SCHEDULER_PATIENCE', str(cls.lr_scheduler_patience))
            ),
            min_learning_rate=float(
                env_str('TRAIN_MIN_LEARNING_RATE', str(cls.min_learning_rate))
            ),
            spectrogram_cache_dir=Path(
                env_str('TRAIN_SPECTROGRAM_CACHE_DIR', str(cls.spectrogram_cache_dir))
            ),
            checkpoint_path=os.environ.get('TRAIN_CHECKPOINT_PATH'),
        )

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> 'TrainConfig':
        defaults = cls.defaults_from_env()
        parser = argparse.ArgumentParser(
            description='Train and evaluate the OpenWhistle CNN classifier.'
        )
        parser.add_argument('--img-size', type=int, default=defaults.img_size)
        parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
        parser.add_argument('--num-epochs', type=int, default=defaults.num_epochs)
        parser.add_argument('--patience', type=int, default=defaults.patience)
        parser.add_argument('--learning-rate', type=float, default=defaults.learning_rate)
        parser.add_argument('--random-state', type=int, default=defaults.random_state)
        parser.add_argument('--num-workers', type=int, default=defaults.num_workers)
        parser.add_argument('--dataset-source', default=defaults.dataset_source)
        parser.add_argument('--models-dir', default=defaults.models_dir)
        parser.add_argument('--figs-dir', default=defaults.figs_dir)
        parser.add_argument('--reports-dir', default=defaults.reports_dir)
        parser.add_argument('--train-split', default=defaults.train_split)
        parser.add_argument('--validation-split', default=defaults.validation_split)
        parser.add_argument('--test-split', default=defaults.test_split)
        parser.add_argument(
            '--train-input-source',
            choices=input_source_choices(),
            default=defaults.train_input_source,
        )
        parser.add_argument(
            '--normalization-mean',
            type=parse_float_tuple,
            default=defaults.normalization_mean,
            help='Comma-separated RGB normalization mean; defaults to torchvision ImageNet.',
        )
        parser.add_argument(
            '--normalization-std',
            type=parse_float_tuple,
            default=defaults.normalization_std,
            help='Comma-separated RGB normalization std; defaults to torchvision ImageNet.',
        )
        add_bool_arg(parser, 'wandb-enabled', defaults.wandb_enabled, 'Log to wandb.')
        parser.add_argument('--wandb-project', default=defaults.wandb_project)
        parser.add_argument('--wandb-entity', default=defaults.wandb_entity)
        parser.add_argument('--wandb-run-name', default=defaults.wandb_run_name)
        parser.add_argument('--wandb-mode', default=defaults.wandb_mode)
        parser.add_argument('--wandb-console', default=defaults.wandb_console)
        add_bool_arg(parser, 'use-amp', defaults.use_amp, 'Use CUDA AMP when available.')
        add_bool_arg(parser, 'eval-only', defaults.eval_only, 'Only evaluate a checkpoint.')
        add_bool_arg(
            parser,
            'spectrogram-cache-enabled',
            defaults.spectrogram_cache_enabled,
            'Cache generated spectrograms on disk.',
        )
        add_bool_arg(
            parser,
            'pretrained-backbone',
            defaults.pretrained_backbone,
            'Initialize the CNN backbone with ImageNet weights.',
        )
        add_bool_arg(
            parser,
            'freeze-backbone',
            defaults.freeze_backbone,
            'Train only the classifier head.',
        )
        parser.add_argument(
            '--spectrogram-cut-low-frequency',
            type=float,
            default=defaults.spectrogram_config.cut_low_frequency,
        )
        parser.add_argument(
            '--spectrogram-cut-high-frequency',
            type=float,
            default=defaults.spectrogram_config.cut_high_frequency,
        )
        parser.add_argument(
            '--spectrogram-wlen',
            type=int,
            default=defaults.spectrogram_config.wlen,
        )
        parser.add_argument(
            '--spectrogram-nfft',
            type=int,
            default=defaults.spectrogram_config.nfft,
        )
        parser.add_argument(
            '--spectrogram-sliding-window',
            type=float,
            default=defaults.spectrogram_config.sliding_window,
        )
        parser.add_argument(
            '--spectrogram-target-fs',
            type=parse_optional_int,
            default=defaults.spectrogram_config.target_fs,
            help='Target sampling rate for audio input; use "none" to disable resampling.',
        )
        parser.add_argument(
            '--lr-scheduler-factor',
            type=float,
            default=defaults.lr_scheduler_factor,
        )
        parser.add_argument(
            '--lr-scheduler-patience',
            type=int,
            default=defaults.lr_scheduler_patience,
        )
        parser.add_argument(
            '--min-learning-rate',
            type=float,
            default=defaults.min_learning_rate,
        )
        parser.add_argument(
            '--spectrogram-cache-dir',
            type=Path,
            default=defaults.spectrogram_cache_dir,
        )
        parser.add_argument('--checkpoint-path', default=defaults.checkpoint_path)
        args = parser.parse_args(argv)
        values = vars(args)
        img_size = values['img_size']
        values['spectrogram_config'] = SpectrogramConfig(
            image_size=(img_size, img_size),
            cut_low_frequency=values.pop('spectrogram_cut_low_frequency'),
            cut_high_frequency=values.pop('spectrogram_cut_high_frequency'),
            wlen=values.pop('spectrogram_wlen'),
            nfft=values.pop('spectrogram_nfft'),
            sliding_window=values.pop('spectrogram_sliding_window'),
            target_fs=values.pop('spectrogram_target_fs'),
        )
        return cls(**values)

    @property
    def required_splits(self) -> tuple[str, str]:
        return self.train_split, self.validation_split

    @property
    def preferred_split_order(self) -> tuple[str, str, str]:
        return self.train_split, self.validation_split, self.test_split

    @property
    def best_model_path(self) -> str:
        return os.path.join(self.models_dir, 'model_best.pt')

    @property
    def eval_checkpoint_path(self) -> str:
        return self.checkpoint_path or self.best_model_path

    def spectrogram_cache_active(self) -> bool:
        return bool(
            self.train_input_source == 'audio'
            and self.spectrogram_cache_enabled
        )

    def validate(self) -> None:
        if self.train_input_source not in input_source_choices():
            raise ValueError(
                f'Unsupported TRAIN_INPUT_SOURCE={self.train_input_source!r}. '
                "Expected 'audio' or 'spectrogram'."
            )
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError('normalization_mean and normalization_std must be RGB tuples.')
        if any(value <= 0 for value in self.normalization_std):
            raise ValueError('normalization_std values must be positive.')
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive.')
        if self.num_epochs <= 0:
            raise ValueError('num_epochs must be positive.')
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive.')
        if self.patience < 0:
            raise ValueError('patience must be non-negative.')
        if self.lr_scheduler_factor <= 0 or self.lr_scheduler_factor >= 1:
            raise ValueError('lr_scheduler_factor must be between 0 and 1.')
        if self.lr_scheduler_patience < 0:
            raise ValueError('lr_scheduler_patience must be non-negative.')
        if self.min_learning_rate <= 0:
            raise ValueError('min_learning_rate must be positive.')
        if self.spectrogram_config.image_size != (self.img_size, self.img_size):
            raise ValueError('spectrogram_config.image_size must match img_size.')
        if self.spectrogram_config.cut_low_frequency < 0:
            raise ValueError('spectrogram_cut_low_frequency must be non-negative.')
        if (
            self.spectrogram_config.cut_high_frequency
            <= self.spectrogram_config.cut_low_frequency
        ):
            raise ValueError(
                'spectrogram_cut_high_frequency must be greater than '
                'spectrogram_cut_low_frequency.'
            )
        if self.spectrogram_config.wlen <= 0:
            raise ValueError('spectrogram_wlen must be positive.')
        if self.spectrogram_config.nfft < self.spectrogram_config.wlen:
            raise ValueError('spectrogram_nfft must be greater than or equal to wlen.')
        if self.spectrogram_config.sliding_window <= 0:
            raise ValueError('spectrogram_sliding_window must be positive.')
        if (
            self.spectrogram_config.target_fs is not None
            and self.spectrogram_config.target_fs <= 0
        ):
            raise ValueError('spectrogram_target_fs must be positive or None.')

    def ensure_output_dirs(self) -> None:
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def normalization_metadata(self) -> dict[str, object]:
        default_normalization = torchvision_image_normalization()
        uses_default = (
            self.normalization_mean == default_normalization.mean
            and self.normalization_std == default_normalization.std
        )
        source = (
            default_normalization.source
            if uses_default
            else 'custom normalization provided by CLI or environment'
        )
        return {
            'mean': list(self.normalization_mean),
            'std': list(self.normalization_std),
            'source': source,
        }

    def wandb_config(self, device: torch.device) -> dict[str, object]:
        payload = asdict(self)
        payload['spectrogram_cache_dir'] = str(self.spectrogram_cache_dir)
        payload['checkpoint_path'] = self.checkpoint_path
        payload['device'] = str(device)
        payload['amp_enabled'] = self.use_amp and device.type == 'cuda'
        payload['normalization'] = self.normalization_metadata()
        payload['spectrogram_config'] = self.spectrogram_config.to_metadata()
        payload['checkpoint_metric'] = 'val_loss'
        return payload

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def extract_session_id(recording_name: str) -> str:
    """Strip _channel_N so simultaneous multi-channel recordings share one ID."""
    return re.sub(r'_channel_\d+$', '', str(recording_name).strip())

def configure_torch_matmul() -> None:
    try:
        torch.set_float32_matmul_precision('high')
    except (AttributeError, RuntimeError):
        pass

def media_column_name(input_source: str) -> str:
    return 'audio' if input_source == 'audio' else 'spectrogram'

def required_dataset_columns(config: TrainConfig) -> list[str]:
    return [media_column_name(config.train_input_source), 'label', 'recording']

def ordered_split_names(
    split_names: list[str] | tuple[str, ...] | set[str],
    config: TrainConfig,
) -> list[str]:
    split_name_set = set(split_names)
    ordered = [
        split_name
        for split_name in config.preferred_split_order
        if split_name in split_name_set
    ]
    extras = sorted(split_name_set - set(ordered))
    return ordered + extras

def prepare_split_dataset(
    split_dataset: HFDataset,
    split_name: str,
    config: TrainConfig,
) -> HFDataset:
    selected_media_column = media_column_name(config.train_input_source)
    missing_columns = {selected_media_column, 'label'} - set(split_dataset.column_names)
    if missing_columns:
        raise ValueError(
            f'Split {split_name} is missing required columns: {sorted(missing_columns)}'
        )

    if selected_media_column == 'audio':
        split_dataset = split_dataset.cast_column('audio', HFAudio(decode=False))
        if 'spectrogram' in split_dataset.column_names:
            split_dataset = split_dataset.remove_columns(['spectrogram'])
    else:
        split_dataset = split_dataset.cast_column('spectrogram', HFImage(decode=False))
        if 'audio' in split_dataset.column_names:
            split_dataset = split_dataset.remove_columns(['audio'])

    return split_dataset

def load_split_dataset(
    repo_id: str,
    split_name: str,
    config: TrainConfig,
) -> HFDataset:
    columns = required_dataset_columns(config)
    try:
        split_dataset = load_dataset(repo_id, split=split_name, columns=columns)
        print(f'Loaded split {split_name!r} with projected columns: {columns}')
    except Exception as exc:
        print(
            f'Projected load failed for split {split_name!r} ({exc}). '
            'Falling back to the full split load.'
        )
        split_dataset = load_dataset(repo_id, split=split_name)
    return split_dataset

def load_hf_splits(repo_id: str, config: TrainConfig) -> DatasetDict:
    available_splits = set(get_dataset_split_names(repo_id))
    missing_required = set(config.required_splits) - available_splits
    if missing_required:
        raise ValueError(
            f'Dataset source {repo_id!r} is missing required splits: '
            f'{sorted(missing_required)}'
        )

    prepared_splits: dict[str, HFDataset] = {}
    for split_name in ordered_split_names(available_splits, config):
        split_dataset = load_split_dataset(repo_id, split_name, config)
        prepared_splits[split_name] = prepare_split_dataset(split_dataset, split_name, config)

    if config.test_split not in prepared_splits:
        print(
            f'Optional split {config.test_split!r} not found in Hugging Face dataset; '
            'test evaluation skipped.'
        )

    return DatasetDict(prepared_splits)

def load_local_splits(dataset_dir: str, config: TrainConfig) -> DatasetDict:
    loaded = load_from_disk(dataset_dir)
    available_splits = set(loaded.keys())
    missing_required = set(config.required_splits) - available_splits
    if missing_required:
        raise ValueError(
            f'Local dataset source {dataset_dir!r} is missing required splits: '
            f'{sorted(missing_required)}'
        )

    prepared_splits: dict[str, HFDataset] = {}
    for split_name in ordered_split_names(available_splits, config):
        prepared_splits[split_name] = prepare_split_dataset(
            loaded[split_name],
            split_name,
            config,
        )

    if config.test_split not in prepared_splits:
        print(
            f'Optional split {config.test_split!r} not found in local dataset; '
            'test evaluation skipped.'
        )

    return DatasetDict(prepared_splits)

def load_dataset_source(config: TrainConfig) -> DatasetDict:
    source = config.dataset_source
    source_path = Path(source)
    if source_path.exists():
        print(f'Loading local dataset from disk: {source_path}')
        return load_local_splits(str(source_path), config)

    print(f'Loading Hugging Face dataset: {source}')
    return load_hf_splits(source, config)

def split_session_ids(split_dataset: HFDataset) -> set[str] | None:
    if 'recording' not in split_dataset.column_names:
        return None
    return {extract_session_id(recording) for recording in split_dataset['recording']}

def print_split_summary(dataset_dict: DatasetDict, config: TrainConfig) -> None:
    print('\nSplit summary:')
    for split_name in ordered_split_names(dataset_dict.keys(), config):
        split_dataset = dataset_dict[split_name]
        labels = np.asarray(split_dataset['label'], dtype=np.int64)
        whistles = int(labels.sum())
        noise = int(len(labels) - whistles)
        session_ids = split_session_ids(split_dataset)
        sessions_str = f'  sessions={len(session_ids):6d}' if session_ids is not None else ''
        print(
            f'  {split_name:10s} rows={split_dataset.num_rows:6d}'
            f'{sessions_str}  whistles={whistles:6d}  noise={noise:6d}'
        )

def build_split_summary_payload(
    dataset_dict: DatasetDict,
    config: TrainConfig,
) -> dict[str, dict[str, int]]:
    payload = {}
    for split_name in ordered_split_names(dataset_dict.keys(), config):
        split_dataset = dataset_dict[split_name]
        labels = np.asarray(split_dataset['label'], dtype=np.int64)
        sessions = split_session_ids(split_dataset)
        payload[split_name] = {
            'rows': int(split_dataset.num_rows),
            'whistles': int(labels.sum()),
            'noise': int(len(labels) - labels.sum()),
            'sessions': int(len(sessions)) if sessions is not None else -1,
        }
    return payload

def assert_no_session_leakage(dataset_dict: DatasetDict, config: TrainConfig) -> None:
    split_sessions = {
        split_name: split_session_ids(split_dataset)
        for split_name, split_dataset in dataset_dict.items()
    }
    if any(session_ids is None for session_ids in split_sessions.values()):
        print('Session leakage check skipped: `recording` column is missing.')
        return

    available_splits = ordered_split_names(split_sessions.keys(), config)
    for left_name, right_name in combinations(available_splits, 2):
        overlap = split_sessions[left_name] & split_sessions[right_name]
        if overlap:
            raise RuntimeError(
                f'Session leakage detected between {left_name} and {right_name}: '
                f'{sorted(overlap)[:5]}'
            )

    print(
        'Session leakage check passed: '
        f'{", ".join(available_splits)} use disjoint sessions.'
    )

def decode_spectrogram_payload_to_uint8(
    image_payload: object,
    spectrogram_config: SpectrogramConfig,
) -> np.ndarray:
    if isinstance(image_payload, dict):
        image_bytes = image_payload.get('bytes')
        image_path = image_payload.get('path')
        if image_bytes is not None:
            with PILImage.open(BytesIO(image_bytes)) as image:
                array = np.asarray(image.convert('RGB'), dtype=np.uint8)
        elif image_path:
            with PILImage.open(image_path) as image:
                array = np.asarray(image.convert('RGB'), dtype=np.uint8)
        else:
            raise ValueError('Unsupported HF image payload: missing both bytes and path.')
    elif isinstance(image_payload, np.ndarray):
        array = np.asarray(image_payload, dtype=np.uint8)
    elif hasattr(image_payload, 'convert'):
        array = np.asarray(image_payload.convert('RGB'), dtype=np.uint8)
    else:
        raise TypeError(f'Unsupported spectrogram payload type: {type(image_payload)!r}')

    if array.ndim == 2:
        array = np.stack([array, array, array], axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]

    target_height, target_width = spectrogram_config.image_size
    if array.shape[:2] != (target_height, target_width):
        array = cv2.resize(array, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    return np.ascontiguousarray(array, dtype=np.uint8)

class LocalSpectrogramCache:
    def __init__(
        self,
        cache_root: Path,
        repo_id: str,
        spectrogram_config: SpectrogramConfig,
        enabled: bool,
    ) -> None:
        self.enabled = enabled
        self.cache_root = cache_root
        self.repo_id = repo_id
        self.spectrogram_config = spectrogram_config
        self.namespace_dir: Path | None = None

        if not self.enabled:
            return

        repo_slug = repo_id.replace('/', '__')
        config_payload = {
            'repo_id': repo_id,
            'spectrogram_config': spectrogram_config.to_metadata(),
            'image_size': list(spectrogram_config.image_size),
        }
        config_hash = hashlib.sha1(
            json.dumps(config_payload, sort_keys=True).encode('utf-8')
        ).hexdigest()[:12]
        self.namespace_dir = cache_root / repo_slug / config_hash
        self.namespace_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.namespace_dir / 'cache_metadata.json'
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(config_payload, indent=2),
                encoding='utf-8',
            )

    def _path(self, split_name: str, index: int) -> Path | None:
        if self.namespace_dir is None:
            return None
        return self.namespace_dir / split_name / f'{int(index):06d}.npy'

    def count_cached_files(self, split_name: str) -> int:
        if self.namespace_dir is None:
            return 0
        split_dir = self.namespace_dir / split_name
        if not split_dir.exists():
            return 0
        return sum(1 for _ in split_dir.glob('*.npy'))

    def load(self, split_name: str, index: int) -> np.ndarray | None:
        path = self._path(split_name, index)
        if path is None or not path.exists():
            return None

        try:
            gray = np.load(path, allow_pickle=False)
        except Exception:
            return None

        if gray.ndim != 2:
            return None
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    def store(self, split_name: str, index: int, image_uint8: np.ndarray) -> None:
        path = self._path(split_name, index)
        if path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        gray = image_uint8[:, :, 0] if image_uint8.ndim == 3 else image_uint8
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        tmp_path = path.with_suffix(f'.{os.getpid()}.tmp.npy')
        np.save(tmp_path, gray, allow_pickle=False)
        os.replace(tmp_path, path)

class TrainingDataset(TorchDataset):
    def __init__(
        self,
        dataset: HFDataset,
        split_name: str,
        input_source: str,
        spectrogram_config: SpectrogramConfig,
        normalization_mean: tuple[float, float, float],
        normalization_std: tuple[float, float, float],
        spectrogram_cache: LocalSpectrogramCache | None = None,
    ) -> None:
        self.dataset = dataset
        self.split_name = split_name
        self.input_source = input_source
        self.spectrogram_config = spectrogram_config
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.spectrogram_cache = spectrogram_cache

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataset[int(index)]
        if self.input_source == 'audio':
            image_uint8 = None
            if self.spectrogram_cache is not None:
                image_uint8 = self.spectrogram_cache.load(self.split_name, int(index))
            if image_uint8 is None:
                image_uint8 = audio_payload_to_spectrogram_image(
                    row['audio'],
                    config=self.spectrogram_config,
                )
                if self.spectrogram_cache is not None:
                    self.spectrogram_cache.store(self.split_name, int(index), image_uint8)
        else:
            image_uint8 = decode_spectrogram_payload_to_uint8(
                row['spectrogram'],
                self.spectrogram_config,
            )
        image = normalize_uint8_image_to_tensor(
            image_uint8,
            mean=self.normalization_mean,
            std=self.normalization_std,
        )
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return image, label

def build_model(config: TrainConfig, device: torch.device) -> nn.Module:
    return build_vgg16_whistle_classifier(
        pretrained_backbone=config.pretrained_backbone,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

def build_optimizer(model: nn.Module, config: TrainConfig) -> Adam:
    return Adam(model.parameters(), lr=config.learning_rate)

def build_scheduler(optimizer: Adam, config: TrainConfig) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )

def amp_enabled(config: TrainConfig, device: torch.device) -> bool:
    return bool(config.use_amp and device.type == 'cuda')

def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {'total': int(total), 'trainable': int(trainable)}

def init_wandb_run(config: TrainConfig, device: torch.device) -> object | None:
    if not config.wandb_enabled:
        print('Weights & Biases logging disabled.')
        return None
    if wandb is None:
        print('Weights & Biases not installed, skipping wandb logging.')
        return None

    init_kwargs = {
        'project': config.wandb_project,
        'config': config.wandb_config(device),
        'mode': config.wandb_mode,
        'tags': ['torch', 'cnn', 'dolphin', 'whistle'],
        'settings': wandb.Settings(console=config.wandb_console),
    }
    if config.wandb_entity:
        init_kwargs['entity'] = config.wandb_entity
    if config.wandb_run_name:
        init_kwargs['name'] = config.wandb_run_name

    try:
        run = wandb.init(**init_kwargs)
        if run is not None:
            run.define_metric('epoch')
            run.define_metric('train/*', step_metric='epoch')
            run.define_metric('validation/*', step_metric='epoch')
            run.define_metric('test/*', step_metric='epoch')
        return run
    except Exception as exc:
        print(f'Warning: unable to initialize wandb ({exc}). Continuing without wandb.')
        return None

def update_wandb_run_config(
    run: object | None,
    model: nn.Module,
    split_summary: dict[str, dict[str, int]],
) -> None:
    if run is None:
        return

    param_counts = count_parameters(model)
    config_update = {
        'parameter_count_total': param_counts['total'],
        'parameter_count_trainable': param_counts['trainable'],
        'split_summary': split_summary,
    }
    try:
        run.config.update(config_update, allow_val_change=True)
    except Exception as exc:
        print(f'Warning: unable to update wandb config ({exc}).')

def wandb_log(run: object | None, payload: dict[str, object]) -> None:
    if run is None or wandb is None:
        return
    try:
        run.log(payload)
    except Exception as exc:
        print(f'Warning: wandb logging failed ({exc}).')

def wandb_log_artifact_images(run: object | None, config: TrainConfig) -> None:
    if run is None or wandb is None:
        return

    payload = {}
    metrics_path = os.path.join(config.figs_dir, 'metrics_training.png')
    roc_path = os.path.join(config.figs_dir, 'roc_validation_test.png')
    if os.path.exists(metrics_path):
        payload['figures/metrics_training'] = wandb.Image(metrics_path)
    if os.path.exists(roc_path):
        payload['figures/roc_validation_test'] = wandb.Image(roc_path)
    for split_name in (config.validation_split, config.test_split):
        confusion_path = os.path.join(config.figs_dir, f'{split_name}_confusion_matrix.png')
        if os.path.exists(confusion_path):
            payload[f'figures/{split_name}_confusion_matrix'] = wandb.Image(confusion_path)
    if payload:
        wandb_log(run, payload)

def wandb_log_table(
    run: object | None,
    key: str,
    rows: list[dict[str, object]],
) -> None:
    if run is None or wandb is None or not rows:
        return

    columns = list(rows[0].keys())
    data = [[row[column] for column in columns] for row in rows]
    try:
        run.log({key: wandb.Table(columns=columns, data=data)})
    except Exception as exc:
        print(f'Warning: unable to log wandb table {key} ({exc}).')

def compute_classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if labels.size == 0:
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'positive_prediction_rate': 0.0,
        }

    return {
        'accuracy': float(np.mean(labels == preds)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'positive_prediction_rate': float(np.mean(preds == 1)),
    }

@dataclass(frozen=True)
class EpochMetric:
    name: str
    display_label: str

TRACKED_EPOCH_METRICS = (
    EpochMetric('loss', 'loss'),
    EpochMetric('accuracy', 'acc'),
    EpochMetric('f1', 'f1'),
    EpochMetric('precision', 'prec'),
    EpochMetric('recall', 'rec'),
    EpochMetric('positive_prediction_rate', 'ppr'),
)

def new_history() -> dict[str, list[float]]:
    history: dict[str, list[float]] = {}
    for metric in TRACKED_EPOCH_METRICS:
        history[metric.name] = []
        history[f'val_{metric.name}'] = []
    return history

def metric_value(metrics: dict[str, np.ndarray | float], metric_name: str) -> float:
    return float(metrics[metric_name])

def append_epoch_history(
    history: dict[str, list[float]],
    train_metrics: dict[str, np.ndarray | float],
    validation_metrics: dict[str, np.ndarray | float],
) -> None:
    for metric in TRACKED_EPOCH_METRICS:
        history[metric.name].append(metric_value(train_metrics, metric.name))
        history[f'val_{metric.name}'].append(
            metric_value(validation_metrics, metric.name)
        )

def format_epoch_metrics(
    metrics: dict[str, np.ndarray | float],
    prefix: str = '',
) -> str:
    parts = [
        f'{prefix}{metric.display_label}={metric_value(metrics, metric.name):.4f}'
        for metric in TRACKED_EPOCH_METRICS
    ]
    return '  '.join(parts)

def build_epoch_wandb_payload(
    epoch: int,
    train_metrics: dict[str, np.ndarray | float],
    validation_metrics: dict[str, np.ndarray | float],
    learning_rate: float,
    best_val_loss: float,
) -> dict[str, object]:
    payload: dict[str, object] = {'epoch': epoch}
    for metric in TRACKED_EPOCH_METRICS:
        payload[f'train/{metric.name}'] = metric_value(train_metrics, metric.name)
        payload[f'validation/{metric.name}'] = metric_value(
            validation_metrics,
            metric.name,
        )

    payload['train/learning_rate'] = float(learning_rate)
    payload['validation/best_loss_so_far'] = min(
        best_val_loss,
        metric_value(validation_metrics, 'loss'),
    )
    return payload

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
    device: torch.device,
    pin_memory: bool,
    description: str,
    optimizer: Adam | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, np.ndarray | float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_scores = []
    all_labels = []
    all_preds = []
    use_autocast = amp_enabled(config, device)

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=description, leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            autocast_manager = (
                torch.amp.autocast(device_type='cuda', enabled=use_autocast)
                if device.type == 'cuda'
                else nullcontext()
            )
            with autocast_manager:
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size
            all_scores.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
            progress.set_postfix(
                loss=f'{total_loss / total_samples:.4f}',
                acc=f'{total_correct / total_samples:.4f}',
            )

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()
    metrics = compute_classification_metrics(labels, preds)
    return {
        'loss': total_loss / total_samples,
        **metrics,
        'scores': scores,
        'labels': labels,
        'predictions': preds,
    }

def save_checkpoint(
    path: str,
    model: nn.Module,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
    test_split_name: str | None,
) -> None:
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'img_size': config.img_size,
            'class_to_idx': {'0': 0, '1': 1},
            'normalization': config.normalization_metadata(),
            'model_class': model.__class__.__name__,
            'pretrained_backbone': config.pretrained_backbone,
            'freeze_backbone': config.freeze_backbone,
            'dataset_source': config.dataset_source,
            'train_input_source': config.train_input_source,
            'validation_split': config.validation_split,
            'test_split': test_split_name,
            'spectrogram_config': config.spectrogram_config.to_metadata(),
            'checkpoint_selection_metric': 'val_loss',
        },
        path,
    )

def plot_training_curves(history: dict[str, list[float]], config: TrainConfig) -> None:
    if not history['loss']:
        print('Training curves skipped: no training history available.')
        return

    fig, (ax_loss, ax_acc, ax_f1) = plt.subplots(1, 3, figsize=(18, 4))

    ax_loss.plot(history['loss'], label='train')
    ax_loss.plot(history['val_loss'], label='val')
    ax_loss.set_title('Model loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='upper right')

    ax_acc.plot(history['accuracy'], label='train')
    ax_acc.plot(history['val_accuracy'], label='val')
    ax_acc.set_title('Model accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower right')

    ax_f1.plot(history['f1'], label='train')
    ax_f1.plot(history['val_f1'], label='val')
    ax_f1.set_title('Model F1 score')
    ax_f1.set_xlabel('Epoch')
    ax_f1.set_ylabel('F1')
    ax_f1.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(config.figs_dir, 'metrics_training.png'))
    plt.close()

def maybe_build_roc_curve(
    split_name: str,
    metrics: dict[str, np.ndarray | float],
) -> tuple[str, np.ndarray, np.ndarray, float] | None:
    labels = metrics['labels']
    scores = metrics['scores']
    if np.unique(labels).size < 2:
        print(f'ROC skipped for {split_name}: split contains a single class.')
        return None

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return split_name, fpr, tpr, roc_auc

def plot_roc_curves(
    curves: list[tuple[str, np.ndarray, np.ndarray, float]],
    config: TrainConfig,
) -> None:
    if not curves:
        print('No ROC curve generated because no evaluation split contained both classes.')
        return

    plt.figure(figsize=(8, 6))
    for split_name, fpr, tpr, roc_auc in curves:
        plt.plot(fpr, tpr, lw=2, label=f'{split_name.capitalize()} ROC (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - validation and test')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(config.figs_dir, 'roc_validation_test.png'))
    plt.close()

def save_confusion_matrix_artifacts(
    split_name: str,
    metrics: dict[str, np.ndarray | float],
    config: TrainConfig,
) -> tuple[str, str, dict[str, int]]:
    labels = np.asarray(metrics['labels'], dtype=np.int64)
    preds = np.asarray(metrics['predictions'], dtype=np.int64)
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    counts = {
        'tn': int(matrix[0, 0]),
        'fp': int(matrix[0, 1]),
        'fn': int(matrix[1, 0]),
        'tp': int(matrix[1, 1]),
    }

    csv_path = os.path.join(config.reports_dir, f'{split_name}_confusion_matrix.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['actual\\predicted', 'noise', 'whistle'])
        writer.writerow(['noise', counts['tn'], counts['fp']])
        writer.writerow(['whistle', counts['fn'], counts['tp']])

    fig_path = os.path.join(config.figs_dir, f'{split_name}_confusion_matrix.png')
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=['noise', 'whistle'],
        yticklabels=['noise', 'whistle'],
        ylabel='Actual',
        xlabel='Predicted',
        title=f'{split_name.capitalize()} confusion matrix',
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                f'{value}',
                ha='center',
                va='center',
                color='white' if value > threshold else 'black',
            )
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return csv_path, fig_path, counts

def write_run_summary_json(
    config: TrainConfig,
    best_epoch: int,
    best_val_loss: float,
    best_model_path: str,
    split_summary: dict[str, dict[str, int]],
    validation_metrics: dict[str, np.ndarray | float],
    test_metrics: dict[str, np.ndarray | float] | None,
    confusion_artifacts: dict[str, dict[str, object]],
    validation_session_report_path: str | None,
    test_session_report_path: str | None,
) -> str:
    payload: dict[str, object] = {
        'dataset_source': config.dataset_source,
        'train_input_source': config.train_input_source,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_model_path': best_model_path,
        'split_summary': split_summary,
        'metrics': {
            config.validation_split: {
                'loss': float(validation_metrics['loss']),
                'accuracy': float(validation_metrics['accuracy']),
                'f1': float(validation_metrics['f1']),
                'precision': float(validation_metrics['precision']),
                'recall': float(validation_metrics['recall']),
                'positive_prediction_rate': float(
                    validation_metrics['positive_prediction_rate']
                ),
            },
        },
        'artifacts': {
            'model': best_model_path,
            'figures_dir': config.figs_dir,
            'reports_dir': config.reports_dir,
            'validation_session_report_path': validation_session_report_path,
            'test_session_report_path': test_session_report_path,
            'confusion_matrices': confusion_artifacts,
        },
    }
    if test_metrics is not None:
        payload['metrics'][config.test_split] = {
            'loss': float(test_metrics['loss']),
            'accuracy': float(test_metrics['accuracy']),
            'f1': float(test_metrics['f1']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'positive_prediction_rate': float(test_metrics['positive_prediction_rate']),
        }

    summary_path = os.path.join(config.reports_dir, 'run_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    return summary_path

def build_session_report_rows(
    split_name: str,
    split_dataset: HFDataset,
    metrics: dict[str, np.ndarray | float],
) -> list[dict[str, object]]:
    if 'recording' not in split_dataset.column_names:
        return []

    recordings = split_dataset['recording']
    labels = np.asarray(metrics['labels'], dtype=np.int64)
    preds = np.asarray(metrics['predictions'], dtype=np.int64)
    scores = np.asarray(metrics['scores'], dtype=np.float64)

    if not (len(recordings) == len(labels) == len(preds) == len(scores)):
        raise RuntimeError(
            f'Per-session report mismatch for {split_name}: '
            f'{len(recordings)=} {len(labels)=} {len(preds)=} {len(scores)=}'
        )

    grouped: dict[str, dict[str, object]] = {}
    for recording, label, pred, score in zip(recordings, labels, preds, scores):
        session_id = extract_session_id(recording)
        stats = grouped.setdefault(
            session_id,
            {
                'split': split_name,
                'session_id': session_id,
                'rows': 0,
                'positives': 0,
                'negatives': 0,
                'predicted_positives': 0,
                'predicted_negatives': 0,
                '_labels': [],
                '_preds': [],
                '_scores': [],
            },
        )
        stats['rows'] += 1
        stats['positives'] += int(label)
        stats['negatives'] += int(1 - label)
        stats['predicted_positives'] += int(pred)
        stats['predicted_negatives'] += int(1 - pred)
        stats['_labels'].append(int(label))
        stats['_preds'].append(int(pred))
        stats['_scores'].append(float(score))

    rows = []
    for session_id, stats in grouped.items():
        session_labels = np.asarray(stats.pop('_labels'), dtype=np.int64)
        session_preds = np.asarray(stats.pop('_preds'), dtype=np.int64)
        session_scores = np.asarray(stats.pop('_scores'), dtype=np.float64)
        session_metrics = compute_classification_metrics(session_labels, session_preds)
        rows.append(
            {
                **stats,
                **session_metrics,
                'mean_positive_score': float(np.mean(session_scores)),
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row['rows']),
            str(row['session_id']),
        )
    )
    return rows

def write_session_report_csv(
    split_name: str,
    rows: list[dict[str, object]],
    config: TrainConfig,
) -> str | None:
    if not rows:
        return None

    path = os.path.join(config.reports_dir, f'{split_name}_session_metrics.csv')
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

def print_session_report_preview(
    split_name: str,
    rows: list[dict[str, object]],
    path: str | None,
    limit: int = 5,
) -> None:
    if not rows:
        return

    if path is not None:
        print(f'  {split_name.capitalize()} session report: {path}')
    preview = rows[:limit]
    for row in preview:
        print(
            f"    {row['session_id']}  rows={row['rows']}  "
            f"acc={row['accuracy']:.4f}  f1={row['f1']:.4f}  "
            f"prec={row['precision']:.4f}  rec={row['recall']:.4f}  "
            f"ppr={row['positive_prediction_rate']:.4f}"
        )

@dataclass
class PreparedData:
    split_datasets: DatasetDict
    has_test_split: bool
    split_summary: dict[str, dict[str, int]]
    spectrogram_cache: LocalSpectrogramCache
    train_dataset: TrainingDataset
    valid_dataset: TrainingDataset
    test_dataset: TrainingDataset | None
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader | None

def print_spectrogram_cache_coverage(
    split_datasets: DatasetDict,
    spectrogram_cache: LocalSpectrogramCache,
    config: TrainConfig,
) -> None:
    print('Spectrogram cache coverage:')
    all_cached = True
    for split_name in ordered_split_names(split_datasets.keys(), config):
        expected = int(split_datasets[split_name].num_rows)
        cached = spectrogram_cache.count_cached_files(split_name)
        if cached != expected:
            all_cached = False
        print(f'  {split_name:10s} cached={cached:6d}/{expected:6d}')
    if all_cached:
        print('Spectrogram cache is warm: all examples are already available locally.')
    else:
        print('Spectrogram cache is partial: missing examples will be generated on demand.')

def make_training_dataset(
    split_datasets: DatasetDict,
    split_name: str,
    config: TrainConfig,
    spectrogram_cache: LocalSpectrogramCache,
) -> TrainingDataset:
    return TrainingDataset(
        split_datasets[split_name],
        split_name,
        config.train_input_source,
        config.spectrogram_config,
        config.normalization_mean,
        config.normalization_std,
        spectrogram_cache=spectrogram_cache,
    )

def make_data_loader(
    dataset: TorchDataset,
    shuffle: bool,
    config: TrainConfig,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

def prepare_data(config: TrainConfig, pin_memory: bool) -> PreparedData:
    split_datasets = load_dataset_source(config)
    has_test_split = config.test_split in split_datasets
    print_split_summary(split_datasets, config)
    assert_no_session_leakage(split_datasets, config)
    split_summary = build_split_summary_payload(split_datasets, config)
    spectrogram_cache = LocalSpectrogramCache(
        cache_root=config.spectrogram_cache_dir,
        repo_id=config.dataset_source,
        spectrogram_config=config.spectrogram_config,
        enabled=config.spectrogram_cache_active(),
    )

    if config.spectrogram_cache_active():
        print_spectrogram_cache_coverage(split_datasets, spectrogram_cache, config)

    train_dataset = make_training_dataset(
        split_datasets,
        config.train_split,
        config,
        spectrogram_cache,
    )
    valid_dataset = make_training_dataset(
        split_datasets,
        config.validation_split,
        config,
        spectrogram_cache,
    )
    test_dataset = (
        make_training_dataset(split_datasets, config.test_split, config, spectrogram_cache)
        if has_test_split
        else None
    )

    return PreparedData(
        split_datasets=split_datasets,
        has_test_split=has_test_split,
        split_summary=split_summary,
        spectrogram_cache=spectrogram_cache,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        train_loader=make_data_loader(
            train_dataset,
            shuffle=True,
            config=config,
            pin_memory=pin_memory,
        ),
        valid_loader=make_data_loader(
            valid_dataset,
            shuffle=False,
            config=config,
            pin_memory=pin_memory,
        ),
        test_loader=(
            make_data_loader(test_dataset, shuffle=False, config=config, pin_memory=pin_memory)
            if test_dataset is not None
            else None
        ),
    )

class TorchTrainingRun:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.config.validate()
        self.config.ensure_output_dirs()
        seed_everything(self.config.random_state)
        configure_torch_matmul()

        self.device = get_device()
        self.pin_memory = self.device.type == 'cuda'
        self.criterion = nn.CrossEntropyLoss()
        self.wandb_run: object | None = None
        self.prepared_data: PreparedData | None = None
        self.net: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler: ReduceLROnPlateau | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.history = new_history()
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.resolved_model_artifact_path = self.config.best_model_path

    def print_startup_summary(self) -> None:
        print(f'Using device: {self.device}')
        print(
            f'Training config: input_source={self.config.train_input_source}  '
            f'batch_size={self.config.batch_size}  '
            f'num_workers={self.config.num_workers}  '
            f'amp={"on" if amp_enabled(self.config, self.device) else "off"}'
        )
        if self.config.eval_only:
            print('Evaluation-only mode: enabled')
        if self.config.spectrogram_cache_active():
            print(
                'Local spectrogram cache: enabled  '
                f'dir={self.config.spectrogram_cache_dir}'
            )
        else:
            print('Local spectrogram cache: disabled')

    def setup(self) -> None:
        self.print_startup_summary()
        self.wandb_run = init_wandb_run(self.config, self.device)
        self.prepared_data = prepare_data(self.config, self.pin_memory)
        self.net = build_model(self.config, self.device)
        update_wandb_run_config(
            self.wandb_run,
            self.net,
            self.prepared_data.split_summary,
        )
        self.optimizer = build_optimizer(self.net, self.config)
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=amp_enabled(self.config, self.device),
        )
        self.scheduler = build_scheduler(self.optimizer, self.config)

    def run_epoch(
        self,
        loader: DataLoader,
        description: str,
        optimizer: Adam | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> dict[str, np.ndarray | float]:
        if self.net is None:
            raise RuntimeError('Training run has not been set up.')
        return run_epoch(
            self.net,
            loader,
            self.criterion,
            self.config,
            self.device,
            self.pin_memory,
            description,
            optimizer=optimizer,
            scaler=scaler,
        )

    def load_eval_checkpoint(self) -> None:
        if self.net is None:
            raise RuntimeError('Training run has not been set up.')
        checkpoint_path = self.config.eval_checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                'Evaluation-only mode requested but checkpoint does not exist: '
                f'{checkpoint_path}'
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.best_epoch = int(checkpoint.get('epoch', 0))
        self.best_val_loss = float(checkpoint.get('val_loss', np.inf))
        self.resolved_model_artifact_path = checkpoint_path
        print(
            f'Loaded checkpoint for evaluation: {checkpoint_path}  '
            f'(epoch={self.best_epoch}, val_loss={self.best_val_loss:.4f})'
        )

    def train(self) -> None:
        if self.prepared_data is None or self.net is None:
            raise RuntimeError('Training run has not been set up.')
        if self.optimizer is None or self.scheduler is None or self.scaler is None:
            raise RuntimeError('Optimizer, scheduler, or scaler missing.')

        epochs_without_improvement = 0
        try:
            for epoch in range(1, self.config.num_epochs + 1):
                train_metrics = self.run_epoch(
                    self.prepared_data.train_loader,
                    description=f'train epoch {epoch}/{self.config.num_epochs}',
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                )
                val_metrics = self.run_epoch(
                    self.prepared_data.valid_loader,
                    description=f'validation epoch {epoch}/{self.config.num_epochs}',
                )
                self.scheduler.step(float(val_metrics['loss']))

                append_epoch_history(self.history, train_metrics, val_metrics)
                current_lr = float(self.optimizer.param_groups[0]['lr'])
                print(
                    f'Epoch {epoch:02d}/{self.config.num_epochs}  '
                    f'{format_epoch_metrics(train_metrics)}  '
                    f'{format_epoch_metrics(val_metrics, prefix="val_")}  '
                    f'lr={current_lr:.2e}'
                )
                wandb_log(
                    self.wandb_run,
                    build_epoch_wandb_payload(
                        epoch,
                        train_metrics,
                        val_metrics,
                        current_lr,
                        self.best_val_loss,
                    ),
                )

                val_loss = float(val_metrics['loss'])
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.best_state = copy.deepcopy(self.net.state_dict())
                    save_checkpoint(
                        self.config.best_model_path,
                        self.net,
                        self.config,
                        epoch,
                        self.best_val_loss,
                        (
                            self.config.test_split
                            if self.prepared_data.has_test_split
                            else None
                        ),
                    )
                    print(
                        '  -> Best checkpoint saved  '
                        f'(val_loss = {self.best_val_loss:.4f})'
                    )
                    wandb_log(
                        self.wandb_run,
                        {
                            'epoch': epoch,
                            'validation/checkpoint_saved': 1,
                            'validation/best_loss': float(self.best_val_loss),
                            'validation/best_epoch': int(self.best_epoch),
                        },
                    )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.config.patience:
                    print(f'  -> Early stopping triggered after {epoch} epochs.')
                    break
        except RuntimeError as exc:
            message = str(exc).lower()
            if 'out of memory' in message:
                print(
                    'CUDA OOM during training. Try lowering --batch-size, keep '
                    '--use-amp enabled, and close other GPU-heavy apps.'
                )
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            raise

        if self.best_state is not None:
            self.net.load_state_dict(self.best_state)

    def evaluate_and_write_artifacts(self) -> None:
        if self.prepared_data is None:
            raise RuntimeError('Training run has not been set up.')

        plot_training_curves(self.history, self.config)
        validation_metrics = self.run_epoch(
            self.prepared_data.valid_loader,
            description='validation final',
        )
        test_metrics = (
            self.run_epoch(
                self.prepared_data.test_loader,
                description='test final',
            )
            if self.prepared_data.test_loader is not None
            else None
        )
        validation_session_rows = build_session_report_rows(
            self.config.validation_split,
            self.prepared_data.split_datasets[self.config.validation_split],
            validation_metrics,
        )
        test_session_rows = (
            build_session_report_rows(
                self.config.test_split,
                self.prepared_data.split_datasets[self.config.test_split],
                test_metrics,
            )
            if test_metrics is not None
            else []
        )
        validation_session_report_path = write_session_report_csv(
            self.config.validation_split,
            validation_session_rows,
            self.config,
        )
        test_session_report_path = (
            write_session_report_csv(
                self.config.test_split,
                test_session_rows,
                self.config,
            )
            if test_session_rows
            else None
        )

        validation_confusion = save_confusion_matrix_artifacts(
            self.config.validation_split,
            validation_metrics,
            self.config,
        )
        (
            validation_confusion_csv_path,
            validation_confusion_fig_path,
            validation_confusion_counts,
        ) = validation_confusion
        test_confusion_csv_path = None
        test_confusion_fig_path = None
        test_confusion_counts = None
        if test_metrics is not None:
            (
                test_confusion_csv_path,
                test_confusion_fig_path,
                test_confusion_counts,
            ) = save_confusion_matrix_artifacts(
                self.config.test_split,
                test_metrics,
                self.config,
            )

        roc_curves = []
        validation_roc = maybe_build_roc_curve(
            self.config.validation_split,
            validation_metrics,
        )
        if validation_roc is not None:
            roc_curves.append(validation_roc)
        if test_metrics is not None:
            test_roc = maybe_build_roc_curve(self.config.test_split, test_metrics)
            if test_roc is not None:
                roc_curves.append(test_roc)
        plot_roc_curves(roc_curves, self.config)
        wandb_log_artifact_images(self.wandb_run, self.config)

        confusion_artifacts: dict[str, dict[str, object]] = {
            self.config.validation_split: {
                'csv_path': validation_confusion_csv_path,
                'figure_path': validation_confusion_fig_path,
                'counts': validation_confusion_counts,
            }
        }
        if test_confusion_counts is not None:
            confusion_artifacts[self.config.test_split] = {
                'csv_path': test_confusion_csv_path,
                'figure_path': test_confusion_fig_path,
                'counts': test_confusion_counts,
            }

        run_summary_path = write_run_summary_json(
            config=self.config,
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
            best_model_path=self.resolved_model_artifact_path,
            split_summary=self.prepared_data.split_summary,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            confusion_artifacts=confusion_artifacts,
            validation_session_report_path=validation_session_report_path,
            test_session_report_path=test_session_report_path,
        )

        self.print_final_summary(
            validation_metrics,
            test_metrics,
            validation_session_rows,
            test_session_rows,
            validation_session_report_path,
            test_session_report_path,
            run_summary_path,
        )
        self.log_final_wandb(
            validation_metrics,
            test_metrics,
            validation_session_rows,
            test_session_rows,
            validation_session_report_path,
            test_session_report_path,
            run_summary_path,
            validation_confusion_csv_path,
            validation_confusion_fig_path,
            test_confusion_csv_path,
            test_confusion_fig_path,
        )

    def print_final_summary(
        self,
        validation_metrics: dict[str, np.ndarray | float],
        test_metrics: dict[str, np.ndarray | float] | None,
        validation_session_rows: list[dict[str, object]],
        test_session_rows: list[dict[str, object]],
        validation_session_report_path: str | None,
        test_session_report_path: str | None,
        run_summary_path: str,
    ) -> None:
        print('\nFinal summary:')
        print(
            f'  Validation  loss={validation_metrics["loss"]:.4f}  '
            f'acc={validation_metrics["accuracy"]:.4f}  '
            f'f1={validation_metrics["f1"]:.4f}  '
            f'precision={validation_metrics["precision"]:.4f}  '
            f'recall={validation_metrics["recall"]:.4f}  '
            f'ppr={validation_metrics["positive_prediction_rate"]:.4f}'
        )
        if test_metrics is not None:
            print(
                f'  Test        loss={test_metrics["loss"]:.4f}  '
                f'acc={test_metrics["accuracy"]:.4f}  '
                f'f1={test_metrics["f1"]:.4f}  '
                f'precision={test_metrics["precision"]:.4f}  '
                f'recall={test_metrics["recall"]:.4f}  '
                f'ppr={test_metrics["positive_prediction_rate"]:.4f}'
            )
        else:
            print('  Test        skipped (no test split in dataset source)')
        print(f'  Best epoch: {self.best_epoch}')
        print(f'  Best model saved to: {self.resolved_model_artifact_path}')
        print(f'  Run summary: {run_summary_path}')
        print('  Session-level preview:')
        print_session_report_preview(
            self.config.validation_split,
            validation_session_rows,
            validation_session_report_path,
        )
        if test_session_rows:
            print_session_report_preview(
                self.config.test_split,
                test_session_rows,
                test_session_report_path,
            )

    def log_final_wandb(
        self,
        validation_metrics: dict[str, np.ndarray | float],
        test_metrics: dict[str, np.ndarray | float] | None,
        validation_session_rows: list[dict[str, object]],
        test_session_rows: list[dict[str, object]],
        validation_session_report_path: str | None,
        test_session_report_path: str | None,
        run_summary_path: str,
        validation_confusion_csv_path: str,
        validation_confusion_fig_path: str,
        test_confusion_csv_path: str | None,
        test_confusion_fig_path: str | None,
    ) -> None:
        final_wandb_payload = {
            'epoch': int(self.best_epoch),
            'validation/final_loss': float(validation_metrics['loss']),
            'validation/final_accuracy': float(validation_metrics['accuracy']),
            'validation/final_f1': float(validation_metrics['f1']),
            'validation/final_precision': float(validation_metrics['precision']),
            'validation/final_recall': float(validation_metrics['recall']),
            'validation/final_positive_prediction_rate': float(
                validation_metrics['positive_prediction_rate']
            ),
            'training/best_val_loss': float(self.best_val_loss),
            'training/best_epoch': int(self.best_epoch),
            'training/best_model_path': self.resolved_model_artifact_path,
        }
        if test_metrics is not None:
            final_wandb_payload.update(
                {
                    'test/loss': float(test_metrics['loss']),
                    'test/accuracy': float(test_metrics['accuracy']),
                    'test/f1': float(test_metrics['f1']),
                    'test/precision': float(test_metrics['precision']),
                    'test/recall': float(test_metrics['recall']),
                    'test/positive_prediction_rate': float(
                        test_metrics['positive_prediction_rate']
                    ),
                }
            )
        wandb_log(self.wandb_run, final_wandb_payload)
        wandb_log_table(
            self.wandb_run,
            'validation/session_metrics',
            validation_session_rows,
        )
        if test_session_rows:
            wandb_log_table(self.wandb_run, 'test/session_metrics', test_session_rows)
        self.update_wandb_summary(
            validation_metrics,
            test_metrics,
            validation_session_report_path,
            test_session_report_path,
            run_summary_path,
            validation_confusion_csv_path,
            validation_confusion_fig_path,
            test_confusion_csv_path,
            test_confusion_fig_path,
        )

    def update_wandb_summary(
        self,
        validation_metrics: dict[str, np.ndarray | float],
        test_metrics: dict[str, np.ndarray | float] | None,
        validation_session_report_path: str | None,
        test_session_report_path: str | None,
        run_summary_path: str,
        validation_confusion_csv_path: str,
        validation_confusion_fig_path: str,
        test_confusion_csv_path: str | None,
        test_confusion_fig_path: str | None,
    ) -> None:
        if self.wandb_run is None:
            return
        try:
            summary = self.wandb_run.summary
            summary['best_epoch'] = int(self.best_epoch)
            summary['best_val_loss'] = float(self.best_val_loss)
            summary['validation_accuracy'] = float(validation_metrics['accuracy'])
            summary['validation_f1'] = float(validation_metrics['f1'])
            summary['validation_precision'] = float(validation_metrics['precision'])
            summary['validation_recall'] = float(validation_metrics['recall'])
            summary['validation_positive_prediction_rate'] = float(
                validation_metrics['positive_prediction_rate']
            )
            if test_metrics is not None:
                summary['test_accuracy'] = float(test_metrics['accuracy'])
                summary['test_f1'] = float(test_metrics['f1'])
                summary['test_precision'] = float(test_metrics['precision'])
                summary['test_recall'] = float(test_metrics['recall'])
                summary['test_positive_prediction_rate'] = float(
                    test_metrics['positive_prediction_rate']
                )
            if validation_session_report_path is not None:
                summary['validation_session_report_path'] = validation_session_report_path
            if test_session_report_path is not None:
                summary['test_session_report_path'] = test_session_report_path
            summary['run_summary_path'] = run_summary_path
            summary['validation_confusion_csv_path'] = validation_confusion_csv_path
            summary['validation_confusion_fig_path'] = validation_confusion_fig_path
            if test_confusion_csv_path is not None:
                summary['test_confusion_csv_path'] = test_confusion_csv_path
            if test_confusion_fig_path is not None:
                summary['test_confusion_fig_path'] = test_confusion_fig_path
            summary['best_model_path'] = self.resolved_model_artifact_path
        except Exception as exc:
            print(f'Warning: unable to write wandb summary ({exc}).')

    def finish_wandb(self) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.finish()
        except Exception as exc:
            print(f'Warning: unable to finish wandb run ({exc}).')

    def cleanup(self) -> None:
        self.prepared_data = None
        self.net = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.best_state = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def run(self) -> None:
        try:
            self.setup()
            if self.config.eval_only:
                self.load_eval_checkpoint()
            else:
                self.train()
            self.evaluate_and_write_artifacts()
        finally:
            self.finish_wandb()
            self.cleanup()

def main(argv: list[str] | None = None) -> None:
    TorchTrainingRun(TrainConfig.from_args(argv)).run()

if __name__ == '__main__':
    main()
