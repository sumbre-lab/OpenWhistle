import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, get_args

import torch
from huggingface_hub import hf_hub_download

from utils.model import (
    SpectrogramConfig,
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

CNN_RUNS_DIR = Path(__file__).resolve().parents[1] / 'runs'

@dataclass(frozen=True)
class TrainConfig:
    img_size: int = 224
    batch_size: int = 4
    num_epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-5
    random_state: int = 7
    num_workers: int = 0
    dataset_source: str = 'dolphinteam/OpenWhistle-CNN'
    models_dir: str = str(CNN_RUNS_DIR / 'models')
    figs_dir: str = str(CNN_RUNS_DIR / 'figures')
    reports_dir: str = str(CNN_RUNS_DIR / 'reports')
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
    wandb_enabled: bool = False
    wandb_project: str = 'dolphin-whistle-training'
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = 'online'
    wandb_console: str = 'redirect'
    use_amp: bool = True
    cpu_only: bool = False
    eval_only: bool = False
    test_only: bool = False
    spectrogram_cache_enabled: bool = True
    pretrained_backbone: bool = True
    freeze_backbone: bool = False
    spectrogram_config: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    min_learning_rate: float = 1e-6
    spectrogram_cache_dir: Path = CNN_RUNS_DIR / 'spectrogram_cache'
    checkpoint_path: str | None = None
    checkpoint_repo: str = 'dolphinteam/OpenWhistle-CNN-VGG16'
    checkpoint_filename: str = 'model_vgg_final_best.pt'

    def __post_init__(self) -> None:
        for name in ('models_dir', 'figs_dir', 'reports_dir'):
            object.__setattr__(self, name, os.path.expanduser(getattr(self, name)))
        object.__setattr__(self, 'spectrogram_cache_dir', Path(self.spectrogram_cache_dir).expanduser())

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
            cpu_only=env_bool('TRAIN_CPU_ONLY', cls.cpu_only),
            eval_only=env_bool('TRAIN_EVAL_ONLY', cls.eval_only),
            test_only=env_bool('TRAIN_TEST_ONLY', cls.test_only),
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
            checkpoint_repo=env_str('TRAIN_CHECKPOINT_REPO', cls.checkpoint_repo),
            checkpoint_filename=env_str(
                'TRAIN_CHECKPOINT_FILENAME',
                cls.checkpoint_filename,
            ),
        )

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> 'TrainConfig':
        defaults = cls.defaults_from_env()
        parser = argparse.ArgumentParser(
            description='Train and evaluate the OpenWhistle CNN classifier.'
        )
        hidden = argparse.SUPPRESS
        parser.add_argument('--img-size', type=int, default=defaults.img_size, help=hidden)
        parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
        parser.add_argument('--num-epochs', type=int, default=defaults.num_epochs)
        parser.add_argument('--patience', type=int, default=defaults.patience)
        parser.add_argument('--learning-rate', type=float, default=defaults.learning_rate)
        parser.add_argument('--random-state', type=int, default=defaults.random_state)
        parser.add_argument('--num-workers', type=int, default=defaults.num_workers)
        parser.add_argument('--dataset-source', default=defaults.dataset_source)
        parser.add_argument('--models-dir', default=defaults.models_dir, help=hidden)
        parser.add_argument('--figs-dir', default=defaults.figs_dir, help=hidden)
        parser.add_argument('--reports-dir', default=defaults.reports_dir, help=hidden)
        parser.add_argument('--train-split', default=defaults.train_split, help=hidden)
        parser.add_argument(
            '--validation-split',
            default=defaults.validation_split,
            help=hidden,
        )
        parser.add_argument('--test-split', default=defaults.test_split, help=hidden)
        parser.add_argument(
            '--train-input-source',
            choices=input_source_choices(),
            default=defaults.train_input_source,
        )
        parser.add_argument(
            '--normalization-mean',
            type=parse_float_tuple,
            default=defaults.normalization_mean,
            help=hidden,
        )
        parser.add_argument(
            '--normalization-std',
            type=parse_float_tuple,
            default=defaults.normalization_std,
            help=hidden,
        )
        add_bool_arg(parser, 'wandb-enabled', defaults.wandb_enabled, 'Log to wandb.')
        parser.add_argument('--wandb-project', default=defaults.wandb_project, help=hidden)
        parser.add_argument('--wandb-entity', default=defaults.wandb_entity, help=hidden)
        parser.add_argument('--wandb-run-name', default=defaults.wandb_run_name, help=hidden)
        parser.add_argument('--wandb-mode', default=defaults.wandb_mode, help=hidden)
        parser.add_argument('--wandb-console', default=defaults.wandb_console, help=hidden)
        add_bool_arg(parser, 'use-amp', defaults.use_amp, 'Use CUDA AMP when available.')
        add_bool_arg(parser, 'cpu-only', defaults.cpu_only, 'Force training on CPU.')
        add_bool_arg(parser, 'eval-only', defaults.eval_only, 'Only evaluate a checkpoint.')
        add_bool_arg(parser, 'test-only', defaults.test_only, 'Only evaluate the test split.')
        add_bool_arg(
            parser,
            'spectrogram-cache-enabled',
            defaults.spectrogram_cache_enabled,
            hidden,
        )
        add_bool_arg(
            parser,
            'pretrained-backbone',
            defaults.pretrained_backbone,
            hidden,
        )
        add_bool_arg(
            parser,
            'freeze-backbone',
            defaults.freeze_backbone,
            hidden,
        )
        parser.add_argument(
            '--spectrogram-cut-low-frequency',
            type=float,
            default=defaults.spectrogram_config.cut_low_frequency,
            help=hidden,
        )
        parser.add_argument(
            '--spectrogram-cut-high-frequency',
            type=float,
            default=defaults.spectrogram_config.cut_high_frequency,
            help=hidden,
        )
        parser.add_argument(
            '--spectrogram-wlen',
            type=int,
            default=defaults.spectrogram_config.wlen,
            help=hidden,
        )
        parser.add_argument(
            '--spectrogram-nfft',
            type=int,
            default=defaults.spectrogram_config.nfft,
            help=hidden,
        )
        parser.add_argument(
            '--spectrogram-sliding-window',
            type=float,
            default=defaults.spectrogram_config.sliding_window,
            help=hidden,
        )
        parser.add_argument(
            '--spectrogram-target-fs',
            type=parse_optional_int,
            default=defaults.spectrogram_config.target_fs,
            help=hidden,
        )
        parser.add_argument(
            '--lr-scheduler-factor',
            type=float,
            default=defaults.lr_scheduler_factor,
            help=hidden,
        )
        parser.add_argument(
            '--lr-scheduler-patience',
            type=int,
            default=defaults.lr_scheduler_patience,
            help=hidden,
        )
        parser.add_argument(
            '--min-learning-rate',
            type=float,
            default=defaults.min_learning_rate,
            help=hidden,
        )
        parser.add_argument(
            '--spectrogram-cache-dir',
            type=Path,
            default=defaults.spectrogram_cache_dir,
            help=hidden,
        )
        parser.add_argument('--checkpoint-path', default=defaults.checkpoint_path)
        parser.add_argument('--checkpoint-repo', default=defaults.checkpoint_repo, help=hidden)
        parser.add_argument(
            '--checkpoint-filename',
            default=defaults.checkpoint_filename,
            help=hidden,
        )
        args = parser.parse_args(argv)
        values = vars(args)
        if values['test_only']:
            values['eval_only'] = True
        if values['eval_only']:
            values['pretrained_backbone'] = False
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
        if self.checkpoint_path:
            return os.path.expanduser(self.checkpoint_path)
        if self.eval_only:
            return hf_hub_download(
                repo_id=self.checkpoint_repo,
                filename=self.checkpoint_filename,
            )
        return self.best_model_path

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
        payload['checkpoint_repo'] = self.checkpoint_repo
        payload['checkpoint_filename'] = self.checkpoint_filename
        payload['device'] = str(device)
        payload['amp_enabled'] = self.use_amp and device.type == 'cuda'
        payload['normalization'] = self.normalization_metadata()
        payload['spectrogram_config'] = self.spectrogram_config.to_metadata()
        payload['checkpoint_metric'] = 'val_loss'
