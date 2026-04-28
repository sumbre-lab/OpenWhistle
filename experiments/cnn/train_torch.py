import copy
import csv
import gc
import hashlib
import json
import os
import random
import re
import sys
from contextlib import nullcontext
from itertools import combinations
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend – plots are saved, never displayed
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Audio as HFAudio
from datasets import Dataset as HFDataset
from datasets import DatasetDict, Image as HFImage, get_dataset_split_names, load_dataset, load_from_disk
from PIL import Image as PILImage
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset as TorchDataset

try:
    import wandb
except ImportError:
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from whistle_torch import (  # noqa: E402
    ARCHITECTURE_TRAINABLE_SMALL_HEAD,
    DEFAULT_NORMALIZATION_MEAN,
    DEFAULT_NORMALIZATION_STD,
    SpectrogramConfig,
    audio_payload_to_spectrogram_image,
    build_torch_model,
    normalize_uint8_image_to_tensor,
)


# ── PATHS & HYPER-PARAMETERS ───────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE', '4'))
NUM_EPOCHS = int(os.environ.get('TRAIN_NUM_EPOCHS', '50'))
PATIENCE = int(os.environ.get('TRAIN_PATIENCE', '10'))
LEARNING_RATE = float(os.environ.get('TRAIN_LEARNING_RATE', '1e-5'))
RANDOM_STATE = 7
NUM_WORKERS = int(os.environ.get('TRAIN_NUM_WORKERS', '0'))
DEFAULT_HF_DATASET_REPO = 'dolphinteam/OpenWhistle-1.0-CNN'
DATASET_SOURCE = os.environ.get('TRAIN_DATASET_SOURCE', DEFAULT_HF_DATASET_REPO).strip()
MODELS_DIR = 'models/run3'
FIGS_DIR = 'figs/run3'
REPORTS_DIR = 'reports/run3'

TRAIN_SPLIT = 'train'
VALIDATION_SPLIT = 'validation'
TEST_SPLIT = 'test'
REQUIRED_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT)
PREFERRED_SPLIT_ORDER = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
ARCHITECTURE = ARCHITECTURE_TRAINABLE_SMALL_HEAD
TRAIN_INPUT_SOURCE = os.environ.get('TRAIN_INPUT_SOURCE', 'audio').strip().lower()

NORMALIZATION_MEAN = DEFAULT_NORMALIZATION_MEAN
NORMALIZATION_STD = DEFAULT_NORMALIZATION_STD
SPECTROGRAM_CONFIG = SpectrogramConfig(image_size=(IMG_SIZE, IMG_SIZE))

WANDB_ENABLED = os.environ.get('WANDB_ENABLED', '1').lower() not in {'0', 'false', 'no'}
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'dolphin-whistle-training')
WANDB_ENTITY = os.environ.get('WANDB_ENTITY')
WANDB_RUN_NAME = os.environ.get('WANDB_RUN_NAME')
WANDB_MODE = os.environ.get('WANDB_MODE', 'online')
WANDB_CONSOLE = os.environ.get('WANDB_CONSOLE', 'redirect')
USE_AMP = os.environ.get('TRAIN_USE_AMP', '1').lower() not in {'0', 'false', 'no'}
EVAL_ONLY = os.environ.get('TRAIN_EVAL_ONLY', '0').lower() not in {'0', 'false', 'no'}
SPECTROGRAM_CACHE_ENABLED = (
    os.environ.get('TRAIN_SPECTROGRAM_CACHE', '1').lower() not in {'0', 'false', 'no'}
)
PRETRAINED_BACKBONE = (
    os.environ.get('TRAIN_PRETRAINED_BACKBONE', '1').lower() not in {'0', 'false', 'no'}
)
SPECTROGRAM_CACHE_DIR = Path(
    os.environ.get(
        'TRAIN_SPECTROGRAM_CACHE_DIR',
        str(REPO_ROOT / 'cnn_dataset' / 'spectrogram_cache'),
    )
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


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


def spectrogram_cache_active() -> bool:
    return bool(TRAIN_INPUT_SOURCE == 'audio' and SPECTROGRAM_CACHE_ENABLED)


if TRAIN_INPUT_SOURCE not in {'audio', 'spectrogram'}:
    raise ValueError(
        f'Unsupported TRAIN_INPUT_SOURCE={TRAIN_INPUT_SOURCE!r}. '
        "Expected 'audio' or 'spectrogram'."
    )



seed_everything(RANDOM_STATE)
DEVICE = get_device()
PIN_MEMORY = DEVICE.type == 'cuda'

try:
    torch.set_float32_matmul_precision('high')
except (AttributeError, RuntimeError):
    pass


# ── DATA ───────────────────────────────────────────────────────────────────────
def media_column_name() -> str:
    return 'audio' if TRAIN_INPUT_SOURCE == 'audio' else 'spectrogram'


def required_dataset_columns() -> list[str]:
    return [media_column_name(), 'label', 'recording']


def ordered_split_names(split_names: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    split_name_set = set(split_names)
    ordered = [split_name for split_name in PREFERRED_SPLIT_ORDER if split_name in split_name_set]
    extras = sorted(split_name_set - set(ordered))
    return ordered + extras


def prepare_split_dataset(split_dataset: HFDataset, split_name: str) -> HFDataset:
    selected_media_column = media_column_name()
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


def load_split_dataset(repo_id: str, split_name: str) -> HFDataset:
    columns = required_dataset_columns()
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


def load_hf_splits(repo_id: str) -> DatasetDict:
    available_splits = set(get_dataset_split_names(repo_id))
    missing_required = set(REQUIRED_SPLITS) - available_splits
    if missing_required:
        raise ValueError(
            f'Dataset source {repo_id!r} is missing required splits: '
            f'{sorted(missing_required)}'
        )

    prepared_splits: dict[str, HFDataset] = {}
    for split_name in ordered_split_names(available_splits):
        split_dataset = load_split_dataset(repo_id, split_name)
        prepared_splits[split_name] = prepare_split_dataset(split_dataset, split_name)

    if TEST_SPLIT not in prepared_splits:
        print('Optional split \'test\' not found in Hugging Face dataset; test evaluation skipped.')

    return DatasetDict(prepared_splits)


def load_local_splits(dataset_dir: str) -> DatasetDict:
    loaded = load_from_disk(dataset_dir)
    available_splits = set(loaded.keys())
    missing_required = set(REQUIRED_SPLITS) - available_splits
    if missing_required:
        raise ValueError(
            f'Local dataset source {dataset_dir!r} is missing required splits: '
            f'{sorted(missing_required)}'
        )

    prepared_splits: dict[str, HFDataset] = {}
    for split_name in ordered_split_names(available_splits):
        prepared_splits[split_name] = prepare_split_dataset(loaded[split_name], split_name)

    if TEST_SPLIT not in prepared_splits:
        print('Optional split \'test\' not found in local dataset; test evaluation skipped.')

    return DatasetDict(prepared_splits)


def load_dataset_source(source: str) -> DatasetDict:
    source_path = Path(source)
    if source_path.exists():
        print(f'Loading local dataset from disk: {source_path}')
        return load_local_splits(str(source_path))

    print(f'Loading Hugging Face dataset: {source}')
    return load_hf_splits(source)


def split_session_ids(split_dataset: HFDataset) -> set[str] | None:
    if 'recording' not in split_dataset.column_names:
        return None
    return {extract_session_id(recording) for recording in split_dataset['recording']}


def print_split_summary(dataset_dict: DatasetDict) -> None:
    print('\nSplit summary:')
    for split_name in ordered_split_names(dataset_dict.keys()):
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


def build_split_summary_payload(dataset_dict: DatasetDict) -> dict[str, dict[str, int]]:
    payload = {}
    for split_name in ordered_split_names(dataset_dict.keys()):
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


def assert_no_session_leakage(dataset_dict: DatasetDict) -> None:
    split_sessions = {
        split_name: split_session_ids(split_dataset)
        for split_name, split_dataset in dataset_dict.items()
    }
    if any(session_ids is None for session_ids in split_sessions.values()):
        print('Session leakage check skipped: `recording` column is missing.')
        return

    available_splits = ordered_split_names(split_sessions.keys())
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


def decode_spectrogram_payload_to_uint8(image_payload: object) -> np.ndarray:
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

    target_height, target_width = SPECTROGRAM_CONFIG.image_size
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
        spectrogram_config: SpectrogramConfig,
        spectrogram_cache: LocalSpectrogramCache | None = None,
    ) -> None:
        self.dataset = dataset
        self.split_name = split_name
        self.spectrogram_config = spectrogram_config
        self.spectrogram_cache = spectrogram_cache

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataset[int(index)]
        if TRAIN_INPUT_SOURCE == 'audio':
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
            image_uint8 = decode_spectrogram_payload_to_uint8(row['spectrogram'])
        image = normalize_uint8_image_to_tensor(
            image_uint8,
            mean=NORMALIZATION_MEAN,
            std=NORMALIZATION_STD,
        )
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return image, label


# ── MODEL BUILDER ──────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    return build_torch_model(
        architecture=ARCHITECTURE,
        pretrained_backbone=PRETRAINED_BACKBONE,
    ).to(DEVICE)


def build_optimizer(model: nn.Module) -> Adam:
    return Adam(model.parameters(), lr=LEARNING_RATE)


def build_scheduler(optimizer: Adam) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )


def amp_enabled() -> bool:
    return bool(USE_AMP and DEVICE.type == 'cuda')


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {'total': int(total), 'trainable': int(trainable)}


def init_wandb_run() -> object | None:
    if not WANDB_ENABLED:
        print('Weights & Biases logging disabled by WANDB_ENABLED.')
        return None
    if wandb is None:
        print('Weights & Biases not installed, skipping wandb logging.')
        return None

    config = {
        'dataset_source': DATASET_SOURCE,
        'architecture': ARCHITECTURE,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'train_input_source': TRAIN_INPUT_SOURCE,
        'checkpoint_metric': 'val_loss',
        'spectrogram_cache_enabled': spectrogram_cache_active(),
        'spectrogram_cache_dir': str(SPECTROGRAM_CACHE_DIR),
        'num_epochs': NUM_EPOCHS,
        'patience': PATIENCE,
        'learning_rate': LEARNING_RATE,
        'random_state': RANDOM_STATE,
        'num_workers': NUM_WORKERS,
        'device': str(DEVICE),
        'amp_enabled': amp_enabled(),
        'pretrained_backbone': PRETRAINED_BACKBONE,
        'normalization_mean': NORMALIZATION_MEAN,
        'normalization_std': NORMALIZATION_STD,
        'spectrogram_config': SPECTROGRAM_CONFIG.to_metadata(),
    }

    init_kwargs = {
        'project': WANDB_PROJECT,
        'config': config,
        'mode': WANDB_MODE,
        'tags': ['torch', 'vgg16', 'dolphin', 'whistle', ARCHITECTURE],
        'settings': wandb.Settings(console=WANDB_CONSOLE),
    }
    if WANDB_ENTITY:
        init_kwargs['entity'] = WANDB_ENTITY
    if WANDB_RUN_NAME:
        init_kwargs['name'] = WANDB_RUN_NAME

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


wandb_run = init_wandb_run()

print(f'Using device: {DEVICE}')
print(
    f'Training config: input_source={TRAIN_INPUT_SOURCE}  '
    f'batch_size={BATCH_SIZE}  num_workers={NUM_WORKERS}  '
    f'amp={"on" if (USE_AMP and DEVICE.type == "cuda") else "off"}'
)
if EVAL_ONLY:
    print('Evaluation-only mode: enabled')
if spectrogram_cache_active():
    print(f'Local spectrogram cache: enabled  dir={SPECTROGRAM_CACHE_DIR}')
else:
    print('Local spectrogram cache: disabled')


def wandb_log(run: object | None, payload: dict[str, object]) -> None:
    if run is None or wandb is None:
        return
    try:
        run.log(payload)
    except Exception as exc:
        print(f'Warning: wandb logging failed ({exc}).')


def wandb_log_artifact_images(run: object | None) -> None:
    if run is None or wandb is None:
        return

    payload = {}
    metrics_path = os.path.join(FIGS_DIR, 'metrics_training.png')
    roc_path = os.path.join(FIGS_DIR, 'roc_validation_test.png')
    if os.path.exists(metrics_path):
        payload['figures/metrics_training'] = wandb.Image(metrics_path)
    if os.path.exists(roc_path):
        payload['figures/roc_validation_test'] = wandb.Image(roc_path)
    for split_name in (VALIDATION_SPLIT, TEST_SPLIT):
        confusion_path = os.path.join(FIGS_DIR, f'{split_name}_confusion_matrix.png')
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
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
    use_autocast = amp_enabled()

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            autocast_manager = (
                torch.amp.autocast(device_type='cuda', enabled=use_autocast)
                if DEVICE.type == 'cuda'
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
    epoch: int,
    val_loss: float,
    test_split_name: str | None,
) -> None:
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'img_size': IMG_SIZE,
            'class_to_idx': {'0': 0, '1': 1},
            'normalization': {'mean': NORMALIZATION_MEAN, 'std': NORMALIZATION_STD},
            'architecture': ARCHITECTURE,
            'dataset_source': DATASET_SOURCE,
            'train_input_source': TRAIN_INPUT_SOURCE,
            'validation_split': VALIDATION_SPLIT,
            'test_split': test_split_name,
            'spectrogram_config': SPECTROGRAM_CONFIG.to_metadata(),
            'checkpoint_selection_metric': 'val_loss',
        },
        path,
    )


def plot_training_curves(history: dict[str, list[float]]) -> None:
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
    plt.savefig(os.path.join(FIGS_DIR, 'metrics_training.png'))
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
    plt.savefig(os.path.join(FIGS_DIR, 'roc_validation_test.png'))
    plt.close()


def save_confusion_matrix_artifacts(
    split_name: str,
    metrics: dict[str, np.ndarray | float],
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

    csv_path = os.path.join(REPORTS_DIR, f'{split_name}_confusion_matrix.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['actual\\predicted', 'noise', 'whistle'])
        writer.writerow(['noise', counts['tn'], counts['fp']])
        writer.writerow(['whistle', counts['fn'], counts['tp']])

    fig_path = os.path.join(FIGS_DIR, f'{split_name}_confusion_matrix.png')
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
        'dataset_source': DATASET_SOURCE,
        'train_input_source': TRAIN_INPUT_SOURCE,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_model_path': best_model_path,
        'split_summary': split_summary,
        'metrics': {
            VALIDATION_SPLIT: {
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
            'figures_dir': FIGS_DIR,
            'reports_dir': REPORTS_DIR,
            'validation_session_report_path': validation_session_report_path,
            'test_session_report_path': test_session_report_path,
            'confusion_matrices': confusion_artifacts,
        },
    }
    if test_metrics is not None:
        payload['metrics'][TEST_SPLIT] = {
            'loss': float(test_metrics['loss']),
            'accuracy': float(test_metrics['accuracy']),
            'f1': float(test_metrics['f1']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'positive_prediction_rate': float(test_metrics['positive_prediction_rate']),
        }

    summary_path = os.path.join(REPORTS_DIR, 'run_summary.json')
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
) -> str | None:
    if not rows:
        return None

    path = os.path.join(REPORTS_DIR, f'{split_name}_session_metrics.csv')
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


split_datasets = load_dataset_source(DATASET_SOURCE)
HAS_TEST_SPLIT = TEST_SPLIT in split_datasets
print_split_summary(split_datasets)
assert_no_session_leakage(split_datasets)
split_summary_payload = build_split_summary_payload(split_datasets)
spectrogram_cache = LocalSpectrogramCache(
    cache_root=SPECTROGRAM_CACHE_DIR,
    repo_id=DATASET_SOURCE,
    spectrogram_config=SPECTROGRAM_CONFIG,
    enabled=spectrogram_cache_active(),
)
if spectrogram_cache_active():
    print('Spectrogram cache coverage:')
    all_cached = True
    for split_name in ordered_split_names(split_datasets.keys()):
        expected = int(split_datasets[split_name].num_rows)
        cached = spectrogram_cache.count_cached_files(split_name)
        if cached != expected:
            all_cached = False
        print(f'  {split_name:10s} cached={cached:6d}/{expected:6d}')
    if all_cached:
        print('Spectrogram cache is warm: all examples are already available locally.')
    else:
        print('Spectrogram cache is partial: missing examples will be generated on demand.')

train_dataset = TrainingDataset(
    split_datasets[TRAIN_SPLIT],
    TRAIN_SPLIT,
    SPECTROGRAM_CONFIG,
    spectrogram_cache=spectrogram_cache,
)
valid_dataset = TrainingDataset(
    split_datasets[VALIDATION_SPLIT],
    VALIDATION_SPLIT,
    SPECTROGRAM_CONFIG,
    spectrogram_cache=spectrogram_cache,
)
test_dataset = (
    TrainingDataset(
        split_datasets[TEST_SPLIT],
        TEST_SPLIT,
        SPECTROGRAM_CONFIG,
        spectrogram_cache=spectrogram_cache,
    )
    if HAS_TEST_SPLIT
    else None
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
test_loader = (
    DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    if test_dataset is not None
    else None
)


# ── TRAINING LOOP ──────────────────────────────────────────────────────────────
best_model_path = os.path.join(MODELS_DIR, 'model_vgg_best.pt')
eval_checkpoint_path = os.environ.get('TRAIN_CHECKPOINT_PATH', best_model_path)
resolved_model_artifact_path = best_model_path
criterion = nn.CrossEntropyLoss()

net = build_model()
update_wandb_run_config(wandb_run, net, split_summary_payload)
optimizer = build_optimizer(net)
scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled())
scheduler = build_scheduler(optimizer)

history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': [],
    'f1': [],
    'val_f1': [],
    'precision': [],
    'val_precision': [],
    'recall': [],
    'val_recall': [],
    'positive_prediction_rate': [],
    'val_positive_prediction_rate': [],
}
best_state = None
best_val_loss = np.inf
best_epoch = 0
epochs_without_improvement = 0

if EVAL_ONLY:
    if not os.path.exists(eval_checkpoint_path):
        raise FileNotFoundError(
            f'Evaluation-only mode requested but checkpoint does not exist: '
            f'{eval_checkpoint_path}'
        )
    checkpoint = torch.load(eval_checkpoint_path, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = int(checkpoint.get('epoch', 0))
    best_val_loss = float(checkpoint.get('val_loss', np.inf))
    resolved_model_artifact_path = eval_checkpoint_path
    print(
        f'Loaded checkpoint for evaluation: {eval_checkpoint_path}  '
        f'(epoch={best_epoch}, val_loss={best_val_loss:.4f})'
    )
else:
    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = run_epoch(net, train_loader, criterion, optimizer=optimizer, scaler=scaler)
            val_metrics = run_epoch(net, valid_loader, criterion)
            scheduler.step(float(val_metrics['loss']))

            history['loss'].append(float(train_metrics['loss']))
            history['val_loss'].append(float(val_metrics['loss']))
            history['accuracy'].append(float(train_metrics['accuracy']))
            history['val_accuracy'].append(float(val_metrics['accuracy']))
            history['f1'].append(float(train_metrics['f1']))
            history['val_f1'].append(float(val_metrics['f1']))
            history['precision'].append(float(train_metrics['precision']))
            history['val_precision'].append(float(val_metrics['precision']))
            history['recall'].append(float(train_metrics['recall']))
            history['val_recall'].append(float(val_metrics['recall']))
            history['positive_prediction_rate'].append(float(train_metrics['positive_prediction_rate']))
            history['val_positive_prediction_rate'].append(
                float(val_metrics['positive_prediction_rate'])
            )

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
                f"loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}  "
                f"f1={train_metrics['f1']:.4f}  prec={train_metrics['precision']:.4f}  "
                f"rec={train_metrics['recall']:.4f}  ppr={train_metrics['positive_prediction_rate']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
                f"val_f1={val_metrics['f1']:.4f}  val_prec={val_metrics['precision']:.4f}  "
                f"val_rec={val_metrics['recall']:.4f}  val_ppr={val_metrics['positive_prediction_rate']:.4f}  "
                f"lr={current_lr:.2e}"
            )
            wandb_log(
                wandb_run,
                {
                    'epoch': epoch,
                    'train/loss': float(train_metrics['loss']),
                    'train/accuracy': float(train_metrics['accuracy']),
                    'train/f1': float(train_metrics['f1']),
                    'train/precision': float(train_metrics['precision']),
                    'train/recall': float(train_metrics['recall']),
                    'train/positive_prediction_rate': float(train_metrics['positive_prediction_rate']),
                    'validation/loss': float(val_metrics['loss']),
                    'validation/accuracy': float(val_metrics['accuracy']),
                    'validation/f1': float(val_metrics['f1']),
                    'validation/precision': float(val_metrics['precision']),
                    'validation/recall': float(val_metrics['recall']),
                    'validation/positive_prediction_rate': float(
                        val_metrics['positive_prediction_rate']
                    ),
                    'train/learning_rate': float(current_lr),
                    'validation/best_loss_so_far': float(min(best_val_loss, float(val_metrics['loss']))),
                },
            )

            if float(val_metrics['loss']) < best_val_loss:
                best_val_loss = float(val_metrics['loss'])
                best_epoch = epoch
                best_state = copy.deepcopy(net.state_dict())
                save_checkpoint(
                    best_model_path,
                    net,
                    epoch,
                    best_val_loss,
                    TEST_SPLIT if HAS_TEST_SPLIT else None,
                )
                print(f'  -> Best checkpoint saved  (val_loss = {best_val_loss:.4f})')
                wandb_log(
                    wandb_run,
                    {
                        'epoch': epoch,
                        'validation/checkpoint_saved': 1,
                        'validation/best_loss': float(best_val_loss),
                        'validation/best_epoch': int(best_epoch),
                    },
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(f'  -> Early stopping triggered after {epoch} epochs.')
                break
    except RuntimeError as exc:
        message = str(exc).lower()
        if 'out of memory' in message:
            print(
                'CUDA OOM during training. Try lowering TRAIN_BATCH_SIZE further '
                '(for example 2), keep TRAIN_USE_AMP=1, and close other GPU-heavy apps.'
            )
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        raise

if best_state is not None:
    net.load_state_dict(best_state)

plot_training_curves(history)

validation_metrics = run_epoch(net, valid_loader, criterion)
test_metrics = run_epoch(net, test_loader, criterion) if test_loader is not None else None
validation_session_rows = build_session_report_rows(
    VALIDATION_SPLIT,
    split_datasets[VALIDATION_SPLIT],
    validation_metrics,
)
test_session_rows = (
    build_session_report_rows(
        TEST_SPLIT,
        split_datasets[TEST_SPLIT],
        test_metrics,
    )
    if test_metrics is not None
    else []
)
validation_session_report_path = write_session_report_csv(
    VALIDATION_SPLIT,
    validation_session_rows,
)
test_session_report_path = (
    write_session_report_csv(
        TEST_SPLIT,
        test_session_rows,
    )
    if test_session_rows
    else None
)
validation_confusion_csv_path, validation_confusion_fig_path, validation_confusion_counts = (
    save_confusion_matrix_artifacts(VALIDATION_SPLIT, validation_metrics)
)
test_confusion_csv_path = None
test_confusion_fig_path = None
test_confusion_counts = None
if test_metrics is not None:
    (
        test_confusion_csv_path,
        test_confusion_fig_path,
        test_confusion_counts,
    ) = save_confusion_matrix_artifacts(TEST_SPLIT, test_metrics)

roc_curves = []
validation_roc = maybe_build_roc_curve(VALIDATION_SPLIT, validation_metrics)
if validation_roc is not None:
    roc_curves.append(validation_roc)
if test_metrics is not None:
    test_roc = maybe_build_roc_curve(TEST_SPLIT, test_metrics)
    if test_roc is not None:
        roc_curves.append(test_roc)
plot_roc_curves(roc_curves)
wandb_log_artifact_images(wandb_run)

confusion_artifacts: dict[str, dict[str, object]] = {
    VALIDATION_SPLIT: {
        'csv_path': validation_confusion_csv_path,
        'figure_path': validation_confusion_fig_path,
        'counts': validation_confusion_counts,
    }
}
if test_confusion_counts is not None:
    confusion_artifacts[TEST_SPLIT] = {
        'csv_path': test_confusion_csv_path,
        'figure_path': test_confusion_fig_path,
        'counts': test_confusion_counts,
    }
run_summary_path = write_run_summary_json(
    best_epoch=best_epoch,
    best_val_loss=best_val_loss,
    best_model_path=resolved_model_artifact_path,
    split_summary=split_summary_payload,
    validation_metrics=validation_metrics,
    test_metrics=test_metrics,
    confusion_artifacts=confusion_artifacts,
    validation_session_report_path=validation_session_report_path,
    test_session_report_path=test_session_report_path,
)

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
print(f'  Best epoch: {best_epoch}')
print(f'  Best model saved to: {resolved_model_artifact_path}')
print(f'  Run summary: {run_summary_path}')
print('  Session-level preview:')
print_session_report_preview(
    VALIDATION_SPLIT,
    validation_session_rows,
    validation_session_report_path,
)
if test_session_rows:
    print_session_report_preview(
        TEST_SPLIT,
        test_session_rows,
        test_session_report_path,
    )

final_wandb_payload = {
    'epoch': int(best_epoch),
    'validation/final_loss': float(validation_metrics['loss']),
    'validation/final_accuracy': float(validation_metrics['accuracy']),
    'validation/final_f1': float(validation_metrics['f1']),
    'validation/final_precision': float(validation_metrics['precision']),
    'validation/final_recall': float(validation_metrics['recall']),
    'validation/final_positive_prediction_rate': float(
        validation_metrics['positive_prediction_rate']
    ),
    'training/best_val_loss': float(best_val_loss),
    'training/best_epoch': int(best_epoch),
    'training/best_model_path': resolved_model_artifact_path,
}
if test_metrics is not None:
    final_wandb_payload.update(
        {
            'test/loss': float(test_metrics['loss']),
            'test/accuracy': float(test_metrics['accuracy']),
            'test/f1': float(test_metrics['f1']),
            'test/precision': float(test_metrics['precision']),
            'test/recall': float(test_metrics['recall']),
            'test/positive_prediction_rate': float(test_metrics['positive_prediction_rate']),
        }
    )
wandb_log(wandb_run, final_wandb_payload)
wandb_log_table(wandb_run, 'validation/session_metrics', validation_session_rows)
if test_session_rows:
    wandb_log_table(wandb_run, 'test/session_metrics', test_session_rows)
if wandb_run is not None:
    try:
        wandb_run.summary['best_epoch'] = int(best_epoch)
        wandb_run.summary['best_val_loss'] = float(best_val_loss)
        wandb_run.summary['validation_accuracy'] = float(validation_metrics['accuracy'])
        wandb_run.summary['validation_f1'] = float(validation_metrics['f1'])
        wandb_run.summary['validation_precision'] = float(validation_metrics['precision'])
        wandb_run.summary['validation_recall'] = float(validation_metrics['recall'])
        wandb_run.summary['validation_positive_prediction_rate'] = float(
            validation_metrics['positive_prediction_rate']
        )
        if test_metrics is not None:
            wandb_run.summary['test_accuracy'] = float(test_metrics['accuracy'])
            wandb_run.summary['test_f1'] = float(test_metrics['f1'])
            wandb_run.summary['test_precision'] = float(test_metrics['precision'])
            wandb_run.summary['test_recall'] = float(test_metrics['recall'])
            wandb_run.summary['test_positive_prediction_rate'] = float(
                test_metrics['positive_prediction_rate']
            )
        if validation_session_report_path is not None:
            wandb_run.summary['validation_session_report_path'] = validation_session_report_path
        if test_session_report_path is not None:
            wandb_run.summary['test_session_report_path'] = test_session_report_path
        wandb_run.summary['run_summary_path'] = run_summary_path
        wandb_run.summary['validation_confusion_csv_path'] = validation_confusion_csv_path
        wandb_run.summary['validation_confusion_fig_path'] = validation_confusion_fig_path
        if test_confusion_csv_path is not None:
            wandb_run.summary['test_confusion_csv_path'] = test_confusion_csv_path
        if test_confusion_fig_path is not None:
            wandb_run.summary['test_confusion_fig_path'] = test_confusion_fig_path
        wandb_run.summary['best_model_path'] = resolved_model_artifact_path
    except Exception as exc:
        print(f'Warning: unable to write wandb summary ({exc}).')
    finally:
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f'Warning: unable to finish wandb run ({exc}).')

del train_dataset, valid_dataset, test_dataset
del train_loader, valid_loader, test_loader
del split_datasets, net, optimizer, scheduler, scaler, best_state, split_summary_payload
del spectrogram_cache
gc.collect()
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
