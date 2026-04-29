import hashlib
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np
import torch
from datasets import Audio as HFAudio
from datasets import Dataset as HFDataset
from datasets import DatasetDict, Image as HFImage, get_dataset_split_names, load_dataset, load_from_disk
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset as TorchDataset

from utils.config import TrainConfig
from utils.model import (
    SpectrogramConfig,
    audio_payload_to_spectrogram_image,
    normalize_uint8_image_to_tensor,
)
from utils.runtime_utils import extract_session_id

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