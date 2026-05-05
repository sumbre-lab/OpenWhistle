import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        'OPENWHISTLE_CNN_INFERENCE_OUTPUT_ROOT',
        REPO_ROOT / 'cnn' / 'runs' / 'external_inference',
    )
).expanduser()

DOLPHIN_WHISTLE_EXTRACTOR_ROOT = Path(
    os.environ.get(
        'DOLPHIN_WHISTLE_EXTRACTOR_ROOT',
        Path.home() / 'Documents' / 'DolphinWhistleExtractor',
    )
).expanduser()


@dataclass(frozen=True)
class ExternalDatasetSpec:
    name: str
    env_vars: tuple[str, ...]
    default_root: Path | None
    recursive: bool
    description: str


@dataclass(frozen=True)
class ResolvedExternalDataset:
    name: str
    source_dir: Path
    recordings_dir: Path
    output_dir: Path
    specific_files_path: Path | None
    recursive: bool


EXTERNAL_DATASETS = {
    'wmmsd': ExternalDatasetSpec(
        name='wmmsd',
        env_vars=(
            'OPENWHISTLE_CNN_WMMSD_DIR',
            'WMMSD_PREPARED_DIR',
            'WMSD_PREPARED_DIR',
        ),
        default_root=(
            DOLPHIN_WHISTLE_EXTRACTOR_ROOT
            / 'benchmark_data'
            / 'wmmsd_first_pass'
        ),
        recursive=False,
        description='Watkins/WMMSD prepared folder or raw audio folder.',
    ),
    'wmsd': ExternalDatasetSpec(
        name='wmmsd',
        env_vars=(
            'OPENWHISTLE_CNN_WMMSD_DIR',
            'WMMSD_PREPARED_DIR',
            'WMSD_PREPARED_DIR',
        ),
        default_root=(
            DOLPHIN_WHISTLE_EXTRACTOR_ROOT
            / 'benchmark_data'
            / 'wmmsd_first_pass'
        ),
        recursive=False,
        description='Alias for wmmsd.',
    ),
    'dclde': ExternalDatasetSpec(
        name='dclde',
        env_vars=('OPENWHISTLE_CNN_DCLDE_DIR', 'DCLDE_PREPARED_DIR'),
        default_root=(
            DOLPHIN_WHISTLE_EXTRACTOR_ROOT
            / 'benchmark_data'
            / 'dclde_clip'
        ),
        recursive=False,
        description='DCLDE prepared clip folder or raw audio folder.',
    ),
}


def _first_env_path(env_vars: tuple[str, ...]) -> Path | None:
    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            return Path(value).expanduser()
    return None


def dataset_names() -> list[str]:
    return sorted(EXTERNAL_DATASETS)


def get_dataset_spec(name: str) -> ExternalDatasetSpec:
    normalized = name.strip().lower()
    if normalized not in EXTERNAL_DATASETS:
        choices = ', '.join(dataset_names())
        raise ValueError(f'Unknown external dataset {name!r}. Choices: {choices}')
    return EXTERNAL_DATASETS[normalized]


def resolve_dataset(
    name: str,
    output_root: Path | None = None,
) -> ResolvedExternalDataset:
    spec = get_dataset_spec(name)
    source_dir = _first_env_path(spec.env_vars) or spec.default_root
    if source_dir is None:
        env_hint = ' or '.join(spec.env_vars)
        raise FileNotFoundError(
            f'No default path is configured for {spec.name}. Set {env_hint}.'
        )

    source_dir = source_dir.expanduser()
    flat_recordings_dir = source_dir / 'flat_recordings'
    if flat_recordings_dir.is_dir():
        recordings_dir = flat_recordings_dir
        recursive = False
    else:
        recordings_dir = source_dir
        recursive = spec.recursive

    specific_files_path = source_dir / 'specific_files.txt'
    if not specific_files_path.exists():
        specific_files_path = None

    resolved_output_root = (
        Path(output_root).expanduser()
        if output_root is not None
        else DEFAULT_OUTPUT_ROOT
    )
    return ResolvedExternalDataset(
        name=spec.name,
        source_dir=source_dir,
        recordings_dir=recordings_dir,
        output_dir=resolved_output_root / spec.name,
        specific_files_path=specific_files_path,
        recursive=recursive,
    )
