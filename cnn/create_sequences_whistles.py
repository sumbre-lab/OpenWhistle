import argparse
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


REQUIRED_COLUMNS = {'initial_point', 'finish_point'}
SEQUENCE_COLUMNS = [
    'recording_name',
    'start_time',
    'end_time',
    'duration',
    'detections_count',
    'mean_confidence',
    'max_confidence',
]
DEFAULT_PREDICTION_PATTERN = '*_predictions.csv'
SequenceSource = Literal['auto', 'local', 'hf']

def sequence_source_choices() -> tuple[str, ...]:
    return get_args(SequenceSource)


@dataclass(frozen=True)
class SequenceConfig:
    input_source: str
    output_csv: Path | None = None
    prediction_pattern: str = DEFAULT_PREDICTION_PATTERN
    max_gap: float = 6.0
    min_duration: float = 2.0
    max_duration: float = 20.0
    include_empty_recordings: bool = True
    use_summary_csv: bool = False
    source: SequenceSource = 'auto'
    hf_revision: str | None = None
    hf_subdir: str | None = None
    hf_split: str = 'train'

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> 'SequenceConfig':
        defaults = cls(input_source='')
        parser = argparse.ArgumentParser(
            description='Create whistle sequences from cnn/inference.py CSV outputs.'
        )
        parser.add_argument(
            'input_source',
            help=(
                'Local inference output directory, a local CSV, or a Hugging Face '
                'dataset repo id containing inference CSVs.'
            ),
        )
        parser.add_argument(
            '--source',
            choices=sequence_source_choices(),
            default=defaults.source,
            help='Input source type. Auto uses local paths when they exist, else HF.',
        )
        parser.add_argument(
            '--output-csv',
            type=Path,
            default=defaults.output_csv,
            help=(
                'Destination CSV. Defaults to <local_input>/whistle_sequences.csv '
                'for directories, or cnn/runs/sequences/<hf_repo>/whistle_sequences.csv '
                'for HF inputs.'
            ),
        )
        parser.add_argument(
            '--prediction-pattern',
            default=defaults.prediction_pattern,
            help='CSV glob used when inference_output is a directory.',
        )
        parser.add_argument('--max-gap', type=float, default=defaults.max_gap)
        parser.add_argument('--min-duration', type=float, default=defaults.min_duration)
        parser.add_argument('--max-duration', type=float, default=defaults.max_duration)
        parser.add_argument(
            '--use-summary-csv',
            action='store_true',
            default=defaults.use_summary_csv,
            help='Read detections.csv instead of per-recording *_predictions.csv files.',
        )
        parser.add_argument(
            '--no-empty-recordings',
            dest='include_empty_recordings',
            action='store_false',
            default=defaults.include_empty_recordings,
            help='Do not include recordings with no sequence in the output CSV.',
        )
        parser.add_argument(
            '--hf-revision',
            default=defaults.hf_revision,
            help='Optional Hugging Face dataset revision, branch, or commit.',
        )
        parser.add_argument(
            '--hf-subdir',
            default=defaults.hf_subdir,
            help='Optional subdirectory inside the Hugging Face dataset repo.',
        )
        parser.add_argument(
            '--hf-split',
            default=defaults.hf_split,
            help='Hugging Face split to read when no inference CSVs are stored in the repo.',
        )
        args = parser.parse_args(argv)
        return cls(
            input_source=args.input_source,
            output_csv=args.output_csv,
            prediction_pattern=args.prediction_pattern,
            max_gap=args.max_gap,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            include_empty_recordings=args.include_empty_recordings,
            use_summary_csv=args.use_summary_csv,
            source=args.source,
            hf_revision=args.hf_revision,
            hf_subdir=args.hf_subdir,
            hf_split=args.hf_split,
        )

    def validate(self) -> None:
        if self.source == 'local' and not self.local_input_path.exists():
            raise FileNotFoundError(f'Input source not found: {self.input_source}')
        if self.max_gap < 0:
            raise ValueError('max_gap must be non-negative.')
        if self.min_duration < 0:
            raise ValueError('min_duration must be non-negative.')
        if self.max_duration <= 0:
            raise ValueError('max_duration must be positive.')
        if self.max_duration < self.min_duration:
            raise ValueError('max_duration must be greater than or equal to min_duration.')

    @property
    def local_input_path(self) -> Path:
        return Path(self.input_source).expanduser()

    @property
    def source_kind(self) -> SequenceSource:
        if self.source != 'auto':
            return self.source
        return 'local' if self.local_input_path.exists() else 'hf'

    @property
    def hf_repo_id(self) -> str:
        return self.input_source.strip()

    @property
    def hf_repo_slug(self) -> str:
        return self.hf_repo_id.replace('/', '__')

    @property
    def resolved_output_csv(self) -> Path:
        if self.output_csv is not None:
            return self.output_csv.expanduser()
        if self.source_kind == 'hf':
            return (
                (Path(__file__).resolve().parent / 'runs/sequences')
                / self.hf_repo_slug
                / 'whistle_sequences.csv'
            )
        local_input = self.local_input_path
        if local_input.is_dir():
            return local_input / 'whistle_sequences.csv'
        return local_input.with_name(f'{local_input.stem}_sequences.csv')

def recording_name_from_prediction_path(csv_path: Path) -> str:
    name = csv_path.name
    if name.endswith('.wav_predictions.csv'):
        return name[: -len('_predictions.csv')]
    if name.endswith('_predictions.csv'):
        return name[: -len('_predictions.csv')]
    return csv_path.stem

def read_prediction_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return normalize_prediction_dataframe(df, str(csv_path))

def normalize_prediction_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f'{source_name} is missing columns: {sorted(missing_columns)}')

    df = df.copy()
    df['initial_point'] = pd.to_numeric(df['initial_point'], errors='coerce')
    df['finish_point'] = pd.to_numeric(df['finish_point'], errors='coerce')
    if 'confidence' in df.columns:
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    else:
        df['confidence'] = pd.NA
    df = df.dropna(subset=['initial_point', 'finish_point'])
    df = df[df['finish_point'] >= df['initial_point']]
    return df.sort_values(['initial_point', 'finish_point']).reset_index(drop=True)

def split_sequence_if_needed(
    sequences: list[dict[str, object]],
    recording_name: str,
    current_rows: list[dict[str, float]],
    config: SequenceConfig,
) -> None:
    if not current_rows:
        return

    start_time = float(current_rows[0]['initial_point'])
    end_time = float(max(row['finish_point'] for row in current_rows))
    duration = end_time - start_time
    if duration < config.min_duration:
        return

    confidences = [
        float(row['confidence'])
        for row in current_rows
        if not pd.isna(row.get('confidence'))
    ]
    sequences.append(
        {
            'recording_name': recording_name,
            'start_time': round(start_time, 2),
            'end_time': round(end_time, 2),
            'duration': round(duration, 2),
            'detections_count': len(current_rows),
            'mean_confidence': (
                round(sum(confidences) / len(confidences), 6) if confidences else pd.NA
            ),
            'max_confidence': round(max(confidences), 6) if confidences else pd.NA,
        }
    )

def build_sequences_for_recording(
    recording_name: str,
    detections: pd.DataFrame,
    config: SequenceConfig,
) -> list[dict[str, object]]:
    sequences: list[dict[str, object]] = []
    current_rows: list[dict[str, float]] = []
    previous_end: float | None = None

    for row in detections.to_dict('records'):
        start = float(row['initial_point'])
        end = float(row['finish_point'])
        starts_new_group = (
            previous_end is not None
            and (start - previous_end) >= config.max_gap
        )

        if current_rows and not starts_new_group:
            candidate_start = float(current_rows[0]['initial_point'])
            candidate_end = max(
                float(item['finish_point'])
                for item in current_rows + [row]
            )
            starts_new_group = (candidate_end - candidate_start) > config.max_duration

        if starts_new_group:
            split_sequence_if_needed(sequences, recording_name, current_rows, config)
            current_rows = []

        current_rows.append(row)
        previous_end = end

    split_sequence_if_needed(sequences, recording_name, current_rows, config)
    return sequences

def prediction_csv_paths(config: SequenceConfig) -> list[Path]:
    if config.source_kind == 'hf':
        return hf_prediction_csv_paths(config)

    source = config.local_input_path
    if source.is_file():
        return [source]

    if config.use_summary_csv:
        summary_csv = source / 'detections.csv'
        if not summary_csv.exists():
            raise FileNotFoundError(f'Summary CSV not found: {summary_csv}')
        return [summary_csv]

    csv_paths = sorted(source.rglob(config.prediction_pattern))
    if not csv_paths:
        summary_csv = source / 'detections.csv'
        if summary_csv.exists():
            print(
                'No per-recording prediction CSVs found; falling back to detections.csv. '
                'Recordings without detections cannot be included.'
            )
            return [summary_csv]
        raise FileNotFoundError(
            f'No prediction CSVs matching {config.prediction_pattern!r} found in {source}'
        )
    return csv_paths

def hf_file_in_subdir(repo_file: str, hf_subdir: str | None) -> bool:
    if hf_subdir is None:
        return True
    normalized_subdir = hf_subdir.strip('/')
    return (
        repo_file == normalized_subdir
        or repo_file.startswith(f'{normalized_subdir}/')
    )

def hf_file_matches_pattern(repo_file: str, pattern: str) -> bool:
    return (
        fnmatch.fnmatch(Path(repo_file).name, pattern)
        or fnmatch.fnmatch(repo_file, pattern)
    )

def download_hf_file(config: SequenceConfig, repo_file: str) -> Path:
    downloaded_path = hf_hub_download(
        repo_id=config.hf_repo_id,
        filename=repo_file,
        repo_type='dataset',
        revision=config.hf_revision,
    )
    return Path(downloaded_path)

def hf_prediction_csv_paths(config: SequenceConfig) -> list[Path]:
    repo_files = sorted(
        repo_file
        for repo_file in list_repo_files(
            repo_id=config.hf_repo_id,
            repo_type='dataset',
            revision=config.hf_revision,
        )
        if hf_file_in_subdir(repo_file, config.hf_subdir)
    )

    if config.use_summary_csv:
        summary_files = [
            repo_file
            for repo_file in repo_files
            if Path(repo_file).name == 'detections.csv'
        ]
        if not summary_files:
            print(
                f'No detections.csv found in Hugging Face dataset '
                f'{config.hf_repo_id!r}; falling back to split {config.hf_split!r}.'
            )
            return []
        if len(summary_files) > 1:
            print(
                f'Found {len(summary_files)} detections.csv files; using the first one: '
                f'{summary_files[0]}'
            )
        return [download_hf_file(config, summary_files[0])]

    prediction_files = [
        repo_file
        for repo_file in repo_files
        if hf_file_matches_pattern(repo_file, config.prediction_pattern)
    ]
    if not prediction_files:
        summary_files = [
            repo_file
            for repo_file in repo_files
            if Path(repo_file).name == 'detections.csv'
        ]
        if summary_files:
            print(
                'No per-recording prediction CSVs found on HF; falling back to '
                'detections.csv. Recordings without detections cannot be included.'
            )
            return [download_hf_file(config, summary_files[0])]
        print(
            f'No prediction CSVs matching {config.prediction_pattern!r} found in '
            f'Hugging Face dataset {config.hf_repo_id!r}; falling back to split '
            f'{config.hf_split!r}.'
        )
        return []

    print(f'Downloading {len(prediction_files)} prediction CSV file(s) from HF.')
    return [download_hf_file(config, repo_file) for repo_file in prediction_files]

def load_hf_prediction_dataframe(config: SequenceConfig) -> pd.DataFrame:
    dataset = load_dataset(
        config.hf_repo_id,
        split=config.hf_split,
        revision=config.hf_revision,
    )
    df = dataset.to_pandas()
    return normalize_prediction_dataframe(
        df,
        f'Hugging Face dataset {config.hf_repo_id!r} split {config.hf_split!r}',
    )

def recording_groups_from_csv(csv_path: Path) -> tuple[set[str], dict[str, pd.DataFrame]]:
    fallback_recording = recording_name_from_prediction_path(csv_path)
    df = read_prediction_csv(csv_path)
    return recording_groups_from_dataframe(df, fallback_recording)

def recording_groups_from_dataframe(
    df: pd.DataFrame,
    fallback_recording: str,
) -> tuple[set[str], dict[str, pd.DataFrame]]:
    if 'file_name' not in df.columns:
        return {fallback_recording}, {fallback_recording: df}

    if df.empty:
        return {fallback_recording}, {fallback_recording: df}

    df['file_name'] = df['file_name'].fillna(fallback_recording).astype(str)
    recording_names = set(df['file_name'])
    groups = {
        recording_name: recording_df
        for recording_name, recording_df in df.groupby('file_name', sort=True)
    }
    return recording_names, groups

def normalize_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if 'detections_count' in df.columns:
        df['detections_count'] = df['detections_count'].astype('Int64')
    return df

def build_sequence_dataframe(config: SequenceConfig) -> pd.DataFrame:
    csv_paths = prediction_csv_paths(config)
    all_recordings: set[str] = set()
    sequence_rows: list[dict[str, object]] = []

    if csv_paths:
        print(f'Found {len(csv_paths)} prediction CSV file(s).')
        grouped_sources = []
        for index, csv_path in enumerate(csv_paths, start=1):
            print(f'Processing {index}/{len(csv_paths)}: {csv_path}')
            grouped_sources.append(recording_groups_from_csv(csv_path))
    elif config.source_kind == 'hf':
        print(
            f'Loading Hugging Face dataset split: '
            f'{config.hf_repo_id}@{config.hf_split}'
        )
        hf_df = load_hf_prediction_dataframe(config)
        grouped_sources = [recording_groups_from_dataframe(hf_df, config.hf_repo_id)]
    else:
        grouped_sources = []

    for recording_names, grouped_detections in grouped_sources:
        all_recordings.update(recording_names)
        for recording_name, detections in grouped_detections.items():
            sequences = build_sequences_for_recording(
                recording_name,
                detections,
                config,
            )
            sequence_rows.extend(sequences)

    sequences_df = pd.DataFrame(sequence_rows, columns=SEQUENCE_COLUMNS)
    if not config.include_empty_recordings:
        sequences_df = sequences_df.sort_values(
            ['recording_name', 'start_time']
        ).reset_index(drop=True)
        return normalize_output_dtypes(sequences_df)

    recordings_df = pd.DataFrame({'recording_name': sorted(all_recordings)})
    if sequences_df.empty:
        empty_columns = [
            column for column in SEQUENCE_COLUMNS if column != 'recording_name'
        ]
        for column in empty_columns:
            recordings_df[column] = pd.NA
        return normalize_output_dtypes(recordings_df)

    final_df = recordings_df.merge(sequences_df, on='recording_name', how='left')
    final_df = final_df.sort_values(
        ['recording_name', 'start_time']
    ).reset_index(drop=True)
    return normalize_output_dtypes(final_df)

class SequenceRun:
    def __init__(self, config: SequenceConfig) -> None:
        self.config = config
        self.output_csv = config.resolved_output_csv

    def print_startup_summary(self) -> None:
        print(f'Input source    : {self.config.input_source}')
        print(f'Source type     : {self.config.source_kind}')
        print(f'Output CSV      : {self.output_csv}')
        print(
            f'Sequence rules  : max_gap={self.config.max_gap}s  '
            f'min_duration={self.config.min_duration}s  '
            f'max_duration={self.config.max_duration}s'
        )

    def run(self) -> None:
        self.config.validate()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.print_startup_summary()

        sequences_df = build_sequence_dataframe(self.config)
        sequences_df.to_csv(self.output_csv, index=False)
        self.print_summary(sequences_df)

    def print_summary(self, sequences_df: pd.DataFrame) -> None:
        if sequences_df.empty:
            recordings_with_sequences = 0
            total_sequences = 0
            total_recordings = 0
        else:
            recordings_with_sequences = (
                sequences_df['start_time']
                .notna()
                .groupby(sequences_df['recording_name'])
                .any()
                .sum()
            )
            total_sequences = int(sequences_df['start_time'].notna().sum())
            total_recordings = int(sequences_df['recording_name'].nunique())

        print('Done.')
        print(f'  Recordings processed      : {total_recordings}')
        print(f'  Recordings with sequences : {recordings_with_sequences}')
        print(f'  Sequences found           : {total_sequences}')
        print(f'  Saved to                  : {self.output_csv}')

def main(argv: list[str] | None = None) -> None:
    SequenceRun(SequenceConfig.from_args(argv)).run()


if __name__ == '__main__':
    main()
