import argparse
from pathlib import Path

from inference import InferenceConfig, InferenceRun, parse_optional_int
from inference_conf import dataset_names, resolve_dataset
from utils.model import SpectrogramConfig


def parse_args() -> argparse.Namespace:
    defaults = InferenceConfig()
    parser = argparse.ArgumentParser(
        description='Run cnn/inference.py presets on external OpenWhistle datasets.'
    )
    parser.add_argument(
        'datasets',
        nargs='+',
        choices=dataset_names(),
        help='External dataset preset(s) to run.',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Root output directory. Defaults to cnn/runs/external_inference.',
    )
    parser.add_argument(
        '--checkpoint-path',
        type=Path,
        default=defaults.checkpoint_path,
        help='Local checkpoint override. Defaults to the Hugging Face model.',
    )
    parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
    parser.add_argument('--threshold', type=float, default=defaults.threshold)
    parser.add_argument('--start-time', type=float, default=defaults.start_time)
    parser.add_argument('--end-time', type=float, default=defaults.end_time)
    parser.add_argument('--limit', type=int, default=defaults.limit)
    parser.add_argument('--cpu-only', action='store_true', default=defaults.cpu_only)
    parser.add_argument(
        '--save-positive-spectrograms',
        action='store_true',
        default=defaults.save_positive_spectrograms,
    )
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
    return parser.parse_args()


def build_config(
    dataset_name: str,
    args: argparse.Namespace,
) -> InferenceConfig:
    dataset = resolve_dataset(dataset_name, output_root=args.output_root)
    print(f'Dataset     : {dataset.name}')
    print(f'Source dir  : {dataset.source_dir}')
    print(f'Recordings  : {dataset.recordings_dir}')
    print(f'Output dir  : {dataset.output_dir}')
    if dataset.specific_files_path is not None:
        print(f'File list   : {dataset.specific_files_path}')
    print(f'Recursive   : {dataset.recursive}')

    return InferenceConfig(
        checkpoint_path=args.checkpoint_path,
        recordings_dir=dataset.recordings_dir,
        output_dir=dataset.output_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        start_time=args.start_time,
        end_time=args.end_time,
        save_positive_spectrograms=args.save_positive_spectrograms,
        specific_files_path=dataset.specific_files_path,
        recursive=dataset.recursive,
        limit=args.limit,
        cpu_only=args.cpu_only,
        spectrogram_config=SpectrogramConfig(
            cut_low_frequency=args.cut_low_frequency,
            cut_high_frequency=args.cut_high_frequency,
            target_fs=args.target_fs,
        ),
    )


def main() -> None:
    args = parse_args()
    for index, dataset_name in enumerate(args.datasets):
        if index:
            print('====================================================')
        config = build_config(dataset_name, args)
        InferenceRun(config).run()


if __name__ == '__main__':
    main()
