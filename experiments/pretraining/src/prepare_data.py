#!/usr/bin/env python3
"""Prepare OpenWhistle Hugging Face audio for fairseq wav2vec pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import soundfile as sf


DATASET_NAME = "dolphinteam/OpenWhistle-Pretraining"
RUNS_DIR = Path(__file__).resolve().parents[3] / "runs/pretraining"


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--dataset_config", dest="config_name", default="default")
    parser.add_argument(
        "--revision",
        help="Dataset commit; defaults to resolving the current commit once.",
    )
    parser.add_argument(
        "--train_fraction", type=float, choices=[0.1, 0.5, 1.0], default=1.0
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio_dir", type=Path, default=RUNS_DIR / "data/audio")
    parser.add_argument(
        "--manifest_dir", type=Path, default=RUNS_DIR / "data/manifests"
    )
    parser.add_argument("--target_sample_rate", default=44100, type=positive_int)
    parser.add_argument("--audio_format", choices=["flac", "wav"], default="flac")
    parser.add_argument("--max_train", type=positive_int)
    parser.add_argument("--max_validation", type=positive_int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_split(args: argparse.Namespace, split: str):
    from datasets import Audio, load_dataset

    dataset = load_dataset(
        args.dataset_name,
        args.config_name,
        split=split,
        streaming=args.streaming,
        revision=args.revision,
    )
    return dataset.cast_column(
        "audio",
        Audio(sampling_rate=args.target_sample_rate, mono=True),
    )


def audio_array(example: dict) -> np.ndarray:
    array = np.asarray(example["audio"]["array"], dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Decoded audio must be a nonempty, finite mono waveform.")
    return array


def selected(index: int, seed: int, fraction: float) -> bool:
    # Same seed/revision/ordering gives nested 10/50/100% training subsets.
    digest = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < fraction


def write_audio(
    path: Path, array: np.ndarray, sample_rate: int, audio_format: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        sf.write(
            temporary, array, sample_rate, format=audio_format.upper(), subtype="PCM_16"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_split(
    args: argparse.Namespace, hf_split: str, manifest_name: str, max_rows: int | None
) -> None:
    manifest_path = args.manifest_dir / manifest_name
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}; choose another directory or --overwrite."
        )
    dataset = iter(enumerate(load_split(args, hf_split)))
    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    total_samples = 0
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.manifest_dir,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as manifest:
        temporary_manifest = Path(manifest.name)
        try:
            manifest.write(str(args.audio_dir.resolve()) + "\n")
            while max_rows is None or count < max_rows:
                try:
                    index, example = next(dataset)
                except StopIteration:
                    break
                if hf_split == "train" and not selected(
                    index, args.seed, args.train_fraction
                ):
                    continue

                array = audio_array(example)
                if example["audio"]["sampling_rate"] != args.target_sample_rate:
                    raise ValueError(
                        "Decoded sampling rate differs from the requested rate."
                    )
                rel_path = Path(hf_split) / f"{index:08d}.{args.audio_format}"
                out_path = args.audio_dir / rel_path
                reused = out_path.exists() and not args.overwrite
                if not reused:
                    write_audio(
                        out_path, array, args.target_sample_rate, args.audio_format
                    )

                info = sf.info(out_path)
                if (info.frames, info.samplerate, info.channels) != (
                    len(array),
                    args.target_sample_rate,
                    1,
                ):
                    raise ValueError(
                        f"Existing audio does not match the dataset: {out_path}"
                    )
                if reused:
                    decoded, _ = sf.read(out_path, dtype="float32")
                    # PCM_16 rounds/clips samples. Header equality alone cannot
                    # distinguish recordings from another dataset revision.
                    if not np.allclose(
                        decoded, np.clip(array, -1, 1), rtol=0, atol=1 / 32768 + 1e-7
                    ):
                        raise ValueError(
                            f"Existing audio content does not match the dataset: {out_path}"
                        )

                num_samples = int(array.shape[0])
                manifest.write(f"{rel_path.as_posix()}\t{num_samples}\n")
                total_samples += num_samples
                count += 1
                if count % 100 == 0:
                    hours = total_samples / args.target_sample_rate / 3600
                    print(f"{hf_split}: wrote {count} examples ({hours:.2f} h)")
            if count == 0:
                raise ValueError(f"No examples selected for {hf_split}.")
            manifest.close()
            temporary_manifest.replace(manifest_path)
        finally:
            temporary_manifest.unlink(missing_ok=True)

    metadata = {
        "dataset_name": args.dataset_name,
        "config_name": args.config_name,
        "dataset_revision": args.revision,
        "train_fraction": args.train_fraction,
        "subset_seed": args.seed,
        "split": hf_split,
        "manifest": str(manifest_path),
        "audio_root": str(args.audio_dir.resolve()),
        "target_sample_rate": args.target_sample_rate,
        "audio_format": args.audio_format,
        "num_examples": count,
        "num_samples": total_samples,
        "duration_hours": total_samples / args.target_sample_rate / 3600,
    }
    metadata_path = args.manifest_dir / f"{manifest_path.stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{hf_split}: done -> {manifest_path}")


def main() -> None:
    args = parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Pretraining requires the Python 3.10 environment and the pinned "
            "dependencies in experiments/pretraining/requirements.txt. "
            "Activate that environment before preparing audio."
        )
    for name in ("train.tsv", "valid.tsv"):
        if (args.manifest_dir / name).exists() and not args.overwrite:
            raise FileExistsError(
                f"Manifest already exists: {args.manifest_dir / name}; choose another directory or --overwrite."
            )
    from huggingface_hub import HfApi

    args.revision = HfApi().dataset_info(args.dataset_name, revision=args.revision).sha
    prepare_split(args, "train", "train.tsv", args.max_train)
    prepare_split(args, "validation", "valid.tsv", args.max_validation)


if __name__ == "__main__":
    main()
