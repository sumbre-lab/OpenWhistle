# Tools to preprocess a repository of audio files and generate a dataset
# that can then be uploaded to Hugging Face.
import argparse
import os
from pathlib import Path
from typing import Union

import datasets
import numpy as np
from datasets import Audio, Dataset, Features
from transformers import Wav2Vec2FeatureExtractor

from ANNpretraining.runtime import ensure_dir


def read_audio_dataset(
    dir_data: Union[Path, str],
    feature_extractor: Wav2Vec2FeatureExtractor,
) -> Dataset:
    folders = [str(dir_data)]

    def rec_search(folder: str):
        entries = os.listdir(folder)
        results = []
        for entry in entries:
            path = os.path.join(folder, entry)
            if os.path.isdir(path):
                results.append(rec_search(path))
            else:
                results.append(np.array([path]))
        return np.concatenate(results) if results else np.array([])

    folder_list = np.concatenate([rec_search(folder) for folder in folders])

    audio_list = [path for path in folder_list if path.endswith(".mp3") or path.endswith(".wav")]
    features = Features({"audio": Audio(sampling_rate=feature_extractor.sampling_rate)})
    ds = Dataset.from_dict({"audio": audio_list}, features=features)

    max_length = int(20 * feature_extractor.sampling_rate)
    min_length = int(2 * feature_extractor.sampling_rate)

    def prepare_dataset(batch, idx):
        try:
            sample = batch["audio"]
            inputs = feature_extractor(
                sample["array"],
                sampling_rate=sample["sampling_rate"],
                max_length=max_length,
                truncation=True,
            )
            batch["input_values"] = np.array(inputs.input_values[0], dtype=np.float32)
            batch["input_length"] = len(inputs.input_values[0])
        except Exception:
            batch["input_values"] = np.array([0], dtype=np.float32)
            batch["input_length"] = 1
        return batch

    vectorized_datasets = ds.map(
        prepare_dataset,
        remove_columns=ds.column_names,
        with_indices=True,
    )
    if min_length > 0.0:
        vectorized_datasets = vectorized_datasets.filter(
            lambda x: x > min_length,
            input_columns=["input_length"],
        )
        vectorized_datasets = vectorized_datasets.remove_columns("input_length")

    return vectorized_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a save_to_disk dataset from audio folders.")
    parser.add_argument("--train_dir", required=True, help="Directory containing the training audio files.")
    parser.add_argument("--validation_dir", required=True, help="Directory containing the validation audio files.")
    parser.add_argument("--output_dir", required=True, help="Where to save the generated DatasetDict.")
    parser.add_argument(
        "--preprocessor_path",
        default=str(
            Path(__file__).resolve().parents[1]
            / "models"
            / "wav2vec2"
            / "config"
            / "preprocessor_dolphin.json"
        ),
        help="Path to the Wav2Vec2 feature extractor config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(args.preprocessor_path)
    output_dir = ensure_dir(args.output_dir)

    ds_train = read_audio_dataset(args.train_dir, feature_extractor)
    ds_valid = read_audio_dataset(args.validation_dir, feature_extractor)
    ds_all = datasets.DatasetDict({"train": ds_train, "validation": ds_valid})
    ds_all.save_to_disk(str(output_dir))


if __name__ == "__main__":
    main()
