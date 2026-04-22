# Tools to preprocess a repository of .wav files and generate a dataset
# that will then be uploaded to Huggingface
import datasets
from datasets import Dataset, Features, Audio
import numpy as np
import os
from transformers import Wav2Vec2FeatureExtractor
from pathlib import Path
from typing import Union

def read_audio_dataset(dir_data: Union[Path,str],dir_package : Union[Path,str]) -> Dataset:
    ### pre-process the fma dataset into a pyarrow dataset
    # that we can then upload to huggingface.
    folders = [dir_data]

    # unroll the folders:
    def rec_search(f):
        d = os.listdir(f)
        res = []
        for o in d:
            if os.path.isdir(os.path.join(f,o)):
                res += [rec_search(os.path.join(f,o))]
            else:
                res += [np.array([os.path.join(f,o)])]
        return np.concatenate([r for r in res])
    folder_list = np.concatenate([rec_search(f) for f in folders])

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Path(dir_package) / "ANNpretraining"
                                                                 / "models" / "wav2vec2" / "config" /"preprocessor_config.json")

    # filter by subsets and extract all audio files
    audio_list = list(filter(lambda e: e.endswith(".mp3") or e.endswith(".wav"), folder_list))
    features = Features({"audio": Audio(sampling_rate=feature_extractor.sampling_rate)})
    ds =  Dataset.from_dict({"audio": audio_list} , features=features)

    # set max & min audio length in number of samples
    max_length = int(20 * feature_extractor.sampling_rate)
    min_length = int(2 * feature_extractor.sampling_rate)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    bad_indices = np.zeros(len(ds),dtype=bool)
    def prepare_dataset(batch,idx):
        try:
            sample = batch["audio"]
            inputs = feature_extractor(
                sample["array"], sampling_rate=sample["sampling_rate"], max_length=max_length, truncation=True
            )
            batch["input_values"] = np.array(inputs.input_values[0],dtype=np.float32)
            batch["input_length"] = len(inputs.input_values[0])

            return batch
        except:
            # loading error,we save he indice and will remove it after when filtering my minimal length
            bad_indices[idx] = True
            batch["input_values"] = np.array([0],dtype=np.float32)
            batch["input_length"] = 1
            return batch

    vectorized_datasets = ds.map(
        prepare_dataset,
        remove_columns=ds.column_names,
        with_indices = True
    )
    if min_length > 0.0:
        vectorized_datasets = vectorized_datasets.filter(
            lambda x: x > min_length,
            input_columns=["input_length"],
        )
        vectorized_datasets = vectorized_datasets.remove_columns("input_length")

    return vectorized_datasets



dir_data = Path("/media/pierre/NeuroData2/datasets/nsd/sounds")
dir_package = Path("/home/pierre/Documents/Wav2vec2Pretraining/")
save_path = Path("/media/pierre/NeuroData2/datasets/nsd/asHf")

ds_train = read_audio_dataset(dir_data / "train",dir_package)
ds_valid = read_audio_dataset(dir_data / "validation",dir_package)
ds_all = datasets.DatasetDict({"train":ds_train,"validation":ds_valid})
ds_all.save_to_disk(save_path)
