from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import julius
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """Pad already-preprocessed ``input_values`` for self-supervised pretraining."""

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            truncation=True,
            max_length=int(20 * self.feature_extractor.sampling_rate),
        )
        return batch


def get_collator_Pretraining(file_configPreprocessor):
    feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(file_configPreprocessor)
    return DataCollatorForWav2Vec2Pretraining(feature_extractor)


@dataclass
class DataCollatorForWav2Vec2Pretraining_withPreprocesing:
    """Prepare raw audio rows and pad them for Wav2Vec2 pretraining."""

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if len(features) == 0:
            raise ValueError("Cannot collate an empty batch.")

        if "audio" in features[0]:
            all_sr = [f["audio"]["sampling_rate"] for f in features]
            sr = np.unique(all_sr)
            if len(sr) != 1:
                resamplers = {
                    int(s): julius.ResampleFrac(
                        old_sr=int(s),
                        new_sr=self.feature_extractor.sampling_rate,
                    )
                    for s in sr
                }
                raw_speech = [
                    resamplers[int(f["audio"]["sampling_rate"])](
                        torch.as_tensor(f["audio"]["array"])
                    ).cpu().numpy()
                    for f in features
                ]
                sampling_rate = self.feature_extractor.sampling_rate
            else:
                sampling_rate = int(sr[0])
                raw_speech = [f["audio"]["array"] for f in features]
        else:
            raw_speech = [f["input_values"] for f in features]
            sampling_rate = self.feature_extractor.sampling_rate

        batch = self.feature_extractor(
            raw_speech=raw_speech,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            truncation=False,
            sampling_rate=sampling_rate,
        )

        return batch


def get_collator_withPreprocessing(file_configPreprocessor):
    feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(file_configPreprocessor)
    return DataCollatorForWav2Vec2Pretraining_withPreprocesing(feature_extractor)
