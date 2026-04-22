from typing import  Dict, List, Union
import torch
import torch.utils.checkpoint
import numpy as np
from dataclasses import dataclass
from typing import Optional
from transformers import Wav2Vec2FeatureExtractor
import julius

@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs for self-supervised pretraining.
    This data collator is to be used with Datasets saved in the Datasets spirit, in the pyarrow format.

    Important: We assume that Wav2vec2FeatureExtractor was all-ready applied once on the dataset
    (i.e the dataset is allready preprocessed: normalize (if set) and with a sampling rate of 16000 Hz)
    This should be done when the dataset is saved in the pyarrow format that is used by HuggingFace!

    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            truncation= True, ## Added Pierre, 03/06/2024
            max_length= int(20* self.feature_extractor.sampling_rate),
        )
        return batch

def get_collator_Pretraining(file_configPreprocessor):
    feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(file_configPreprocessor)
    data_collator = DataCollatorForWav2Vec2Pretraining(feature_extractor)
    return  data_collator


@dataclass
class DataCollatorForWav2Vec2Pretraining_withPreprocesing:
    """
    A DataCollator that can be used with soundmat type dataset.
    It preprocesses by normalizing the data, truncating too long sequences and padding.

    23/08/2023: Changed Truncation to False to allow for longer sequences
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # No safe-checking on the sampling rate, make sure it is 16000 before...

        batch = self.feature_extractor(
            raw_speech = [f["input_values"] for f in features],
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",

            # Truncation parameters for the sounds!
            truncation= False,
            # max_length = int(20* self.feature_extractor.sampling_rate),
            # min_length = int(2*self.feature_extractor.sampling_rate),

            # return_attention_mask=True,
            sampling_rate = 16000
        )

        return batch

def get_collator_withPreprocessing(file_configPreprocessor):
    feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(file_configPreprocessor)
    data_collator = DataCollatorForWav2Vec2Pretraining_withPreprocesing(feature_extractor)
    return  data_collator




@dataclass
class DataCollatorForWav2Vec2Pretraining_resampling:
    """
    A DataCollator that can be used with soundmat type dataset.
    It preprocesses by normalizing the data, truncating too long sequences and padding.

    (tmp): Collator for AnimalSpeak dataset
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # No safe-checking on the sampling rate, make sure it is 16000 before...

        all_sr = [f["audio"]["sampling_rate"] for f in features]
        sr = np.unique(all_sr)
        [print(s) for s in sr]
        resamplers = {s:julius.ResampleFrac(old_sr=s,new_sr=self.feature_extractor.sampling_rate) for s in sr}
        audio_resampled = [resamplers(f["audio"]["sampling_rate"])(f["audio"]["array"]) for f in features]
        
        batch = self.feature_extractor(
            raw_speech =audio_resampled,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            # Truncation parameters for the sounds!
            truncation= False,
            # max_length = int(20* self.feature_extractor.sampling_rate),
            # min_length = int(2*self.feature_extractor.sampling_rate),
            # return_attention_mask=True,
            sampling_rate = self.feature_extractor.sampling_rate, #we have resampled so we can safely say that all audio files are at the good sampling rate.
        )

        return batch

def get_collator_withPreprocessing_resampling(file_configPreprocessor,config):
    feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(file_configPreprocessor)
    data_collator = DataCollatorForWav2Vec2Pretraining_resampling(feature_extractor)
    return  data_collator


# def _downsample_output(x,nb_downsample):
#     x = x.transpose(-2, -1)  # repermute to (layer, neuron, time) or (layer,batch,neuron,time)
#     ## efficient downsampling mean computing:
#     # unravel the array  the target sampling rate, removing the last elements.
#     # use the mean over all remaining elements for the last:
#     size_window = x.shape[-1] // nb_downsample
#     to_remove = x.shape[-1] % nb_downsample
#     y = x[..., :(x.shape[-1] - to_remove)].reshape(x.shape[:-1] + (nb_downsample, size_window))
#     y_downsampled = torch.mean(y, dim=-1)
#     last_window = torch.sum(x[..., x.shape[-1] - to_remove:], dim=-1)
#     y_downsampled[..., -1] = (y_downsampled[..., -1] * size_window + last_window) / (to_remove + size_window)

#     return y_downsampled

