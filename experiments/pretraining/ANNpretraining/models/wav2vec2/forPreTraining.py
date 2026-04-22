import numpy as np
import pandas as pd
import torch

from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining,Wav2Vec2Processor,Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput, _compute_mask_indices, _sample_negative_indices
import torch.nn as nn
from ANNpretraining.models.wav2vec2.with_grad_mult import pierreWav2Vec2ForPreTraining
from ANNpretraining.models.api import forPretraining

from ANNpretraining.models.wav2vec2.utils_data import get_collator_Pretraining,get_collator_withPreprocessing
import torch

from typing import Union
from pathlib import  Path
import os

class _Wav2vec2ForPretraining_saveload(pierreWav2Vec2ForPreTraining,forPretraining):
    def save_loss(self, outputs: dict,csv_path:Union[str,Path]): 
        df = pd.DataFrame({k:outputs[k].detach().cpu().numpy().reshape(-1) for k in ["loss", "contrastive_loss", "diversity_loss", "pen_loss"]})
        df.to_csv(csv_path,header=not os.path.exists(csv_path),mode="a")

    @classmethod
    def get_mainloss_name(self):
        return "contrastive_loss_notReduce"

    @classmethod
    def load_config(cls,path : Union[str,Path]):
        # Load the config from a json file.
        return Wav2Vec2Config.from_json_file(path)

    @classmethod
    def get_collator_pretraining(cls,file_configPreprocessor,config,wasPreprocessedPyarrow = False):
        if wasPreprocessedPyarrow:
            return get_collator_Pretraining(file_configPreprocessor)
        else:
            return get_collator_withPreprocessing(file_configPreprocessor)

    @classmethod
    def preprocessor_from_pretrained(cls,path):
        return Wav2Vec2FeatureExtractor.from_pretrained(path)

    @classmethod
    def pretransform_dataset(cls,ds,path):
        return ds


class Wav2vec2ForPreTraining_randommask(_Wav2vec2ForPretraining_saveload):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

    def forward(self,input_values):
        # input_values = torch.cat([input_values,input_values],dim=1) ## temporary to see if it has a large effect in running time

        batch_size, seq_len = input_values.shape
        latent_length = int(self._get_feat_extract_output_lengths(seq_len).detach().numpy())
        mask_time_indices = _compute_mask_indices((batch_size, latent_length),
                                                  mask_prob=self.config.mask_time_prob,
                                                  mask_length=self.config.mask_time_length,
                                                  min_masks=1)

        mask_indices_torch = torch.tensor(mask_time_indices, device=self.device,dtype=torch.bool)

        negative_indices = _sample_negative_indices((batch_size, latent_length),
                                                    num_negatives=self.config.num_negatives,
                                                    mask_time_indices=mask_time_indices)

        ## The negative indidces have to be a long tensor
        negative_indices_torch = torch.tensor(negative_indices, device=self.device,dtype=torch.long)
        # It is an error in the Huggingface implementation that this tensor should be boolean
        # On the other hand the mask_time_indices should be boolean, despite its name.

        return super().forward(input_values,
                               mask_time_indices=mask_indices_torch,
                               sampled_negative_indices=negative_indices_torch)
