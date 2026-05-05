from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from datasets import DatasetDict, IterableDatasetDict

class forPretraining(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self,input_values):
        pass

    @abstractmethod
    def save_loss(self,outputs : dict,csv_path:Union[str,Path]):
        pass
    @abstractmethod
    def get_mainloss_name(cls):
        pass

    @abstractmethod
    def load_config(cls,path : Union[str,Path]):
        pass

    @abstractmethod
    def from_pretrained(cls,path : Union[str,Path]):
        pass

    @abstractmethod
    def preprocessor_from_pretrained(self,path : Union[str,Path]):
        pass

    @abstractmethod
    def get_collator_pretraining(cls,file_configPreprocessor : Union[str,Path],config):
        # Given the configuration of the preprocessor, this methods should return the DataCollator
        # that is used during Training. As a reminder, the DataCollator batches several dict into a single dict.
        # which is then filtered of unused elements in the forward pass of the model and sent to the model as
        # argument to the forward function.
        pass

    @abstractmethod
    def pretransform_dataset(cls, ds: Union[DatasetDict,IterableDatasetDict],path: Union[str,Path]) -> Union[DatasetDict,IterableDatasetDict]:
        ## Some network needs to perform aditional pre-processing of the dataset
        # which we allow to be done in streaming by returning an IterableDatasetDict.
        pass
