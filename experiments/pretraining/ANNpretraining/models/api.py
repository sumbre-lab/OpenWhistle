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
        pass

    @abstractmethod
    def pretransform_dataset(cls, ds: Union[DatasetDict,IterableDatasetDict],path: Union[str,Path]) -> Union[DatasetDict,IterableDatasetDict]:
        pass
