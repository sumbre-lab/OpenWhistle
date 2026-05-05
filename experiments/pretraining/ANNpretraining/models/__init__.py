from typing import Dict, Type

from ANNpretraining.models.api import forPretraining
from ANNpretraining.models.wav2vec2.forPreTraining import Wav2vec2ForPreTraining_randommask


def _getimplementedmodels() -> Dict[str, Type[forPretraining]]:
    return {"wav2vec2": Wav2vec2ForPreTraining_randommask}


IMPLEMENTED_MODELS = _getimplementedmodels()
