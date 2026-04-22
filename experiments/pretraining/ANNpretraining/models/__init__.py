from .wav2vec2.forPreTraining import Wav2vec2ForPreTraining_randommask
from .Wav2vecDS import Wav2vecDSForPretraining_saveload
from .convnextV2.forPreTraining import ConvNextV2_classificationPretraining,ConvNextV2_classificationPretrainingExtendedHead
from typing import Dict,Type
from ANNpretraining.models.api import forPretraining,forPostAnalysis

def _getimplementedmodels() -> Dict[str,Type[forPretraining]]:
    IMPLEMENTED_MODELS = {"wav2vec2": Wav2vec2ForPreTraining_randommask,
                          "Wav2vecDS": Wav2vecDSForPretraining_saveload,
                          "convnextV2":ConvNextV2_classificationPretraining,
                          "convnextV2_Atto":ConvNextV2_classificationPretraining,
                          "convnextV2text":ConvNextV2_classificationPretrainingExtendedHead} #
    return IMPLEMENTED_MODELS
IMPLEMENTED_MODELS = _getimplementedmodels()