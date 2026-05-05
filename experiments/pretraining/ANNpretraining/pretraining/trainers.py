from transformers.trainer import *
### We modify the trainer to be able to log everythin
import os
from torch.nn.parallel import DistributedDataParallel
from ANNpretraining.models.api import forPretraining

class PretrainingTrainer(Trainer):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        if self.state.is_world_process_zero or self.args.local_rank==-1:
            self.path_csv = os.path.join(self.args.output_dir,"losses.csv")
            # self.zg = zr.open_group(os.path.join(self.args.output_dir,"losses.zarr"),mode="a")

    def compute_loss(self, model: forPretraining, inputs, return_outputs=False,num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.state.is_world_process_zero or self.args.local_rank==-1:
            try:
                model.save_loss(outputs,self.path_csv)
            except:
                try:
                    model.module.save_loss(outputs,self.path_csv)
                except:
                    model.save_loss(outputs,self.path_csv)
        return (loss, outputs) if return_outputs else loss
