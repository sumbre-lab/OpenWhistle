from transformers.trainer import *
import os
from ANNpretraining.models.api import forPretraining

class PretrainingTrainer(Trainer):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        if self.state.is_world_process_zero or self.args.local_rank==-1:
            self.path_csv = os.path.join(self.args.output_dir,"losses.csv")

    def compute_loss(self, model: forPretraining, inputs, return_outputs=False,num_items_in_batch=None):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        
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
