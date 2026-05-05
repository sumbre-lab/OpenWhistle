import numpy as np
from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import DefaultFlowCallback

class LogarithmicStepSaveCallback(DefaultFlowCallback):
    def __init__(self,output_dir,schedule=None):
        super().__init__()
        if  schedule is None:
            self.schedule = np.concatenate([np.arange(100,10000,step=100),np.arange(10000,401000,step=1000)])
        else:
            self.schedule = schedule

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_evaluate = False

        completed_steps = state.global_step
        if completed_steps in self.schedule:
            control.should_save = True
            return control
        else:
            control.should_save = False
            return control
