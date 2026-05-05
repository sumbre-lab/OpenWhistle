import numpy as np
from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import DefaultFlowCallback

class LogarithmicStepSaveCallback(DefaultFlowCallback):
    def __init__(self,output_dir,schedule=None):
        super().__init__()
        if  schedule is None:
            self.schedule = np.concatenate([np.arange(100,10000,step=100),np.arange(10000,401000,step=1000)]) #[1],np.arange(0,100,step=1)[1:],
        else:
            self.schedule = schedule
        # self.schedule = np.concatenate([np.array([50]),np.arange(100,20000,step=200),np.arange(20000,800000,step=2000)]) #[1],np.arange(0,100,step=1)[1:],

        ## Not taken into account:
        ## We continously monitor if the optimizer was run and correct for the saving step knowing that
        # so that each model is still saved at the same set of optimization steps.
        # self.zg = zr.open_group(os.path.join(output_dir,"losses.zarr"),mode="r")
        # self.last_save = -1
    #
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        # if "optimizer_was_run" in self.zg.keys():
        #     completed_steps = np.sum(self.zg["optimizer_was_run"][:state.global_step])
        #     #state.global_step - np.sum(np.logical_not(self.zg[:state.global_step]))
        #
        #     if (completed_steps in self.schedule and completed_steps!=self.last_save) or state.global_step==args.max_steps:
        #         # We save on the saving schedule and the last network
        #         control.should_save = True
        #         self.last_save = completed_steps
        #         return control
        #     else:
        #         control.should_save = False
        #         return control
        # else:

        ## We never want to evaluate during pretraining for the sake of speed:
        control.should_evaluate = False

        completed_steps = state.global_step
        if completed_steps in self.schedule:
            control.should_save = True
            return control
        else:
            control.should_save = False
            return control
