import torch.distributed
import transformers
import os
from transformers import  HfArgumentParser, TrainingArguments
import datasets

from ANNpretraining.pretraining.trainers import pierreTrainer
from ANNpretraining.pretraining.checkpoint_utils import checkpoint_loading
from ANNpretraining.pretraining.callbacks import LogarithmicStepSaveCallback,AllStepSaveCallback

from ANNpretraining.models.wav2vec2.utils_data import get_collator_Pretraining
from ANNpretraining.models import IMPLEMENTED_MODELS
import numpy as np


def model_load(modelType:str,path_model:str):
    os.environ["WANDB_DISABLED"] = "true"

    if path_model.endswith(".json"): # .json config
        config = IMPLEMENTED_MODELS[modelType].load_config(path_model)
        model = IMPLEMENTED_MODELS[modelType](config)
    else: # directory with known model weights.
        model = IMPLEMENTED_MODELS[modelType].from_pretrained(path_model)
        config = model.config
    return model,config


def parseHf(path_train_arg,output_dir):
    hfArg = HfArgumentParser(TrainingArguments)
    hfArg_out = hfArg.parse_yaml_file(path_train_arg, True)[0]
    # additional parameter for this training:
    HfArgumentParser.__setattr__(hfArg_out,"output_dir",str(output_dir))
    # hfArg_out.path_target_model = path_target_model

    if "LOCAL_RANK" in os.environ.keys():
        local_rank = int(os.environ["LOCAL_RANK"])
        HfArgumentParser.__setattr__(hfArg_out, "local_rank", local_rank)

        torch.cuda.set_device(local_rank)
        print("torch is initialied",torch.distributed.is_initialized())
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl",
                                                 init_method="env://")

            if not os.path.exists(hfArg_out.output_dir):
                os.makedirs(hfArg_out.output_dir)
        print("starting the Trainer of hugginface")
    transformers.logging.set_verbosity_info()
    last_checkpoint = checkpoint_loading(hfArg_out)

    return hfArg_out,last_checkpoint

def train(modelType : str, path_model : str, output_dir :str,
                              path_preprocessor : str, path_train_arg: str,
                              preprocess : bool ,
                              path_data : str):

    print("starting loading the data")
    ds_processed = datasets.load_from_disk(path_data)
    print("finished loading the data")

    model,config = model_load(modelType,path_model)
    data_collator = IMPLEMENTED_MODELS[modelType].get_collator_pretraining(file_configPreprocessor=path_preprocessor,
                                                    config = config)
    hfArg_out,last_checkpoint = parseHf(path_train_arg,output_dir)
    if last_checkpoint != False:
        del model
        # make sure the model is loaded from the checkpoint
        model = IMPLEMENTED_MODELS[modelType].from_pretrained(last_checkpoint)
        model.train()

    ds_processed["train"] = ds_processed["train"]
    if "validation" in ds_processed.keys():
        ds_processed["validation"] = ds_processed["validation"].select(range(hfArg_out.eval_batch_size))
    else:
        ds_processed["validation"] = ds_processed["train"].select(range(hfArg_out.eval_batch_size))

    print(hfArg_out.eval_batch_size)
    # Remove shuffling as it is done by the RandomSampler of the dataloader
    # ds_processed = ds_processed.shuffle()

    print("starting the Trainer")
    trainer = pierreTrainer(model=model,
            args=hfArg_out,
            data_collator= data_collator,
            train_dataset=ds_processed["train"],
            eval_dataset=ds_processed["validation"])
    
    trainer.add_callback(LogarithmicStepSaveCallback(output_dir))

    if hfArg_out.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    print("waiting for all other nodes")
    torch.distributed.barrier()
    return None

def _clean_store(output_dir):
    # After the training is completed we remove the optimisers
    # from the store to save memory space. (For Wav2vec2 Optimiser is two time the size of the initial model)
    # If the gradient had been of interest we could re-run an optimisation on it.
    for chk in os.listdir(output_dir):
        if chk.__contains__("checkpoint-"):
            if os.path.exists(os.path.join(output_dir,chk,"optimizer.pt")):
                os.remove(os.path.join(output_dir,chk,"optimizer.pt"))
                os.remove(os.path.join(output_dir, chk, "scaler.pt"))
                os.remove(os.path.join(output_dir, chk, "scheduler.pt"))
                os.remove(os.path.join(output_dir, chk, "trainer_state.json"))
                os.remove(os.path.join(output_dir, chk, "training_args.bin"))

