import torch.distributed
import transformers
import os
import json
from pathlib import Path
from transformers import HfArgumentParser, TrainingArguments
import datasets

from ANNpretraining.pretraining.trainers import PretrainingTrainer
from ANNpretraining.pretraining.checkpoint_utils import checkpoint_loading
from ANNpretraining.pretraining.callbacks import LogarithmicStepSaveCallback
from ANNpretraining.models import IMPLEMENTED_MODELS


def model_load(modelType:str,path_model:str):
    os.environ["WANDB_DISABLED"] = "true"

    if str(path_model).endswith(".json"):
        config = IMPLEMENTED_MODELS[modelType].load_config(path_model)
        model = IMPLEMENTED_MODELS[modelType](config)
    else:
        model = IMPLEMENTED_MODELS[modelType].from_pretrained(path_model)
        config = model.config
    return model,config


def parseHf(path_train_arg,output_dir):
    hfArg = HfArgumentParser(TrainingArguments)
    hfArg_out = hfArg.parse_yaml_file(str(path_train_arg), True)[0]
    HfArgumentParser.__setattr__(hfArg_out,"output_dir",str(output_dir))

    if "LOCAL_RANK" in os.environ.keys():
        local_rank = int(os.environ["LOCAL_RANK"])
        HfArgumentParser.__setattr__(hfArg_out, "local_rank", local_rank)

        torch.cuda.set_device(local_rank)
        print("torch is initialized",torch.distributed.is_initialized())
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl",
                                                 init_method="env://")

            if not os.path.exists(hfArg_out.output_dir):
                os.makedirs(hfArg_out.output_dir)
        print("starting the Hugging Face Trainer")
    transformers.logging.set_verbosity_info()
    last_checkpoint = checkpoint_loading(hfArg_out)

    return hfArg_out,last_checkpoint


def _target_sampling_rate(path_preprocessor: str) -> int:
    with open(path_preprocessor, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return int(config["sampling_rate"])


def _load_training_dataset(path_data: str, path_data_config: str | None, target_sampling_rate: int):
    path_like = Path(path_data)
    if path_like.exists():
        print(f"loading local dataset from disk: {path_like}")
        ds_processed = datasets.load_from_disk(str(path_like))
    else:
        print(f"loading Hugging Face dataset: repo_id={path_data} config={path_data_config!r}")
        if path_data_config:
            ds_processed = datasets.load_dataset(path_data, name=path_data_config, trust_remote_code=True)
        else:
            ds_processed = datasets.load_dataset(path_data, trust_remote_code=True)

    if not isinstance(ds_processed, datasets.DatasetDict):
        ds_processed = datasets.DatasetDict({"train": ds_processed})

    for split_name in ds_processed.keys():
        if "audio" in ds_processed[split_name].column_names:
            ds_processed[split_name] = ds_processed[split_name].cast_column(
                "audio",
                datasets.Audio(sampling_rate=target_sampling_rate),
            )

    print(f"loaded splits: {list(ds_processed.keys())}")
    for split_name in ds_processed.keys():
        print(f"  {split_name}: columns={ds_processed[split_name].column_names}")

    return ds_processed

def train(modelType : str, path_model : str, output_dir :str,
                              path_preprocessor : str, path_train_arg: str,
                              preprocess : bool ,
                              path_data : str,
                              path_data_config : str | None = None):

    print("starting loading the data")
    ds_processed = _load_training_dataset(
        path_data=path_data,
        path_data_config=path_data_config,
        target_sampling_rate=_target_sampling_rate(path_preprocessor),
    )
    print("finished loading the data")

    model,config = model_load(modelType,path_model)
    train_columns = set(ds_processed["train"].column_names)
    uses_audio_column = "audio" in train_columns
    uses_preprocessed_pyarrow = "input_values" in train_columns and not preprocess
    if not uses_audio_column and "input_values" not in train_columns:
        raise ValueError(
            "Unsupported training dataset columns. Expected either an 'audio' column "
            f"or an 'input_values' column, got: {sorted(train_columns)}"
        )
    data_collator = IMPLEMENTED_MODELS[modelType].get_collator_pretraining(file_configPreprocessor=path_preprocessor,
                                                    config = config,
                                                    wasPreprocessedPyarrow=uses_preprocessed_pyarrow)
    hfArg_out,last_checkpoint = parseHf(path_train_arg,output_dir)
    if uses_audio_column:
        hfArg_out.remove_unused_columns = False
    if last_checkpoint != False:
        del model
        model = IMPLEMENTED_MODELS[modelType].from_pretrained(last_checkpoint)
        model.train()

    if "validation" in ds_processed.keys():
        eval_size = min(len(ds_processed["validation"]), hfArg_out.eval_batch_size)
        ds_processed["validation"] = ds_processed["validation"].select(range(eval_size))
    else:
        eval_size = min(len(ds_processed["train"]), hfArg_out.eval_batch_size)
        ds_processed["validation"] = ds_processed["train"].select(range(eval_size))

    print(hfArg_out.eval_batch_size)

    print("starting the Trainer")
    trainer = PretrainingTrainer(model=model,
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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return None

def _clean_store(output_dir):
    """Remove optimizer state from checkpoints to reduce storage after training."""
    for chk in os.listdir(output_dir):
        if chk.__contains__("checkpoint-"):
            if os.path.exists(os.path.join(output_dir,chk,"optimizer.pt")):
                os.remove(os.path.join(output_dir,chk,"optimizer.pt"))
                os.remove(os.path.join(output_dir, chk, "scaler.pt"))
                os.remove(os.path.join(output_dir, chk, "scheduler.pt"))
                os.remove(os.path.join(output_dir, chk, "trainer_state.json"))
                os.remove(os.path.join(output_dir, chk, "training_args.bin"))
