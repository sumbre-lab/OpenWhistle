from pathlib import Path
import os
import sys

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

existing_pythonpath = os.environ.get("PYTHONPATH")
pythonpath_entries = existing_pythonpath.split(os.pathsep) if existing_pythonpath else []
if str(PACKAGE_PARENT) not in pythonpath_entries:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(PACKAGE_PARENT), *pythonpath_entries]
    )

import submitit
from ANNpretraining.models import IMPLEMENTED_MODELS
from ANNpretraining.runtime import artifact_dir, ensure_dir, submitit_parameters
import argparse

DEFAULT_SLURM_PARTITION = os.environ.get("OPENWHISTLE_SLURM_PARTITION", "")
DEFAULT_SLURM_ACCOUNT = os.environ.get("OPENWHISTLE_SLURM_ACCOUNT", "")

def get_parser():
    parser = argparse.ArgumentParser(description='Launch the training loop')
    parser.add_argument('--path_data', type=str,
                        default="OpenWhistleNeurIPS26/OpenWhistle-Pretraining",
                        help='local dataset path or Hugging Face dataset id')
    parser.add_argument('--path_data_config', type=str,
                        default="default",
                        help='Hugging Face dataset config name, ignored for local datasets')
    parser.add_argument('--output_dir', type=str,
                        default="",
                        help='path where the model is output')
    parser.add_argument('--log_dir', type=str,
                        default="",
                        help='directory used by submitit for job logs')
    parser.add_argument('--preprocess', type=bool,
                        default=False,
                        help='sets to true if the dataset is not'
                             ' preprocessed and comes as a matrix of sounds instead of a pyarrow datasets.'
                             ' If False we will just pad the sequences when batching, masking will be done on the fly '
                             'by the model forward pass ')
    parser.add_argument('--modelType',type=str,
                        default="wav2vec2")
    parser.add_argument('--path_model', type=str,
                        default=os.path.join("config","configTiny.json"),
                        help='path to the model config file (.json)')
    parser.add_argument('--path_preprocessor', type=str,
                        default=os.path.join("config","preprocessor_config.json"),
                        help='path to the preprocessor config file (.json)')

    parser.add_argument('--path_train_arg', type=str,
                        default=os.path.join("config", "trainingArg.yaml"),
                        help='path to the training argument for huggingface')
    parser.add_argument('--nb_gpu',type=int,default=8,help="nb gpu per node")
    parser.add_argument('--nb_nodes',type=int,default=8,help="nb nodes")
    parser.add_argument('--cpus_per_task', type=int, default=10, help='number of CPU cores per task')
    parser.add_argument('--timeout_min', type=int, default=10 * 60, help='job timeout in minutes')
    parser.add_argument('--slurm_partition', type=str, default=DEFAULT_SLURM_PARTITION,
                        help='optional SLURM partition')
    parser.add_argument('--slurm_account', type=str, default=DEFAULT_SLURM_ACCOUNT,
                        help='optional SLURM account')
    return parser

def main(args):
    parser = get_parser()
    args_class = parser.parse_args(args)

    package_root = Path(__file__).resolve().parents[1]
    args_class.path_preprocessor = package_root / "models" / "wav2vec2" / "config" / "preprocessor_dolphin.json"
    args_class.path_model = package_root / "models" / "wav2vec2" / "config" / "config_dolphin.json"
    args_class.path_train_arg = package_root / "models" / "wav2vec2" / "config" / "trainingArg.yaml"

    if not args_class.output_dir:
        args_class.output_dir = str(artifact_dir("outputs", "pretraining_run"))
    if not args_class.log_dir:
        args_class.log_dir = str(artifact_dir("logs", "pretraining_submitit"))

    args_class.modelType = "wav2vec2"

    try:
        assert args_class.modelType in IMPLEMENTED_MODELS.keys()
    except:
        raise Exception("Model not recognized, implemented models are "+str(list(IMPLEMENTED_MODELS.keys())))
    ensure_dir(args_class.output_dir)

    if str(args_class.path_model).endswith(".json"):
        config = IMPLEMENTED_MODELS[args_class.modelType].load_config(args_class.path_model)
        model = IMPLEMENTED_MODELS[args_class.modelType](config)
        dir_init = os.path.join(args_class.output_dir,"initialmodel")
        if not os.path.exists(dir_init):
            model.save_pretrained(dir_init)
        args_class.path_model = dir_init

    log_dir = ensure_dir(args_class.log_dir)
    executor = submitit.AutoExecutor(folder=str(log_dir),slurm_max_num_timeout=20)

    NUM_TASKS_PER_NODE = args_class.nb_gpu
    NUM_NODES = args_class.nb_nodes
    executor.update_parameters(**submitit_parameters(
        slurm_partition=args_class.slurm_partition,
        gpus_per_node=NUM_TASKS_PER_NODE,
        nodes=NUM_NODES,
        cpus_per_task=args_class.cpus_per_task,
        tasks_per_node=NUM_TASKS_PER_NODE,
        timeout_min=args_class.timeout_min,
        slurm_gres="gpu:"+str(NUM_TASKS_PER_NODE),
        slurm_account=args_class.slurm_account,
    ))

    class Task:
        def __call__(self):
            from ANNpretraining.pretraining.trainloop import  train

            print("exporting PyTorch distributed environment variables")
            dist_env = submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
            print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
            print(f"rank: {dist_env.rank}")
            print(f"world size: {dist_env.world_size}")
            print(f"local rank: {dist_env.local_rank}")
            print(f"local world size: {dist_env.local_world_size}")

            train(args_class.modelType, args_class.path_model, args_class.output_dir,
                  args_class.path_preprocessor, args_class.path_train_arg,
                  args_class.preprocess,
                  args_class.path_data,
                  args_class.path_data_config)
        def checkpoint(self):
            print("checkpointing")
            return submitit.helpers.DelayedSubmission(self)

    redo = True
    while redo:
        task = Task()
        job = executor.submit(task)
        submitit.helpers.monitor_jobs([job])
        print(job.results()[0])
        redo = not (job.state.upper() == "COMPLETED")

if __name__=="__main__":
    main(sys.argv[1:])
