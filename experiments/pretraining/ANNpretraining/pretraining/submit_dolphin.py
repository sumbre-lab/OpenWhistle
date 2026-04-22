from pathlib import Path
import  os
import submitit
import shutil
import sys
from ANNpretraining.models import IMPLEMENTED_MODELS
from ANNpretraining.models.wav2vec2.forPreTraining import Wav2vec2ForPreTraining_randommask,Wav2Vec2Config
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Launch the training loop')
    # the following should change
    parser.add_argument('--path_data', type=str,
                        default=os.path.join("..", "data", "librispeech_example"),
                        help='path to the data')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join("outputs_librispeech_correctLR_gradmulextractor_featurepen2"),
                        help='path where the model is output')
    parser.add_argument('--preprocess', type=bool,
                        default=False,
                        help='sets to true if the dataset is not'
                             ' preprocessed and comes as a matrix of sounds instead of a pyarrow datasets.'
                             ' If False we will just padds the sequences when batching, masking will be done on the fly '
                             'by the model forward pass ')
    # will probably not change
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
    return parser

def main(args):
    parser = get_parser()
    args_class = parser.parse_args(args)

    args_class.path_preprocessor = Path(__file__).parent.parent/"ANNpretraining/models/wav2vec2/config/preprocessor_dolphin.json"
    args_class.path_model = Path(__file__).parent.parent/"ANNpretraining/ANNpretraining/models/wav2vec2/config/config_dolphin.json"
    args_class.path_train_arg = Path(__file__).parent.parent/"ANNpretraining/ANNpretraining/models/wav2vec2/config/trainingArg.yaml"

    PATH_TO_OUTPUT = "/lustre/fsn1/projects/rech/fqt/uzz43va/pretraining"
    args_class.output_dir = os.path.join(PATH_TO_OUTPUT,"outputs_DolphinTalk_base_0")

    args_class.modelType = "wav2vec2"
    PATH_TO_DATASET = "/lustre/fsn1/projects/rech/fqt/uzz43va/datasets"
    args_class.path_data = os.path.join(PATH_TO_DATASET,"DolphinTalk","DolphinTalk_save")

    try:
        assert args_class.modelType in IMPLEMENTED_MODELS.keys()
    except:
        raise Exception("Model not recognized, implemented models are "+str(list(IMPLEMENTED_MODELS.keys())))
    if not os.path.exists(args_class.output_dir):
        os.makedirs(args_class.output_dir)

    ## We need to initialize the model before the code is spread across the different nodes:
    if args_class.path_model.endswith(".json"):
        config = IMPLEMENTED_MODELS[args_class.modelType].load_config(args_class.path_model)
        model = IMPLEMENTED_MODELS[args_class.modelType](config)
        dir_init = os.path.join(args_class.output_dir,"initialmodel")
        if not os.path.exists(dir_init):
            model.save_pretrained(dir_init)
        args_class.path_model = dir_init

    PATH_TO_LOG = "/lustre/fsn1/projects/rech/fqt/uzz43va/pretraining/logs/log_dolphintraining"
    os.makedirs(PATH_TO_LOG,exist_ok=True)
    executor = submitit.AutoExecutor(folder=PATH_TO_LOG,slurm_max_num_timeout=20)

    NUM_TASKS_PER_NODE = args_class.nb_gpu
    NUM_NODES = args_class.nb_nodes
    executor.update_parameters(slurm_partition="gpu_p13", #gpu_p2
                               gpus_per_node=NUM_TASKS_PER_NODE,
                               nodes=NUM_NODES,
                               cpus_per_task=10,
                               tasks_per_node=NUM_TASKS_PER_NODE,
                               timeout_min= 10*60,#10*60,
                               slurm_gres="gpu:"+str(NUM_TASKS_PER_NODE),
                               account="vzf@v100")

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
                  args_class.path_data)
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
