import shutil
import huggingface_hub
from huggingface_hub import HfApi
import os
import sys
import submitit
from ANNpretraining.hubTools.token_hub import login_to_hub, require_hf_repo_owner
from ANNpretraining.runtime import artifact_dir, submitit_parameters
## Download from huggingface the model of interest
from tools import _upload_batch
from pathlib import Path

def upload_model(model_name,model_type,name_data,version,path_output):
    login_to_hub()

    repo_owner = require_hf_repo_owner()
    repo_id = repo_owner + "/model-" + model_name + "_type-" + \
              model_type + "_data-" + name_data + "_version-" + version

    fileFilter = ["config.json","pytorch_model.bin","model.safetensors"] #+ ["rng_state_"+str(i)+".pth" for i in range(32)]
    #+["optimizer.pt", "trainer_state.json", "training_args.bin","sheduler.pt", "scaler.pt"]


    return _upload_batch(repo_id,path_output,fileFilter,chksize=30)


def main(args):
    # Some pure CPU runs

    PATH_TO_OUTPUT = artifact_dir("outputs", create=True)

    model_names = ["wav2vec2"]
    model_types = ["base"]
    name_datas = ["librispeech"]
    versions = ["2"]
    output_dirs = [
        str(PATH_TO_OUTPUT / ("outputs_" + model_type + "_" + name_data + "_" + version))
        for model_type, name_data, version in zip(model_types, name_datas, versions)
    ]

    local = True

    # executor = submitit.AutoExecutor(folder="log_download")
    # executor.update_parameters(slurm_partition="devlab",
    #                            nodes=1,
    #                            cpus_per_task=20,
    #                            tasks_per_node=1,
    #                            timeout_min=360)

    if local:
        for model_name, model_type, name_data, version, output_dir in zip(
            model_names, model_types, name_datas, versions, output_dirs
        ):
            upload_model(model_name, model_type, name_data, version, output_dir)
    else:
        executor = submitit.AutoExecutor(folder="log_download")
        executor.update_parameters(**submitit_parameters(
            slurm_partition=os.environ.get("OPENWHISTLE_SLURM_PARTITION"),
            nodes=1,
            cpus_per_task=20,
            tasks_per_node=1,
            timeout_min=360,
            account=os.environ.get("OPENWHISTLE_SLURM_ACCOUNT"),
        ))

        class Task:
            def __call__(self, model_name, model_type, name_data, version, output_dir):
                done = upload_model(model_name, model_type, name_data, version, output_dir)
                return done

            def checkpoint(self,model_name, model_type, name_data, version, output_dir):
                print("checkpointing")
                return submitit.helpers.DelayedSubmission(self,model_name, model_type, name_data, version, output_dir)

        task = Task()

        jobs = []
        with executor.batch():
            for model_name, model_type, name_data, version, output_dir in zip(model_names, model_types, name_datas, versions, output_dirs):
                job = executor.submit(task, model_name, model_type, name_data, version, output_dir)
                jobs.append(job)
        submitit.helpers.monitor_jobs(jobs)
        outputs = [job.result() for job in jobs]
        print("Finished downloading the models")

if __name__=="__main__":
    main(sys.argv[1:])
