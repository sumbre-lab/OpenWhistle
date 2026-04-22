import shutil
import huggingface_hub
from huggingface_hub import HfApi
import os
import sys
import submitit
from ANNpretraining.hubTools.token_hub import TOKEN,nameaccount
## Download from huggingface the model of interest
from tools import _upload_batch
from pathlib import Path

def upload_model(model_name,model_type,name_data,version,path_output):
    huggingface_hub.login(token=TOKEN)

    repo_id = nameaccount + "/model-" + model_name + "_type-" + \
              model_type + "_data-" + name_data + "_version-" + version

    fileFilter = ["config.json","pytorch_model.bin","model.safetensors"] #+ ["rng_state_"+str(i)+".pth" for i in range(32)]
    #+["optimizer.pt", "trainer_state.json", "training_args.bin","sheduler.pt", "scaler.pt"]


    return _upload_batch(repo_id,path_output,fileFilter,chksize=30)


def main(args):
    # Some pure CPU runs

    # PATH_TO_OUTPUT = Path("/lustre/fsn1/projects/rech/fqt/uzz43va/pretraining")
    PATH_TO_OUTPUT = Path("/lustre/fsn1/projects/rech/fqt/uzz43va/pretraining")
    # PATH_TO_OUTPUT = Path("/lustre/fsn1/projects/rech/bxp/ubm84dh/wav2vec2_pretraining/pretraining_output")

    model_names = ["wav2vec2"]
    model_types = ["base"]
    name_datas = ["librispeech"]
    versions = ["2"]

    local = True

    # executor = submitit.AutoExecutor(folder="log_download")
    # executor.update_parameters(slurm_partition="devlab",
    #                            nodes=1,
    #                            cpus_per_task=20,
    #                            tasks_per_node=1,
    #                            timeout_min=360)

    if local:
        for model_name, model_type, name_data, version in zip(model_names, model_types, name_datas,
                                                                          versions):
            # output_dir = str(PATH_TO_OUTPUT/("outputs_"+name_data+"_"+model_type+"_"+version))
            # output_dir = str(PATH_TO_OUTPUT/("outputs_"+model_type+"_"+name_data+"_"+version))
            output_dir = str(PATH_TO_OUTPUT / ("outputs_" + "base" + "_" + name_data + "_" + version))
            upload_model(model_name, model_type, name_data, version, output_dir)
    else:
        executor = submitit.AutoExecutor(folder="log_download")
        executor.update_parameters(slurm_partition="prepost",
                                   nodes=1,
                                   cpus_per_task=20,
                                   tasks_per_node=1,
                                   timeout_min=360,
                                   account="fqt@cpu")

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