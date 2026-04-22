import shutil

import huggingface_hub
import pathtools.path
from huggingface_hub import HfApi
import os
import numpy as np
from tqdm import tqdm
import sys
import submitit
import argparse

## Download from huggingface the model of interest
from tools import _download
from ANNpretraining.hubTools.token_hub import TOKEN,nameaccount


from pathlib import  Path

# dir_output = "/media/pierre/NeuroData2/models/"
dir_output = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/pretrainedModels/wav2vec2_onlylast"

def download_model(analyses,model_name,model_type,name_data,version,output_dir):

    path_output = os.path.join(dir_output,model_name,output_dir)

    huggingface_hub.login(token=TOKEN)

    repo_id = nameaccount + "/model-" + model_name + "_type-" + \
              model_type + "_data-" + name_data + "_version-" + version

    _download(repo_id,path_output,fileFilter=analyses,repo_type="model")
    return True

def main(args):
    parser = argparse.ArgumentParser(description='Launch the post analysis')
    # the following should change
    parser.add_argument('--local',type=int,default=1,help="if set to 1: "
     "sequentially download the analysis for all models, if set to 0 dispatch them on the cluster")
    args_class = parser.parse_args(args)

    # output_dirs = ["outputs_audiosetfilter_short","outputs_audiosetfilter_short_2"]
    # model_names = ["wav2vec2","wav2vec2"]
    # model_types = ["base","base"]
    # name_datas = ["audiosetfilter","audiosetfilter"]
    # versions = ["0","2"]
    #
    output_dirs = ["outputs_librispeech_short_3","outputs_librispeech_short_4",
                   "outputs_fma_short_2","outputs_fma_short_3",
                   "outputs_audiosetfilter_short","outputs_audiosetfilter_short_2",
                   "outputs_mergefilter_short","outputs_mergefilter_short_2"]
    model_names = ["wav2vec2","wav2vec2","wav2vec2","wav2vec2","wav2vec2","wav2vec2","wav2vec2","wav2vec2"]
    model_types = ["base","base","base","base","base","base","base","base"]
    name_datas = ["librispeech","librispeech","fma","fma","audiosetfilter","audiosetfilter","mergefilter","mergefilter"]
    versions = ["3","4","2","3","0","2","0","2"]

    fileFilter = ["checkpoint-100000","initialmodel"] #"config.json", "checkpoint-10000","checkpoint-50000",
    if args_class.local==1:
        # Here one define the set of analysis he weants to load from the model
        for model_name, model_type, name_data, version, output_dir in zip(model_names, model_types, name_datas, versions,
                                                                          output_dirs):
            print(model_name," ",model_type," ",name_data," ",version," ",output_dir)
            download_model(fileFilter,model_name, model_type, name_data, version, output_dir)
    else:
        # executor = submitit.AutoExecutor(folder="log_download")
        executor = submitit.AutoExecutor(folder="log_download")
        executor.update_parameters(slurm_partition="prepost",
                                   nodes=1,
                                   cpus_per_task=20,
                                   tasks_per_node=1,
                                   timeout_min=360,
                                   account="fqt@cpu")
        class Task:
            def __call__(self, model_name, model_type, name_data, version, output_dir):
                done = download_model(fileFilter,model_name, model_type, name_data, version, output_dir)
                return done
            def checkpoint(self):
                print("checkpointing")
                return submitit.helpers.DelayedSubmission(self)
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