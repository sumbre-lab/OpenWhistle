import shutil

import huggingface_hub
from huggingface_hub import HfApi
import os
import numpy as np
from tqdm import tqdm
import sys
import submitit
import argparse
from ANNpretraining.hubTools.token_hub import TOKEN,nameaccount

## Download from huggingface the model of interest
from tools import _downloadremovecache

dir_output = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/pretrainedModels"


def download_model(model_name,model_type,name_data,version,output_dir):
    ## TODO: add the possibility to download only a subset of all the checkpoints.

    path_output = os.path.join(dir_output,model_name,output_dir)

    huggingface_hub.login(token=TOKEN)

    api = HfApi()
    repo_id = nameaccount + "/model-" + model_name + "_type-" + \
              model_type + "_data-" + name_data + "_version-" + version
    rf = api.list_repo_files(repo_id=repo_id)
    rf_checkpoints = list(filter(lambda e: e.__contains__("checkpoint"), rf))
    list_checkpoints = np.unique([r.split("/")[1] for r in rf_checkpoints])
    to_download = ["config.json"] #,"pytorch_model.bin" #,"pytorch_model.bin"

    # list allready downloaded checkpoints:
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    ## Verify that the checkpoint-local contains all the files:
    def has_files(e):
        dirs = os.listdir(os.path.join(path_output,e))
        for t in to_download:
            try:
                assert t in dirs
            except:
                return False
        return True

    list_checkpoints_local = os.listdir(os.path.join(path_output))
    list_checkpoints_local = list(filter(lambda e: e.__contains__("checkpoint") or (e=="initialmodel"),
                                         list_checkpoints_local))
    list_checkpoints_local = list(filter(lambda e: has_files(e),list_checkpoints_local))

    check_to_download = np.setdiff1d(list_checkpoints,list_checkpoints_local)

    # From the model we just download the current state without the optimizer
    # as well as the losses
    while len(check_to_download)>=1:

        try:
            print("---download remaining---:")
            print(check_to_download)

            ## Parallelize the download:
            print("downloading  in parallel over the maximum number of cpu available")

            from concurrent.futures import ProcessPoolExecutor, wait
            executor = ProcessPoolExecutor()
            # perform all tasks in parallel
            arguments = zip(check_to_download,[repo_id for _ in range(len(check_to_download))],
                            [path_output for _ in range(len(check_to_download))],
                            [to_download for _ in range(len(check_to_download))])
            results = executor.map(_downloadremovecache, arguments)
            finished = np.stack(results, axis=0) #just loop to make sure all upload are finished

        except RuntimeError:
            pass
        # update the list of checkpoints that remains to be downloaded (in case of sudden error and exit, normally None)
        list_checkpoints_local = os.listdir(path_output)
        list_checkpoints_local = list(filter(lambda e: e.__contains__("checkpoint") or (e == "initialmodel"),
                                             list_checkpoints_local))
        list_checkpoints_local = list(filter(lambda e: has_files(e), list_checkpoints_local))
        check_to_download = np.setdiff1d(list_checkpoints, list_checkpoints_local)

    return True

def main(args):
    parser = argparse.ArgumentParser(description='Launch the post analysis')
    # the following should change
    parser.add_argument('--local', type=int, default=1, help="if set to 1: "
                                                             "sequentially download the analysis for all models, if set to 0 dispatch them on the cluster")
    args_class = parser.parse_args(args)

    output_dirs = ["outputs_librispeech_short_3","outputs_librispeech_short_4",
                   "outputs_fma_short_2","outputs_fma_short_3",
                   "outputs_audiosetfilter_short","outputs_audiosetfilter_short_2",
                   "outputs_mergefilter_short","outputs_mergefilter_short_2"]
    model_names = ["wav2vec2","wav2vec2","wav2vec2","wav2vec2","wav2vec2","wav2vec2","wav2vec2","wav2vec2"]
    model_types = ["base","base","base","base","base","base","base","base"]
    name_datas = ["librispeech","librispeech","fma","fma","audiosetfilter","audiosetfilter","mergefilter","mergefilter"]
    versions = ["3","4","2","3","0","2","0","2"]

    # output_dirs = ["outputs_fma_short_3"]
    # model_names = ["wav2vec2"]
    # model_types = ["base"]
    # name_datas = ["fma"]
    # versions = ["3"]


    ### VERY IMPORTANT:
    ## there can be problem with some models download.
    # If the expected size of the folder is 363M (for example, 363M is for Wav2vec2base + config file)
    # Then execute: du -h --max-depth=2 | awk '$1 ~ /^[0-9.]+[M|G]$/ && $1 < "363M"'
    # In the folder of the downloaded model.
    # This will outputs checkpoint that were not entirely downloaded. Make sure to remove these folders and start
    # the download again.

    if args_class.local==1:
        for model_name, model_type, name_data, version, output_dir in zip(model_names, model_types, name_datas,
                                                                          versions, output_dirs):
           download_model(model_name, model_type, name_data, version, output_dir)
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
                done = download_model(model_name, model_type, name_data, version, output_dir)
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