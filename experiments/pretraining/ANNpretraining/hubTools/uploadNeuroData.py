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
from ANNpretraining.hubTools.token_hub import TOKEN,nameaccount

## Download from huggingface the model of interest
from tools import _rec_search_subfolder ,_upload_batch

from pathlib import  Path

def upload_neurodataset():
    path_output = os.path.join("..","..","continuous_oncluster","datasets_oncluster")
    print(os.listdir(path_output))
    # Here one define the set of analysis he weants to load from the model
    # fileFilter = ["calcium","calcium-control","ephys","fmri","fmri-feature","fus-natural-subset",
    #             "fus_natural","fus_voc"]
    fileFilter = ["fus_natural-feature"]
    # fileFilter = ["calcium_data"]

    huggingface_hub.login(token=TOKEN)

    repo_id = nameaccount + "/NeuroData-auditoryProcessing"
    _upload_batch(repo_id, path_output, fileFilter=fileFilter, chksize=300,repo_type="dataset")

    return True

def main(args):
    parser = argparse.ArgumentParser(description='Launch the post analysis')
    # the following should change
    parser.add_argument('--local',type=int,default=1,help="if set to 1: "
     "sequentially download the analysis for all models, if set to 0 dispatch them on the cluster")
    args_class = parser.parse_args(args)

    if args_class.local == 1:
        upload_neurodataset()
    else:
        # Some pure CPU runs
        executor = submitit.AutoExecutor(folder="log_download")
        executor.update_parameters(slurm_partition="prepost", #devlab
                                   nodes=1,
                                   cpus_per_task=30,
                                   tasks_per_node=1,
                                   timeout_min=360,
                                   account="fqt@cpu")
        class Task:
            def __call__(self):
                done = upload_neurodataset()
                return done
            def checkpoint(self):
                print("checkpointing")
                return submitit.helpers.DelayedSubmission(self)
        task = Task()
        jobs = []
        with executor.batch():
            job = executor.submit(task)
            jobs.append(job)
        submitit.helpers.monitor_jobs(jobs)
        outputs = [job.result() for job in jobs]
        print("Finished downloading the models")

if __name__=="__main__":
    main(sys.argv[1:])