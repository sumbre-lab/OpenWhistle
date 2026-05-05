import argparse
import os
import sys

import submitit

from ANNpretraining.hubTools.token_hub import login_to_hub, require_hf_repo_owner
from ANNpretraining.runtime import artifact_dir, submitit_parameters

from tools import _upload_batch


def upload_dataset(path_output: str, repo_name: str, file_filter: list[str]):
    login_to_hub()
    repo_owner = require_hf_repo_owner()
    repo_id = f"{repo_owner}/{repo_name}"
    _upload_batch(repo_id, path_output, fileFilter=file_filter, chksize=300, repo_type="dataset")
    return True


def main(args):
    parser = argparse.ArgumentParser(description="Upload a local dataset directory to Hugging Face.")
    parser.add_argument("--local", type=int, default=1, help="Run locally when set to 1, otherwise dispatch with submitit.")
    parser.add_argument(
        "--path_output",
        default=str(artifact_dir("datasets", create=True)),
        help="Directory containing the files to upload.",
    )
    parser.add_argument(
        "--repo_name",
        default="anonymous-audio-dataset",
        help="Dataset repository name created under the configured HF owner.",
    )
    parser.add_argument(
        "--file_filter",
        nargs="+",
        default=["train", "validation", "test"],
        help="Only upload files whose relative path contains one of these tokens.",
    )
    args_class = parser.parse_args(args)

    if args_class.local == 1:
        upload_dataset(args_class.path_output, args_class.repo_name, args_class.file_filter)
    else:
        executor = submitit.AutoExecutor(folder="log_download")
        executor.update_parameters(**submitit_parameters(
            slurm_partition=os.environ.get("OPENWHISTLE_SLURM_PARTITION"),
            nodes=1,
            cpus_per_task=30,
            tasks_per_node=1,
            timeout_min=360,
            account=os.environ.get("OPENWHISTLE_SLURM_ACCOUNT"),
        ))

        class Task:
            def __call__(self):
                return upload_dataset(args_class.path_output, args_class.repo_name, args_class.file_filter)

            def checkpoint(self):
                print("checkpointing")
                return submitit.helpers.DelayedSubmission(self)

        task = Task()
        jobs = []
        with executor.batch():
            jobs.append(executor.submit(task))
        submitit.helpers.monitor_jobs(jobs)
        _ = [job.result() for job in jobs]


if __name__ == "__main__":
    main(sys.argv[1:])
