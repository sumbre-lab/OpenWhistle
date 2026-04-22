from transformers import GPTNeoXForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import numpy as np
import tqdm
from pathlib import Path
import shutil
import os
import submitit

def get_hf_repo_branches(repo_id):
    api = HfApi()
    branches =  api.list_repo_refs(repo_id).branches
    branches_names = [b.name for b in branches]
    return (branches,branches_names)

tmp_models = ["pythia-70M-deduped","pythia-160M-deduped","pythia-410M-deduped","pythia-1B-deduped","pythia-1.4B-deduped",
              "pythia-2.8B-deduped","pythia-6.9B-deduped","pythia-12B-deduped"] #
number_params = ["70M","160M","410M","1B","1.4B","2.8B"] #
for number_param in number_params:
    # number_param = "1.4B"
    owner = "EleutherAI"
    repo = "pythia-"+number_param+"-deduped"
    repo_id = owner +"/"+repo  # replace with the repository identifier
    branches = get_hf_repo_branches(repo_id)
    b_names = list(filter(lambda e:e.startswith("step"),branches[1]))
    orderb_names = np.argsort([int(b.replace("step",""))
                               for b in b_names])
    b_names = np.array(b_names)[orderb_names]
    local = False

    cache_dir = Path("/lustre/fsn1/projects/rech/fqt/uzz43va/NeuroData/pretrainedModels/pythia")
    os.makedirs(cache_dir,exist_ok=True)
    # cache_dir = Path("/media/pierre/NeuroData2/models/pythia")

    os.makedirs(cache_dir/("pythia-"+number_param+"-deduped"),exist_ok=True)

    def download_step(step):
        model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-"+number_param+"-deduped",
        revision=step,
        cache_dir=cache_dir / ("pythia-"+number_param+"-deduped") / step)
        tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-"+number_param+"-deduped",
        revision=step,
        cache_dir=cache_dir / ("pythia-"+number_param+"-deduped") / step)

        dir_step = cache_dir / ("pythia-" + number_param + "-deduped") / step

        model_name = list(filter(lambda e:e.startswith("models--"),os.listdir(dir_step)))[0]
        dir_snapshot = dir_step / model_name / "snapshots"
        name_snap = os.listdir(dir_snapshot)[0]
        for t in ["config.json","pytorch_model.bin","special_tokens_map.json",
                  "tokenizer.json","tokenizer_config.json"]:
            shutil.copy2(dir_snapshot/name_snap/t,
                        dir_step/t)
        shutil.rmtree(dir_step /model_name )
        shutil.move(dir_step, str(dir_step).replace("step","checkpoint-"))

    if local:
        for step in tqdm.tqdm(b_names):
            if not str(step).replace("step","checkpoint-") in os.listdir(cache_dir / ("pythia-" + number_param + "-deduped")):
                download_step(step)
    else:
        executor = submitit.AutoExecutor(folder="log_download")
        executor.update_parameters(slurm_partition="compil",
                                   nodes=1,
                                   cpus_per_task=20,
                                   tasks_per_node=1,
                                   timeout_min=30,
                                   account="dab@cpu")
        # executor.update_parameters(slurm_partition="cpu_p1",
        #                            nodes=1,
        #                            cpus_per_task=20,
        #                            tasks_per_node=1,
        #                            timeout_min=20,  # 200
        #                            account="dab@cpu")  #
        class Task:
            def __call__(self, step):
                download_step(step)
                return True

            def checkpoint(self):
                print("checkpointing")
                return submitit.helpers.DelayedSubmission(self)

        task = Task()
        jobs = []
        with executor.batch():
            for step in tqdm.tqdm(b_names[:]):
                if not str(step).replace("step","checkpoint-") in os.listdir(cache_dir / ("pythia-" + number_param + "-deduped")):
                    job = executor.submit(task, step)
                    jobs.append(job)
        submitit.helpers.monitor_jobs(jobs)
        outputs = [job.result() for job in jobs]