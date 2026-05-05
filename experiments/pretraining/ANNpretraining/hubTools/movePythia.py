import shutil
import os
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import numpy as np
import tqdm
from pathlib import Path
from ANNpretraining.runtime import artifact_dir


def get_hf_repo_branches(repo_id):
    api = HfApi()
    branches =  api.list_repo_refs(repo_id).branches
    branches_names = [b.name for b in branches]
    return (branches,branches_names)

number_param = "160m"
owner = "EleutherAI"
repo = "pythia-"+number_param+"-deduped"
repo_id = owner +"/"+repo  # replace with the repository identifier
branches = get_hf_repo_branches(repo_id)
b_names = list(filter(lambda e:e.startswith("step"),branches[1]))
orderb_names = np.argsort([int(b.replace("step",""))
                           for b in b_names])
b_names = np.array(b_names)[orderb_names]

cache_dir = artifact_dir("models", "pythia", create=True)

for step in tqdm.tqdm(b_names[:1]):
    dir_step = cache_dir / ("pythia-" + number_param + "-deduped") / step
    os.makedirs(dir_step,exist_ok=True)
    model_name = os.listdir(dir_step)[0]
    dir_snapshot = dir_step / model_name / "snapshots"
    for t in ["config.json","pytorch_model.bin","special_tokens_map.json",
              "tokenizer.json","tokenizer_config.json"]:
        shutil.move(dir_snapshot/os.listdir(dir_snapshot)[0]/t,
                    dir_step/t)
    shutil.rmtree(dir_step /model_name )
#
for step in tqdm.tqdm(b_names[:1]):
    dir_step = cache_dir / ("pythia-" + number_param + "-deduped") / step
    shutil.move(dir_step, str(dir_step).replace("step","checkpoint-"))
