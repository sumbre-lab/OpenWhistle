import datasets
import huggingface_hub
from huggingface_hub import HfApi
import os
from ANNpretraining.hubTools.token_hub import TOKEN,nameaccount

huggingface_hub.login(token=TOKEN)
api = HfApi()


# repo_id =  "dolphinteam"+"/DolphinTalk" #"porhan/dataset-librispeech960h"
# output_dir = "/lustre/fsn1/projects/rech/dab/uzz43va/datasets"
# os.makedirs(os.path.join(output_dir,"DolphinTalk"),exist_ok=True)
# ds = datasets.load_dataset(path=repo_id,
#                       cache_dir=os.path.join(output_dir,"DolphinTalk_row"))
# ### You need to change the following line:
# path_to_disk = "/lustre/fsn1/projects/rech/fqt/uzz43va/DolphinTalk_save"
# ds.save_to_disk(path_to_disk)

repo_id =  "porhan"+"/dataset-mergeFmaLibrispeechAudiosetfilter" #"porhan/dataset-librispeech960h"
output_dir = "/lustre/fsn1/projects/rech/dab/uzz43va/datasets"
os.makedirs(os.path.join(output_dir,"mergefilter"),exist_ok=True)
ds = datasets.load_dataset(path=repo_id,
                      cache_dir=os.path.join(output_dir,"mergefitler_row"),
                           num_proc=10)
### You need to change the following line:
path_to_disk = "/lustre/fsn1/projects/rech/dab/uzz43va/mergefitler_save"
ds.save_to_disk(path_to_disk)

# os.makedirs(os.path.join(output_dir,"DolphinTalk"),exist_ok=True)
# ds = datasets.load_dataset(path=repo_id,
#                       cache_dir=os.path.join(output_dir,"DolphinTalk_row"))
# ### You need to change the following line:
# path_to_disk = "/lustre/fsn1/projects/rech/fqt/uzz43va/DolphinTalk_save"
# ds.save_to_disk(path_to_disk)

# repo_id =  nameaccount+"/dataset-fmalarge"
# repo_id = NDEM/dataset-librispeech960h"

# repo_id =  nameaccount+"/dataset-audiosetfilter"
# output_dir = "/lustre/fsn1/projects/rech/fqt/uzz43va/datasets"
# os.makedirs(os.path.join(output_dir,"audiosetfilter"),exist_ok=True)
# ds = datasets.load_dataset(path=repo_id,
#                       cache_dir=os.path.join(output_dir,"audiosetfilter_row"))
# ### You need to change the following line:
# path_to_disk = "/lustre/fsn1/projects/rech/fqt/uzz43va/datasets/audiosetfilter_save"
# ds.save_to_disk(path_to_disk)