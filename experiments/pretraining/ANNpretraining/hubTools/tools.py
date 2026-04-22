import huggingface_hub
from huggingface_hub import HfApi
import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import  tqdm


def _downloadremovecache(args):
    chk, repo_id, path_output, to_download = args

    for t in to_download:
        out = huggingface_hub.hf_hub_download(repo_id=repo_id,
                                              filename=t, subfolder=os.path.join("checkpoints", chk),
                                              cache_dir=os.path.join(path_output, chk, "cache"), resume_download=True)
        target = os.readlink(out)
        absPathTarget = os.path.abspath(os.path.join(os.getcwd(), out, "..", target))
        absPathOutput = os.path.abspath(os.path.join(os.getcwd(), path_output, chk, t))
        ## rename and move the target
        os.rename(absPathTarget, absPathOutput)

    # delete the file versioning system once all file of the checkpoints have been downloaded:
    shutil.rmtree(os.path.join(path_output, chk, "cache"))
    return True

# def _uploadmodels(args):
#     api = HfApi()
#     chk, repo_id, path_output, to_upload = args
#     for t in to_upload:
#         api.upload_file(path_or_fileobj=os.path.join(path_output, chk,t),
#                         path_in_repo="checkpoints/" + chk +"/"+t,
#                         repo_id=repo_id)
#     return True

def _downloadAnalysisremovecache(args):
    repo_ids, path_outputs, subfolderPaths, fileNames,repo_types = args
    for repo_id,path_output,subfolderPath,fileName,repo_type in zip(repo_ids, path_outputs, subfolderPaths, fileNames,repo_types):
        out = huggingface_hub.hf_hub_download(repo_id=repo_id,
                                              filename=fileName, subfolder=subfolderPath,
                                              cache_dir=os.path.join(path_output, "cache"),
                                              resume_download=True,
                                              repo_type=repo_type)
        if os.path.islink(out):
            target = os.readlink(out)
        else:
            target = out

        if os.path.isabs(target):
            absPathTarget = target
        else:
            absPathTarget = os.path.abspath(os.path.join(os.getcwd(), out, "..", target))

        if os.path.isabs(path_output):
            absPathOutput = os.path.abspath(os.path.join(path_output, subfolderPath, fileName))
            os.makedirs(os.path.join(path_output, subfolderPath),exist_ok=True)
        else:
            absPathOutput = os.path.abspath(os.path.join(os.getcwd(), path_output, subfolderPath, fileName))
            os.makedirs(os.path.join(os.getcwd(), path_output, subfolderPath),exist_ok=True)
        ## rename and move the target
        os.rename(absPathTarget, absPathOutput)
    return True


# def _uploadAnalysis(args):
#     repo_id, path_output, subfolder, filename = args
#     api = HfApi()
#     api.upload_file(path_or_fileobj=os.path.join(path_output, subfolder,filename),
#                     path_in_repo=subfolder+"/"+filename,
#                     repo_id=repo_id)
#     return True

from typing import List
from huggingface_hub import  CommitOperationAdd

def _uploadSingleBatch(repo_id,path_output,subfolders,filenames,repo_type=None):
    api = HfApi()

    files_to_add: List[CommitOperationAdd] = []
    for subfolder, filename in zip(subfolders,filenames):
        files_to_add.append(
            CommitOperationAdd(
                path_or_fileobj=os.path.join(path_output, subfolder,filename),
                path_in_repo=subfolder+"/"+filename))

    commit_info = api.create_commit(
        repo_id=repo_id,
        operations=files_to_add,
        commit_message="Upload batch with huggingface_hub",
        num_threads=30,
        repo_type=repo_type)

    return True

def _rec_search(f):
    # given a folder return a list of all the file in the folder and in the subfolders
    # along with their path concatenated to the folder path.
    if os.path.isdir(f):
        try:
            return np.concatenate([_rec_search(os.path.join(f,o)) for o in os.listdir(f)])
        except:
            return []
    else:
        return np.array([f])

def _rec_search_subfolder(f):
    # provide only the subfolder path
    r = _rec_search(f)
    return np.array([e.replace(f+"/","") for e in r])


def _upload_batch(repo_id, path_output, fileFilter, chksize=300,repo_type=None):
    ## TODO: add the possibility to upload only a subset of all the checkpoints.

    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, private=True,repo_type=repo_type)
    except:
        pass

    files_server = api.list_repo_files(repo_id=repo_id,repo_type=repo_type)
    files_local = _rec_search_subfolder(path_output)
    files_local = list(filter(lambda e: np.any([e.__contains__(a) for a in fileFilter]), files_local))
    files_to_upload = np.setdiff1d(files_local, files_server)

    while len(files_to_upload) >= 1:

        try:
            # update the list of checkpoints that remains to be uploaded (in case of sudden error and exit, normally None)
            files_server = api.list_repo_files(repo_id=repo_id,repo_type=repo_type)
            files_to_upload = np.setdiff1d(files_local, files_server)

            print("---nb of files yet to download---:")
            print(len(files_to_upload))

            ## Parallelize the download:
            print("upload sequentially using batch")

            subfolders = np.array([str(Path(p).parents[0]) for p in files_to_upload])
            filenames = np.array([Path(p).name for p in files_to_upload])
            # chunk the upload:
            allsubs = [subfolders[i * chksize:np.min([i * chksize + chksize, subfolders.shape[0]])] for i in
                       range(int(np.ceil(subfolders.shape[0] / chksize)))]
            allfiles = [filenames[i * chksize:np.min([i * chksize + chksize, subfolders.shape[0]])] for i in
                        range(int(np.ceil(subfolders.shape[0] / chksize)))]
            # chunks = 600
            for subs, files in tqdm(zip(allsubs, allfiles), total=len(allsubs)):
                print("uploading chunk")
                _uploadSingleBatch(repo_id, path_output, subs, files,repo_type=repo_type)  # to model repository

            # update the list of checkpoints that remains to be uploaded (in case of sudden error and exit, normally None)
            files_server = api.list_repo_files(repo_id=repo_id,repo_type=repo_type)
            files_to_upload = np.setdiff1d(files_local, files_server)
        except RuntimeError:
            pass
            # pass  # infinite loop until all file have been uploaded.

    return True

def _download(repo_id, path_output, fileFilter,repo_type=None):
    api = HfApi()
    files_server = api.list_repo_files(repo_id=repo_id,repo_type=repo_type)
    files_server = list(filter(lambda e: np.any([e.__contains__(a) for a in fileFilter]), files_server))

    # list allready downloaded checkpoints:
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    try:
        files_local = _rec_search_subfolder(path_output)
        files_local = list(filter(lambda e: np.any([e.__contains__(a) for a in fileFilter]), files_local))
    except:
        files_local = []
    check_to_download = np.setdiff1d(files_server, files_local)

    while len(check_to_download) >= 1:

        try:
            print("---download remaining---:")
            print(len(check_to_download))
            print("path_output: ",path_output)

            ## Parallelize the download:
            print("downloading  in parallel over the maximum number of cpu available")

            from concurrent.futures import ProcessPoolExecutor, wait
            nbcpu = min(20,len(check_to_download))
            executor = ProcessPoolExecutor(max_workers=nbcpu)
            # perform all tasks in parallel
            arguments = np.stack([[repo_id for _ in range(len(check_to_download))],
                                  [path_output for _ in range(len(check_to_download))],
                                  [str(Path(p).parents[0]) for p in check_to_download],  # path subfolder
                                  [Path(p).name for p in check_to_download],
                                  [repo_type for _ in check_to_download]])  # path file
            # over nbcpu cpu:
            q = arguments.shape[1] // nbcpu
            arguments_chunked = arguments[:, :(nbcpu * q)].reshape((5, -1, nbcpu))
            arguments_chunked = [arguments_chunked[..., i] for i in range(nbcpu)]
            arguments_chunked[-1] = np.concatenate([arguments_chunked[-1], arguments[:, nbcpu * q:]], axis=-1)
            results = executor.map(_downloadAnalysisremovecache, arguments_chunked)
            finished = np.stack(results, axis=0)  # just loop to make sure all upload are finished
            # delete the file versioning system once all file of the checkpoints have been downloaded:
            shutil.rmtree(os.path.join(path_output, "cache"))

        except:
            pass
        # update the list of checkpoints that remains to be downloaded (in case of sudden error and exit, normally None)
        try:
            files_local = _rec_search_subfolder(path_output)
            files_local = list(filter(lambda e: np.any([e.__contains__(a) for a in fileFilter]), files_local))
        except:
            files_local = []
        check_to_download = np.setdiff1d(files_server, files_local)
    return True