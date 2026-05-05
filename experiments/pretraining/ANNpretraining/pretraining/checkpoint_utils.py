import os
import numpy as np

def checkpoint_loading(hfArg_out):
    """Return the latest checkpoint with a valid Trainer state."""
    import re
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    def get_last_checkpoint(folder,content):
        checkpoints = [
            path
            for path in content
            if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
        ]
        if len(checkpoints) == 0:
            return
        return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))

    last_checkpoint = get_last_checkpoint(hfArg_out.output_dir,os.listdir(hfArg_out.output_dir))
    if last_checkpoint is None:
        return False
    else:
        to_remove = [last_checkpoint]
        while (not "trainer_state.json" in os.listdir(last_checkpoint)) and len(to_remove)<len(os.listdir(hfArg_out.output_dir)):
            remaining = np.setdiff1d(os.listdir(hfArg_out.output_dir),to_remove)
            last_checkpoint = get_last_checkpoint(hfArg_out.output_dir, remaining)
            to_remove+=[last_checkpoint]
        return last_checkpoint
