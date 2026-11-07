"""Train OpenWhistle Wav2Vec2 with native Fairseq on local GPUs or CPU."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "runs/pretraining"
CONFIG = Path(__file__).with_name("pretraining_config.yaml")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data_dir", type=Path, default=RUNS_DIR / "data/manifests")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--world_size", type=positive_int, default=1)
    parser.add_argument("--update_freq", type=positive_int, default=1)
    parser.add_argument("--max_tokens", type=positive_int, default=882000)
    parser.add_argument("--max_sample_size", type=positive_int, default=882000)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry_run", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    if args.max_tokens < args.max_sample_size:
        parser.error("--max_tokens must be at least --max_sample_size")
    if not args.config.is_file():
        parser.error(f"Configuration not found: {args.config}")
    if args.output_dir is None:
        timestamp = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y%m%d-%H%M%S")
        args.output_dir = RUNS_DIR / f"run-{timestamp}"
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.config = args.config.resolve()
    return args, overrides


def training_command(args, overrides=()):
    # JSON strings also provide the quoting required by Hydra for literal paths.
    return [
        sys.executable,
        "-m",
        "fairseq_cli.hydra_train",
        "--config-dir",
        str(args.config.parent),
        "--config-name",
        args.config.stem,
        f"task.data={json.dumps(str(args.data_dir))}",
        f"checkpoint.save_dir={json.dumps(str(args.output_dir))}",
        f"hydra.run.dir={json.dumps(str(args.output_dir / 'logs'))}",
        f"distributed_training.distributed_world_size={args.world_size}",
        f"optimization.update_freq=[{args.update_freq}]",
        f"dataset.max_tokens={args.max_tokens}",
        f"task.max_sample_size={args.max_sample_size}",
        f"common.fp16={str(args.fp16).lower()}",
        *overrides,
    ]


def main(argv=None):
    args, overrides = parse_args(argv)
    command = training_command(args, overrides)
    if args.dry_run:
        print(shlex.join(command))
        return
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Pretraining requires the Python 3.10 environment and the pinned "
            "dependencies in experiments/pretraining/requirements.txt. "
            "Activate that environment before training."
        )
    for name in ("train.tsv", "valid.tsv"):
        if not (args.data_dir / name).is_file():
            raise FileNotFoundError(f"Missing manifest: {args.data_dir / name}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    import torch
    from fairseq_cli.hydra_train import cli_main

    # PyTorch 2.0.1 MKLDNN Conv1d backward can produce invalid CPU gradients.
    # The native CPU kernels avoid this; CUDA kernels are unaffected.
    torch.backends.mkldnn.enabled = False
    print(shlex.join(command), flush=True)
    sys.argv = ["fairseq-hydra-train", *command[3:]]
    cli_main()


if __name__ == "__main__":
    main()
