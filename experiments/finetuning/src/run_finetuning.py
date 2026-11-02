"""Fine-tune six encoders on all/balanced classification and detection."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "benchmark/src"))
from benchmark_revisions import MODEL_REVISIONS as HF_MODEL_REVISIONS  # noqa: E402
from aves_models import AVES_SCRATCH_MODELS  # noqa: E402
from run_benchmark import environment_record  # noqa: E402

MODEL_ID = "dolphinteam/OpenWhistle-Wav2Vec2.0"
HF_MODELS = ("wav2vec2", "aves_continual")
MODELS = (
    ("aves_core", "AVES-core", "General audio (AudioSet, FSD50K)", "aves_core"),
    ("biolingual", "BioLingual", "Audio-text (AnimalSpeak)", "davidrrobinson/biolingual"),
    ("aves_bio", "AVES-bio", "Animal vocalizations (AudioSet, VGGSound)", "aves_bio"),
    ("wav2vec2", "Wav2Vec2.0", "OpenWhistle (ours)", MODEL_ID),
    ("aves_scratch", "AVES-bio (from scratch)", "OpenWhistle (from scratch)",
     "dolphinteam/AVES-OpenWhistle-Stage-2-320-100pct"),
    ("aves_continual", "AVES-bio (continual)", "AVES-bio then OpenWhistle",
     "dolphinteam/AVES-bio-OpenWhistle-100pct-44kHz"),
)
MODEL_REVISIONS = {
    key: value for key, value in {
        **HF_MODEL_REVISIONS,
        **{spec["model_id"]: spec["revision"] for spec in AVES_SCRATCH_MODELS},
    }.items() if key in {model[3] for model in MODELS}
}
TASKS = ("classification", "balanced", "detection")
COMMON = dict(learning_rate=1e-5, head_learning_rate=1e-4, weight_decay=0.01,
              warmup_ratio=0.1, dropout=0.1, max_grad_norm=1.0, mixed_precision="float16",
              eval_batch_size=1, eval_precision="none", num_workers=4,
              refit_train_validation=True)


def recipe(key, task):
    """Retain the audited per-model training settings; unify validation/test."""
    if task == "balanced":
        task = "classification"
    settings = {**COMMON, "gradient_accumulation_steps": 1}
    if task == "classification":
        settings.update(epochs=50, patience=15, batch_size=32, label_smoothing=0.0)
        if key == "wav2vec2":
            settings["gradient_accumulation_steps"] = 2
        elif key == "biolingual":
            settings["learning_rate"] = 5e-6
    else:
        settings.update(epochs=30, patience={"aves_core": 5, "aves_bio": 5,
                                          "biolingual": 10, "wav2vec2": 15,
                                          "aves_scratch": 5, "aves_continual": 5}[key],
                        batch_size=64 if key in ("aves_bio", "aves_scratch", "aves_continual") else 8)
    if key == "wav2vec2":
        settings["sample_rate"] = 44100
    if key == "aves_continual":
        # Three band views per clip; retain the AVES effective batch with a smaller physical batch.
        settings.update(batch_size=4, gradient_accumulation_steps=8 if task == "classification" else 16,
                        sample_rate=16000, feature_mode="btb3_concat")
    return settings


def make_command(output, model, task, seed):
    key, _, _, model_id = model
    prefix = "finetune_hf" if key in HF_MODELS else "finetune"
    suffix = "_detection_map.py" if key in HF_MODELS else "_detection.py"
    script = prefix + (suffix if task == "detection" else "_classification.py")
    command = [sys.executable, "-u", str(ROOT / "src" / script),
               "--output_dir", str(output / key / task / f"seed_{seed}"), "--seed", str(seed)]
    if key in (*HF_MODELS, "biolingual", "aves_scratch"):
        command += ["--model_id", model_id]
    if key not in HF_MODELS:
        command += ["--backbone", key]
    if task != "detection":
        command += ["--dataset_config", "balanced" if task == "balanced" else "all"]
    for option, value in recipe(key, task).items():
        command += [f"--{option}"] if value is True else [f"--{option}", str(value)]
    return command


def load_result(path, model, task, seed, *, require_checkpoint=True, require_predictions=True):
    key, _, _, model_id = model
    with path.open() as handle:
        result = json.load(handle)
    metric = "map" if task == "detection" else "macro_f1"
    expected_config = {"classification": "all", "balanced": "balanced", "detection": "default"}[task]
    if (result.get("protocol") != "supervised_end_to_end_finetuning"
            or result.get("model_id") != model_id or result.get("seed") != seed
            or result.get("dataset_config") != expected_config):
        raise ValueError(f"Incompatible model, seed or task in {path}")
    value = float(result["test"][metric])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid test metric in {path}")
    settings = {**recipe(key, task), "seed": seed}
    if (result.get("model_revision") != MODEL_REVISIONS.get(model_id)
            or result.get("training_args", {}).get("class_weight", "none") != "none"
            or result.get("class_weighting") is not None
            or result.get("final_training", {}).get("class_weighting") is not None
            or result.get("training_args", {}).get("max_duration_seconds") is not None
            or any(result.get("training_args", {}).get(key) != value for key, value in settings.items())):
        raise ValueError(f"Incompatible model revision or training/evaluation settings in {path}")
    final = result.get("final_training", {})
    best_epoch = result.get("best_epoch")
    if (not isinstance(best_epoch, int) or not 1 <= best_epoch <= settings["epochs"]
            or final.get("split") != "train+validation" or final.get("checkpoint") != "final.pt"
            or final.get("epochs") != result.get("best_epoch")
            or final.get("schedule_horizon_epochs") != settings["epochs"]
            or len(result.get("refit_history", [])) != result.get("best_epoch")
            or (require_checkpoint and not path.with_name("final.pt").is_file())):
        raise ValueError(f"Missing or incompatible train+validation refit in {path}")
    if require_predictions and not path.with_name("test_predictions.npz").is_file():
        raise ValueError(f"Missing saved test predictions for {path}")
    return result


def write_table(output, seed, num_bootstrap):
    from metrics import bootstrap_metric_scores, score_predictions

    summary = []
    reference_targets = {}
    for model in MODELS:
        key, label, pretraining, model_id = model
        row = dict(method=label, pretraining=pretraining, model_id=model_id, seed=seed,
                   num_training_seeds=1, num_bootstrap=num_bootstrap)
        for task in TASKS:
            metric_task = "detection" if task == "detection" else "classification"
            path = output / key / task / f"seed_{seed}/results.json"
            entry = load_result(path, model, task, seed, require_checkpoint=False)
            prediction_path = path.with_name("test_predictions.npz")
            with np.load(prediction_path, allow_pickle=False) as predictions:
                y_true, y_pred = predictions["y_true"], predictions["y_pred"]
            if (y_true.shape != y_pred.shape or len(y_true) == 0
                    or not np.isfinite(y_pred).all()
                    or (metric_task == "classification" and y_true.ndim != 1)
                    or (task == "detection" and (y_true.ndim != 2 or y_true.shape[1] != 7))):
                raise ValueError(f"Invalid test predictions in {prediction_path}")
            if task in reference_targets and not np.array_equal(y_true, reference_targets[task]):
                raise ValueError(f"Test target order differs across models: {prediction_path}")
            reference_targets[task] = y_true
            metric = "map" if task == "detection" else "macro_f1"
            score = score_predictions(metric_task, y_true, y_pred)
            if not np.isclose(score, entry["test"][metric], rtol=0, atol=1e-7):
                raise ValueError(f"Saved predictions disagree with test score in {path}")
            scores = bootstrap_metric_scores(metric_task, y_true, y_pred, num_bootstrap, seed)
            row[f"{task}_score"] = score
            row[f"{task}_std"] = float(np.std(scores, ddof=1))
            np.savez_compressed(path.with_name("bootstrap_scores.npz"), scores=np.asarray(scores))
        summary.append(row)
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved results: {output / 'results.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs/manuscript")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tables_only", action="store_true")
    parser.add_argument("--prepare_only", action="store_true",
                        help="Write and validate the manifest without training or building tables.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.num_bootstrap < 2:
        parser.error("Use a nonnegative seed and at least two bootstrap resamples.")
    output = args.output_dir.expanduser().resolve()
    # Keep the recorded campaign order: all/detection first, then balanced.
    jobs = [(model, task, make_command(output, model, task, args.seed))
            for model in MODELS for task in TASKS if task != "balanced"]
    jobs += [(model, "balanced", make_command(output, model, "balanced", args.seed)) for model in MODELS]
    if args.dry_run:
        for _, _, command in jobs:
            print(shlex.join(command))
        print(f"One training seed ({args.seed}); validation/test batch 1, FP32; "
              f"select epoch on validation, refit on train+validation; {args.num_bootstrap} test bootstrap resamples.")
        print(f"Then generate {output / 'results.csv'}")
        return
    manifest = dict(protocol="supervised_end_to_end_finetuning", models=MODELS,
                    seed=args.seed, num_training_seeds=1, num_bootstrap=args.num_bootstrap,
                    recipes={model[0]: {task: recipe(model[0], task) for task in TASKS}
                             for model in MODELS}, cuda_tf32=True)
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output / "run_config.json"
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not (args.resume or args.tables_only) or previous != manifest:
            parser.error("Existing output requires --resume and exactly matching settings.")
    elif args.tables_only:
        parser.error("--tables_only requires an existing run.")
    elif output.exists() and any(output.iterdir()):
        parser.error("Use a new or empty output directory.")
    environment = environment_record()
    env_path = output / "environment.json"
    if env_path.exists() and not args.tables_only and json.loads(env_path.read_text()) != environment:
        parser.error("The training environment changed; use a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not env_path.exists():
        env_path.write_text(json.dumps(environment, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared run: {output}", flush=True)
        return
    if not args.tables_only:
        for model, task, command in jobs:
            key, label, _, _ = model
            path = output / key / task / f"seed_{args.seed}/results.json"
            if path.is_file():
                load_result(path, model, task, args.seed, require_checkpoint=False)
                print(f"Already complete: {label} {task}, seed {args.seed}", flush=True)
                continue
            if path.parent.exists() and any(path.parent.iterdir()):
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archived = output / "interrupted" / key / task / f"seed_{args.seed}-{stamp}"
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path.parent), str(archived))
                print(f"Archived interrupted run: {archived}", flush=True)
            print(f"Training {label} {task}, seed {args.seed}", flush=True)
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True)
    write_table(output, args.seed, args.num_bootstrap)


if __name__ == "__main__":
    main()
