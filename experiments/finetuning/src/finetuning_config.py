"""Fine-tuning settings aligned with the shared linear-probing benchmark."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT.parents[1] / "runs" / "finetuning"
sys.path.insert(0, str(ROOT.parent / "benchmark/src"))

from aves_models import AVES_SCRATCH_MODELS  # noqa: E402
from benchmark_config import EVALUATIONS, EVALUATION_SETTINGS  # noqa: E402
from benchmark_models import BENCHMARK_MODELS, result_model_id  # noqa: E402
from benchmark_revisions import MODEL_REVISIONS as HF_MODEL_REVISIONS  # noqa: E402

NEURAL_KEYS = {"aves_core", "biolingual", "aves_bio", "aves_scratch", "wav2vec2", "aves_continual"}
NEURAL_MODELS = tuple(model for model in BENCHMARK_MODELS if model["key"] in NEURAL_KEYS)
RESULT_MODEL_IDS = {model["key"]: result_model_id(model) for model in NEURAL_MODELS}
# The benchmark uses the short "biolingual" backend ID; training loads its Hub encoder.
MODELS = tuple((model["key"], model["label"], model["pretraining"],
                "davidrrobinson/biolingual" if model["key"] == "biolingual" else result_model_id(model))
               for model in NEURAL_MODELS)
MODEL_ID = next(model[3] for model in MODELS if model[0] == "wav2vec2")
MODEL_REVISIONS = {key: value for key, value in {
    **HF_MODEL_REVISIONS,
    **{spec["model_id"]: spec["revision"] for spec in AVES_SCRATCH_MODELS},
}.items() if key in {model[3] for model in MODELS}}
HF_MODELS = ("wav2vec2", "aves_continual")
TASK_DATASETS = {("balanced" if task == "classification" and config == "balanced" else task): (task, config)
                 for task, config in EVALUATIONS}
TASKS = tuple(TASK_DATASETS)
SEED = EVALUATION_SETTINGS["seed"]
NUM_BOOTSTRAP = EVALUATION_SETTINGS["num_bootstrap"]
CSV_FIELDS = ("model_label", "model_id", "dataset_name", "dataset_config", "metric",
              "validation_score", "best_epoch", "test_mean", "test_std",
              "num_training_seeds", "num_bootstrap", "seed")

COMMON = dict(learning_rate=1e-5, head_learning_rate=1e-4, weight_decay=0.01,
              warmup_ratio=0.1, dropout=0.1, max_grad_norm=1.0, mixed_precision="float16",
              eval_batch_size=EVALUATION_SETTINGS["batch_size"], eval_precision="none", num_workers=4,
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



def run_manifest():
    """Canonical run identity used by the launcher, resume checks and exporter."""
    return dict(
        protocol="supervised_end_to_end_finetuning", models=[list(model) for model in MODELS],
        seed=SEED, num_training_seeds=1, num_bootstrap=NUM_BOOTSTRAP,
        recipes={model[0]: {task: recipe(model[0], task) for task in TASKS} for model in MODELS},
        cuda_tf32=True)
