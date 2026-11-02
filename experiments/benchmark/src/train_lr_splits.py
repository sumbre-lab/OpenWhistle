"""Fit and evaluate a frozen-representation probe on the official dataset splits."""

import argparse
import csv
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from aves_artifacts import (
    get_aves_paths,
    get_aves_sample_rate,
)
from aves_models import AVES_SCRATCH_MODELS
from benchmark_config import CACHE_FINGERPRINT, RUNS_DIR
from benchmark_revisions import MODEL_REVISIONS
from hf_datasets import (
    CLASSIFICATION_BALANCED_CONFIG_NAME,
    CLASSIFICATION_LABEL_NAMES,
    get_detection_label_vector,
    load_classification_splits,
    load_detection_splits,
)
from metrics import bootstrap_metric_scores, score_predictions
from models import (
    MFCC,
    Aves,
    BioLingual,
    HuggingFaceAudioBackbone,
    HuggingFaceAvesBackbone,
    SpectralFeatures,
    Spectrogram,
)
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

LR_MAX_ITER = 20000


def set_seed(seed: int = 42):
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--inverse_regs",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0],
        help=(
            "Candidate C values. The best C is selected on the validation split. "
            "Defaults to the published evaluation grid: 0.1, 1, 10."
        ),
    )
    parser.add_argument(
        "--num_seeds",
        default=1,
        type=int,
        help="Number of consecutive bootstrap RNG seeds starting at --seed; the deterministic LR is fitted once.",
    )
    parser.add_argument(
        "--num_bootstrap",
        default=1000,
        type=int,
        help=(
            "Number of bootstrap resamples per RNG seed used to estimate "
            "the reported standard deviation."
        ),
    )

    parser.add_argument(
        "--no_normalize_data",
        dest="normalize_data",
        action="store_false",
        default=True,
        help="Disable feature standardization before logistic regression.",
    )

    parser.add_argument(
        "--dataset_name",
        choices=[
            "classification",
            "detection",
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset_config",
        default=None,
        help=(
            "Dataset subset/config. Classification supports all and balanced; "
            "detection defaults to default."
        ),
    )

    parser.add_argument(
        "--model",
        choices=[
            "hf",
            "aves_hf",
            "aves_core",
            "aves_bio",
            "biolingual",
            "mfcc",
            "spectrogram",
            "spectral_features",
        ],
        required=True,
    )
    parser.add_argument(
        "--hf_model_id",
        default=None,
        help="Private or public Hub model ID. Required when --model hf is used.",
    )
    parser.add_argument(
        "--hf_sample_rate",
        type=int,
        default=None,
        help="Override an incorrect sampling rate in a Hub preprocessor config.",
    )
    parser.add_argument(
        "--hf_local_dir",
        type=Path,
        default=None,
        help=(
            "Local native AVES export with models/ and config/ folders. "
            "Requires --model aves_hf; files are checked against the pinned Hub hashes."
        ),
    )
    parser.add_argument(
        "--hf_feature_mode",
        choices=("standard", "btb3_concat", "btb3_mean"),
        default="standard",
        help="Input transform and band-pooling mode for Hugging Face backbones.",
    )
    parser.add_argument(
        "--model_label",
        default=None,
        help="Display label stored in the structured results CSV.",
    )
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=None,
        help="Optional structured CSV. The model/task row is updated atomically.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=None,
        help="Save test predictions and split labels for portable table regeneration.",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=1,
        help="Clips per extraction batch; native AVES/BioLingual require --batch_native_models.",
    )
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA automatic mixed precision for embedding extraction.",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--audio_workers",
        type=int,
        default=0,
        help="Threads used to decode/resample audio before each GPU batch.",
    )
    parser.add_argument(
        "--embedding_cache_dir",
        type=Path,
        default=None,
        help="Optional directory used to resume cached split embeddings.",
    )

    parser.add_argument("--target_sample_rate", default=44100, type=int)
    parser.add_argument(
        "--batch_native_models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use real neural batches for native AVES and BioLingual; single-clip inference otherwise.",
    )
    parser.add_argument(
        "--strict_embedding_batches",
        action="store_true",
        default=True,
        help="Stop on extraction failures instead of changing batch composition by splitting batches.",
    )
    parser.add_argument(
        "--hf_local_files_only",
        action="store_true",
        help="Load model artifacts from the local Hub cache; dataset downloads remain enabled.",
    )

    args = parser.parse_args()
    if args.seed < 0 or args.num_seeds < 1 or args.num_bootstrap < 2:
        parser.error(
            "Seed must be nonnegative; use at least one RNG seed and two bootstrap resamples."
        )
    if args.embedding_batch_size < 1 or args.audio_workers < 0:
        parser.error("Batch size must be positive and audio_workers nonnegative.")
    if any(c <= 0 for c in args.inverse_regs):
        parser.error("All C values must be positive.")
    return args


RESULT_FIELDS = (
    "model_label",
    "model_id",
    "family",
    "dataset_name",
    "dataset_config",
    "metric",
    "validation_score",
    "best_c",
    "test_mean",
    "test_std",
    "seed_std",
    "bootstrap_std",
    "num_seeds",
    "num_bootstrap",
    "seed",
    "normalize_data",
    "embedding_batch_size",
    "mixed_precision",
    "amp_dtype",
    "feature_mode",
    "target_sample_rate",
    "embedding_mode",
    "strict_embedding_batches",
)


def upsert_result_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if path.exists():
        with path.open(newline="") as result_file:
            existing_rows = list(csv.DictReader(result_file))

    key = (
        str(row["model_id"]),
        str(row["dataset_name"]),
        str(row["dataset_config"]),
    )
    existing_rows = [
        existing
        for existing in existing_rows
        if (
            existing.get("model_id"),
            existing.get("dataset_name"),
            existing.get("dataset_config", ""),
        )
        != key
    ]
    existing_rows.append({field: row.get(field, "") for field in RESULT_FIELDS})

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)
    temporary_path.replace(path)


def label_for_row(row: dict, dataset_name: str):
    if dataset_name == "detection":
        return np.asarray(get_detection_label_vector(row), dtype=np.int64)
    return int(row["label"])


def embed_split(
    dataset,
    split_name: str,
    dataset_name: str,
    model,
    batch_size: int = 1,
    strict_batches: bool = True,
):
    embeddings = []
    labels = []
    supports_batches = hasattr(model, "embed_batch")
    effective_batch_size = max(1, batch_size) if supports_batches else 1
    pending_audio = []
    pending_labels = []

    def embed_audio_batch(audio_batch, label_batch):
        try:
            if supports_batches:
                batch_embeddings = model.embed_batch(audio_batch)
                embeddings.extend(batch_embeddings.detach().cpu())
                labels.extend(label_batch)
            else:
                embeddings.append(model(audio_batch[0]).detach().cpu())
                labels.append(label_batch[0])
        except Exception as batch_error:
            if strict_batches:
                raise RuntimeError(
                    f"Embedding failed in {split_name} for a batch of {len(audio_batch)} clips; "
                    "strict batching refuses to split this batch; refusing a partial benchmark."
                ) from batch_error
            audio = audio_batch[0]
            audio_path = audio.get("path") if isinstance(audio, dict) else None
            raise RuntimeError(
                f"Embedding failed in {split_name} for "
                f"{audio_path or '<in-memory-audio>'}; refusing a partial benchmark."
            ) from batch_error

    def flush_batch():
        if not pending_audio:
            return
        embed_audio_batch(pending_audio, pending_labels)
        pending_audio.clear()
        pending_labels.clear()

    with tqdm(desc=f"processing {split_name}", total=len(dataset)) as progress:
        for row in dataset:
            pending_audio.append(row["audio"])
            pending_labels.append(label_for_row(row, dataset_name))
            if len(pending_audio) >= effective_batch_size:
                processed_count = len(pending_audio)
                flush_batch()
                progress.update(processed_count)
        processed_count = len(pending_audio)
        flush_batch()
        progress.update(processed_count)

    if not embeddings:
        raise ValueError(f"No embeddings were produced for split {split_name}.")

    x = np.stack([embedding.numpy() for embedding in embeddings])
    y = np.asarray(labels)
    return x, y


def embedding_cache_path(
    cache_dir: Path,
    model_id: str,
    dataset_name: str,
    dataset_config: str,
    split_name: str,
) -> Path:
    safe_model_id = re.sub(r"[^A-Za-z0-9._-]+", "__", model_id)
    safe_config = re.sub(r"[^A-Za-z0-9._-]+", "__", dataset_config)
    return (
        cache_dir
        / CACHE_FINGERPRINT
        / safe_model_id
        / dataset_name
        / safe_config
        / f"{split_name}.npz"
    )


def resolved_embedding_model_id(args, model):
    model_id = args.hf_model_id or args.model
    if args.model == "aves_hf":
        model_id += f"__aves_hf__{model.revision}"
    if args.hf_feature_mode != "standard":
        model_id += f"__{args.hf_feature_mode}"
    if args.batch_native_models and args.model in {
        "aves_hf",
        "aves_core",
        "aves_bio",
        "biolingual",
    }:
        model_id += "__native_batched"
    if args.strict_embedding_batches:
        precision = "fp16" if getattr(model, "mixed_precision", False) else "fp32"
        model_id += f"__strict_batch{args.embedding_batch_size}_{precision}"
    return model_id


def load_complete_cached_embeddings(args, dataset_config, sample_rate):
    """Reuse a complete strict FP32 cache before loading weights or raw audio."""
    # Non-strict and AMP cache identities depend on runtime backbone behavior.
    if (
        args.embedding_cache_dir is None
        or not args.strict_embedding_batches
        or args.mixed_precision
    ):
        return None
    metadata = SimpleNamespace(
        sample_rate=sample_rate, mixed_precision=False, feature_mode="standard"
    )
    if args.model == "aves_hf":
        spec = next(
            (
                spec
                for spec in AVES_SCRATCH_MODELS
                if spec["model_id"] == args.hf_model_id
            ),
            None,
        )
        if spec is None:
            return None
        metadata.revision = spec["revision"]
        metadata.sample_rate = spec["sample_rate"]
    elif args.model == "biolingual":
        metadata.sample_rate = 48000
    if args.model == "hf":
        metadata.feature_mode = args.hf_feature_mode
    model_id = resolved_embedding_model_id(args, metadata)
    paths = [
        embedding_cache_path(
            args.embedding_cache_dir, model_id, args.dataset_name, dataset_config, split
        )
        for split in ("train", "validation", "test")
    ]
    if not all(path.is_file() for path in paths):
        return None
    if args.model == "hf":
        if args.hf_sample_rate is not None:
            metadata.sample_rate = args.hf_sample_rate
        else:
            # Only the small preprocessor config is needed for accurate CSV
            # metadata; encoder weights and raw datasets are unnecessary.
            from transformers import AutoFeatureExtractor

            extractor = AutoFeatureExtractor.from_pretrained(
                args.hf_model_id,
                revision=MODEL_REVISIONS.get(args.hf_model_id),
                local_files_only=args.hf_local_files_only,
            )
            metadata.sample_rate = int(getattr(extractor, "sampling_rate", 44100))
    arrays = []
    for path in paths:
        with np.load(path) as cached:
            arrays.extend((cached["x"], cached["y"]))
        print(f"Loaded cached embeddings: {path}")
    return metadata, arrays


def embed_or_load_split(
    dataset,
    split_name: str,
    dataset_name: str,
    dataset_config: str,
    model,
    model_id: str,
    batch_size: int,
    cache_dir: Path | None,
    strict_batches: bool = True,
):
    cache_path = (
        embedding_cache_path(
            cache_dir,
            model_id,
            dataset_name,
            dataset_config,
            split_name,
        )
        if cache_dir is not None
        else None
    )
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            print(f"Loaded cached embeddings: {cache_path}")
            return cached["x"], cached["y"]

    x, y = embed_split(
        dataset,
        split_name,
        dataset_name,
        model,
        batch_size=batch_size,
        strict_batches=strict_batches,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_path, x=x, y=y)
        temporary_path.replace(cache_path)
        print(f"Saved cached embeddings: {cache_path}")
    return x, y


def normalize_train_eval(x_train, x_eval, normalize_data: bool):
    if not normalize_data:
        return x_train, x_eval
    scaler = StandardScaler()
    return (
        scaler.fit_transform(x_train),
        scaler.transform(x_eval),
    )


def fit_classifier(dataset_name: str, seed: int, c: float, x_train, y_train):
    if dataset_name == "detection":
        base_clf = LogisticRegression(max_iter=LR_MAX_ITER, random_state=seed, C=c)
        clf = MultiOutputClassifier(base_clf)
    else:
        clf = LogisticRegression(max_iter=LR_MAX_ITER, random_state=seed, C=c)
    clf.fit(x_train, y_train)
    return clf


def detection_probabilities(clf, x_eval) -> np.ndarray:
    y_score = clf.predict_proba(x_eval)
    return np.array([np.asarray(scores)[:, 1] for scores in y_score]).T


def evaluate_classifier(dataset_name: str, clf, x_eval, y_eval) -> float:
    return score_predictions(
        dataset_name, y_eval, predict_for_metric(dataset_name, clf, x_eval)
    )


def predict_for_metric(dataset_name: str, clf, x_eval) -> np.ndarray:
    if dataset_name == "detection":
        return detection_probabilities(clf, x_eval)
    return clf.predict(x_eval)


def main():
    args = get_args()
    set_seed(args.seed)
    inverse_regs = args.inverse_regs
    if args.results_csv is None:
        args.results_csv = RUNS_DIR / "single-model/results.csv"

    name2model = {
        "aves_core": Aves,
        "aves_bio": Aves,
        "biolingual": BioLingual,
        "hf": HuggingFaceAudioBackbone,
        "aves_hf": HuggingFaceAvesBackbone,
        "mfcc": MFCC,
        "spectrogram": Spectrogram,
        "spectral_features": SpectralFeatures,
    }

    actual_sample_rate = args.target_sample_rate
    if args.model == "aves_bio":
        aves_model_path, aves_config_path = get_aves_paths("bio", download=False)
        actual_sample_rate = get_aves_sample_rate("bio")
    elif args.model == "aves_core":
        aves_model_path, aves_config_path = get_aves_paths("core", download=False)
        actual_sample_rate = get_aves_sample_rate("core")
    else:
        aves_model_path, aves_config_path = "", ""

    model_args = dict(
        sample_rate=actual_sample_rate,
        hf_model_id=args.hf_model_id,
        hf_sample_rate=args.hf_sample_rate,
        mixed_precision=args.mixed_precision,
        amp_dtype=args.amp_dtype,
        audio_workers=args.audio_workers,
        hf_feature_mode=args.hf_feature_mode,
        batch_native_models=args.batch_native_models,
        hf_local_files_only=args.hf_local_files_only,
    )

    model_args["aves_model_path"] = aves_model_path
    model_args["aves_config_path"] = aves_config_path

    if args.model in {"hf", "aves_hf"} and not args.hf_model_id:
        raise ValueError("--hf_model_id is required for a Hugging Face backbone.")
    if args.hf_local_dir is not None:
        if args.model != "aves_hf":
            raise ValueError("--hf_local_dir requires --model aves_hf.")
        model_args["hf_local_dir"] = args.hf_local_dir
    if args.dataset_name == "detection":
        dataset_config = args.dataset_config or "default"
        if dataset_config != "default":
            raise ValueError("Detection only supports --dataset_config default.")
    else:
        dataset_config = args.dataset_config or CLASSIFICATION_BALANCED_CONFIG_NAME
        if dataset_config not in CLASSIFICATION_LABEL_NAMES:
            raise ValueError(
                "Classification supports only all and balanced configurations."
            )

    cached = load_complete_cached_embeddings(args, dataset_config, actual_sample_rate)
    if cached is not None:
        model, arrays = cached
        x_train, y_train, x_validation, y_validation, x_test, y_test = arrays
    else:
        if args.model in {"aves_core", "aves_bio"}:
            model_args["aves_model_path"], model_args["aves_config_path"] = (
                get_aves_paths("core" if args.model == "aves_core" else "bio")
            )
        model = name2model[args.model](**model_args)
        if args.dataset_name == "detection":
            dataset, _ = load_detection_splits()
        else:
            dataset, _ = load_classification_splits(config_name=dataset_config)
        resolved_model_id = resolved_embedding_model_id(args, model)
        arrays = []
        for split in ("train", "validation", "test"):
            arrays.extend(
                embed_or_load_split(
                    dataset[split],
                    split,
                    args.dataset_name,
                    dataset_config,
                    model,
                    resolved_model_id,
                    args.embedding_batch_size,
                    args.embedding_cache_dir,
                    strict_batches=args.strict_embedding_batches,
                )
            )
        x_train, y_train, x_validation, y_validation, x_test, y_test = arrays

    validation_scores = []
    x_train_for_validation, x_validation_for_selection = normalize_train_eval(
        x_train, x_validation, args.normalize_data
    )
    for c in inverse_regs:
        clf = fit_classifier(
            args.dataset_name, args.seed, c, x_train_for_validation, y_train
        )
        score = evaluate_classifier(
            args.dataset_name, clf, x_validation_for_selection, y_validation
        )
        validation_scores.append((c, score))

    best_c, best_validation_score = max(validation_scores, key=lambda item: item[1])

    x_final_train = np.concatenate([x_train, x_validation], axis=0)
    y_final_train = np.concatenate([y_train, y_validation], axis=0)
    x_final_train, x_test_for_evaluation = normalize_train_eval(
        x_final_train, x_test, args.normalize_data
    )

    final_bootstrap_scores = []
    seeds = [args.seed + offset for offset in range(args.num_seeds)]
    # The default lbfgs LogisticRegression solver is deterministic: random_state
    # does not change the fitted classifier. Fit once, then retain all requested
    # RNG seeds for the bootstrap resampling.
    set_seed(seeds[0])
    clf = fit_classifier(
        args.dataset_name,
        seeds[0],
        best_c,
        x_final_train,
        y_final_train,
    )
    y_pred_or_score = predict_for_metric(
        args.dataset_name,
        clf,
        x_test_for_evaluation,
    )
    final_score = score_predictions(
        args.dataset_name,
        y_test,
        y_pred_or_score,
    )
    for seed in seeds:
        final_bootstrap_scores.extend(
            bootstrap_metric_scores(
                args.dataset_name,
                y_test,
                y_pred_or_score,
                args.num_bootstrap,
                seed,
            )
        )

    mean_score = final_score
    seed_std_score = 0.0  # A single deterministic probe is fitted.
    bootstrap_std_score = (
        float(np.std(final_bootstrap_scores, ddof=1))
        if len(final_bootstrap_scores) > 1
        else 0.0
    )
    std_score = bootstrap_std_score
    if args.predictions_dir is not None:
        model_id = args.hf_model_id or args.model
        safe_model_id = re.sub(r"[^A-Za-z0-9._-]+", "__", model_id)
        prediction_path = (
            args.predictions_dir
            / safe_model_id
            / args.dataset_name
            / f"{dataset_config}.npz"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            y=y_test,
            predictions=y_pred_or_score,
            test_score=final_score,
            best_c=best_c,
            validation_score=best_validation_score,
        )
        temporary.replace(prediction_path)
        # Keep compact ordered labels independently of the embedding cache.
        # Every representation must see the same examples in the same order.
        label_root = (
            args.predictions_dir.parent / "labels" / args.dataset_name / dataset_config
        )
        label_root.mkdir(parents=True, exist_ok=True)
        for split, targets in (
            ("train", y_train),
            ("validation", y_validation),
            ("test", y_test),
        ):
            label_path = label_root / f"{split}.npz"
            if label_path.exists():
                with np.load(label_path) as existing:
                    if not np.array_equal(existing["y"], targets):
                        raise ValueError(
                            f"Split targets or their order differ: {label_path}"
                        )
            else:
                temporary = label_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, y=targets)
                temporary.replace(label_path)
    metric_name = "mAP" if args.dataset_name == "detection" else "Macro-F1"

    result_text = (
        f"Model {args.hf_model_id or args.model} on dataset "
        f"{args.dataset_name}/{dataset_config}:\n"
        f"Best C = {best_c} "
        f"(validation {metric_name} = {best_validation_score:.4f})\n"
        f"Logistic Regression Test {metric_name}: {mean_score:.4f} ± {std_score:.4f}"
    )
    print(result_text)

    if args.results_csv is not None:
        model_id = args.hf_model_id or args.model
        model_label = args.model_label or model_id
        family = (
            "AVES-Bio"
            if "aves" in model_id.lower()
            else "Wav2Vec2.0"
            if "wav2vec2" in model_id.lower()
            else args.model
        )
        upsert_result_csv(
            args.results_csv,
            {
                "model_label": model_label,
                "model_id": model_id,
                "family": family,
                "dataset_name": args.dataset_name,
                "dataset_config": dataset_config,
                "metric": metric_name,
                "validation_score": best_validation_score,
                "best_c": best_c,
                "test_mean": mean_score,
                "test_std": std_score,
                "seed_std": seed_std_score,
                "bootstrap_std": bootstrap_std_score,
                "num_seeds": args.num_seeds,
                "num_bootstrap": args.num_bootstrap,
                "seed": args.seed,
                "normalize_data": args.normalize_data,
                "embedding_batch_size": args.embedding_batch_size,
                "mixed_precision": getattr(
                    model, "mixed_precision", args.mixed_precision
                ),
                "amp_dtype": args.amp_dtype,
                "feature_mode": getattr(model, "feature_mode", "standard"),
                "target_sample_rate": getattr(model, "sample_rate", actual_sample_rate),
                "embedding_mode": (
                    "batched"
                    if args.model == "hf"
                    or (
                        args.batch_native_models
                        and args.model
                        in {"aves_hf", "aves_core", "aves_bio", "biolingual"}
                    )
                    else "individual"
                ),
                "strict_embedding_batches": args.strict_embedding_batches,
            },
        )


if __name__ == "__main__":
    main()
