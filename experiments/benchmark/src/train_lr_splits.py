import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import torch
from conf import (
    dolph2vec_config_path,
    dolph2vec_base,
    get_aves_paths,
    get_aves_sample_rate,
)
from hf_datasets import (
    CLASSIFICATION_BALANCED_CONFIG_NAME,
    get_detection_label_vector,
    load_classification_splits,
    load_detection_splits,
)
from metrics import MeanAveragePrecision
from pipeline_config import EMBEDDING_PIPELINE_VERSION
from models import (
    MFCC,
    Aves,
    BioLingual,
    Dolph2Vec,
    HuggingFaceAudioBackbone,
    HuggingFaceAvesBackbone,
    SpectralFeatures,
    Spectrogram,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--inverse_reg", default=1.0, type=float)
    parser.add_argument(
        "--inverse_regs",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Candidate C values. The best C is selected on the validation split. "
            "Defaults to --inverse_reg for backward compatibility."
        ),
    )
    parser.add_argument(
        "--num_seeds",
        default=10,
        type=int,
        help="Number of final logistic-regression seeds evaluated on the test split.",
    )
    parser.add_argument(
        "--num_bootstrap",
        default=1000,
        type=int,
        help=(
            "Number of bootstrap resamples of the test set used to estimate "
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
    )
    parser.add_argument(
        "--dataset_config", "--classification_config",
        default=None,
        help=(
            "Dataset subset/config. Classification supports balanced, unbalanced, "
            "and all; detection defaults to default."
        ),
    )

    parser.add_argument(
        "--model",
        choices=[
            "dolph2vec",
            "hf",
            "aves_core",
            "aves_bio",
            "biolingual",
            "mfcc",
            "spectrogram",
            "spectral_features",
        ],
        default="dolph2vec",
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
        "--embedding_batch_size",
        type=int,
        default=256,
        help="Batch size for Transformers backbones; ignored by legacy extractors.",
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
        "--fast_cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable cuDNN autotuning; fastest, but not bitwise deterministic.",
    )
    parser.add_argument(
        "--audio_workers",
        type=int,
        default=8,
        help="Threads used to decode/resample audio before each GPU batch.",
    )
    parser.add_argument(
        "--embedding_cache_dir",
        type=Path,
        default=None,
        help="Optional directory used to resume cached split embeddings.",
    )

    parser.add_argument("--target_sample_rate", default=44100, type=int)

    return parser.parse_args()


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
    "normalize_data",
    "embedding_batch_size",
    "mixed_precision",
    "amp_dtype",
    "feature_mode",
    "embedding_pipeline_version",
    "target_sample_rate",
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
            if len(audio_batch) > 1:
                if isinstance(batch_error, torch.OutOfMemoryError):
                    torch.cuda.empty_cache()
                midpoint = len(audio_batch) // 2
                embed_audio_batch(
                    audio_batch[:midpoint],
                    label_batch[:midpoint],
                )
                embed_audio_batch(
                    audio_batch[midpoint:],
                    label_batch[midpoint:],
                )
                return

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
        / EMBEDDING_PIPELINE_VERSION
        / safe_model_id
        / dataset_name
        / safe_config
        / f"{split_name}.npz"
    )


def embed_or_load_split(
    dataset,
    split_name: str,
    dataset_name: str,
    dataset_config: str,
    model,
    model_id: str,
    batch_size: int,
    cache_dir: Path | None,
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
        cached = np.load(cache_path)
        print(f"Loaded cached embeddings: {cache_path}")
        return cached["x"], cached["y"]

    x, y = embed_split(
        dataset,
        split_name,
        dataset_name,
        model,
        batch_size=batch_size,
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


def detection_map_from_scores(y_score, y_eval) -> float:
    map_metric = MeanAveragePrecision()
    map_metric.update(y_score, y_eval)
    return float(map_metric.get_primary_metric())


def detection_map_score(clf, x_eval, y_eval) -> float:
    return detection_map_from_scores(detection_probabilities(clf, x_eval), y_eval)


def evaluate_classifier(dataset_name: str, clf, x_eval, y_eval) -> float:
    if dataset_name == "detection":
        return detection_map_score(clf, x_eval, y_eval)
    return float(
        f1_score(
            y_eval,
            clf.predict(x_eval),
            average="macro",
            zero_division=0,
        )
    )


def predict_for_metric(dataset_name: str, clf, x_eval) -> np.ndarray:
    if dataset_name == "detection":
        return detection_probabilities(clf, x_eval)
    return clf.predict(x_eval)


def score_predictions(dataset_name: str, y_true, y_pred_or_score) -> float:
    if dataset_name == "detection":
        return detection_map_from_scores(y_pred_or_score, y_true)
    return float(
        f1_score(
            y_true,
            y_pred_or_score,
            average="macro",
            zero_division=0,
        )
    )


def bootstrap_metric_scores(
    dataset_name: str,
    y_true,
    y_pred_or_score,
    num_bootstrap: int,
    seed: int,
) -> list[float]:
    if num_bootstrap <= 0:
        return []

    rng = np.random.default_rng(seed)
    n_examples = len(y_true)
    bootstrap_scores = []
    for _ in range(num_bootstrap):
        indices = rng.integers(0, n_examples, size=n_examples)
        bootstrap_scores.append(
            score_predictions(
                dataset_name,
                y_true[indices],
                y_pred_or_score[indices],
            )
        )
    return bootstrap_scores


def main():
    args = get_args()
    set_seed(args.seed)
    if args.fast_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    inverse_regs = (
        args.inverse_regs if args.inverse_regs is not None else [args.inverse_reg]
    )

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    name2model = {
        "aves_core": Aves,
        "aves_bio": Aves,
        "biolingual": BioLingual,
        "dolph2vec": Dolph2Vec,
        "hf": HuggingFaceAudioBackbone,
        "aves_hf": HuggingFaceAvesBackbone,
        "mfcc": MFCC,
        "spectrogram": Spectrogram,
        "spectral_features": SpectralFeatures,
    }

    actual_sample_rate = args.target_sample_rate
    if args.model == "aves_bio":
        aves_model_path, aves_config_path = get_aves_paths("bio")
        actual_sample_rate = get_aves_sample_rate("bio")
    elif args.model == "aves_core":
        aves_model_path, aves_config_path = get_aves_paths("core")
        actual_sample_rate = get_aves_sample_rate("core")
    else:
        aves_model_path, aves_config_path = "", ""

    model_args = dict(
        sample_rate=actual_sample_rate,
        dolph2vec_config_path=dolph2vec_config_path,
        dolph2vec_model_path=dolph2vec_base,
        hf_model_id=args.hf_model_id,
        hf_sample_rate=args.hf_sample_rate,
        mixed_precision=args.mixed_precision,
        amp_dtype=args.amp_dtype,
        audio_workers=args.audio_workers,
        hf_feature_mode=args.hf_feature_mode,
    )

    model_args["aves_model_path"] = aves_model_path
    model_args["aves_config_path"] = aves_config_path

    if args.model in {"hf", "aves_hf"} and not args.hf_model_id:
        raise ValueError("--hf_model_id is required for a Hugging Face backbone.")
    model = name2model[args.model](**model_args)

    if args.dataset_name == "detection":
        dataset_config = args.dataset_config or "default"
        if dataset_config != "default":
            raise ValueError("Detection only supports --dataset_config default.")
        dataset, _ = load_detection_splits()
    else:
        dataset_config = args.dataset_config or CLASSIFICATION_BALANCED_CONFIG_NAME
        dataset, _ = load_classification_splits(config_name=dataset_config)

    resolved_model_id = args.hf_model_id or args.model
    if args.model == "aves_hf":
        resolved_model_id = f"{resolved_model_id}__aves_hf__{model.revision}"
    if args.hf_feature_mode != "standard":
        resolved_model_id = f"{resolved_model_id}__{args.hf_feature_mode}"
    x_train, y_train = embed_or_load_split(
        dataset["train"],
        "train",
        args.dataset_name,
        dataset_config,
        model,
        resolved_model_id,
        args.embedding_batch_size,
        args.embedding_cache_dir,
    )
    x_validation, y_validation = embed_or_load_split(
        dataset["validation"],
        "validation",
        args.dataset_name,
        dataset_config,
        model,
        resolved_model_id,
        args.embedding_batch_size,
        args.embedding_cache_dir,
    )
    x_test, y_test = embed_or_load_split(
        dataset["test"],
        "test",
        args.dataset_name,
        dataset_config,
        model,
        resolved_model_id,
        args.embedding_batch_size,
        args.embedding_cache_dir,
    )

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

    final_scores = []
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
        final_scores.append(final_score)
        final_bootstrap_scores.extend(
            bootstrap_metric_scores(
                args.dataset_name,
                y_test,
                y_pred_or_score,
                args.num_bootstrap,
                seed,
            )
        )

    mean_score = float(np.mean(final_scores))
    seed_std_score = float(np.std(final_scores, ddof=1)) if len(final_scores) > 1 else 0.0
    bootstrap_std_score = (
        float(np.std(final_bootstrap_scores, ddof=1))
        if len(final_bootstrap_scores) > 1
        else 0.0
    )
    std_score = bootstrap_std_score if final_bootstrap_scores else seed_std_score
    metric_name = "mAP" if args.dataset_name == "detection" else "Macro-F1"
    c_grid = ", ".join(str(c) for c in inverse_regs)
    seed_grid = ", ".join(str(seed) for seed in seeds)

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
                "validation_score": f"{best_validation_score:.8f}",
                "best_c": best_c,
                "test_mean": f"{mean_score:.8f}",
                "test_std": f"{std_score:.8f}",
                "seed_std": f"{seed_std_score:.8f}",
                "bootstrap_std": f"{bootstrap_std_score:.8f}",
                "num_seeds": args.num_seeds,
                "num_bootstrap": args.num_bootstrap,
                "normalize_data": args.normalize_data,
                "embedding_batch_size": args.embedding_batch_size,
                "mixed_precision": getattr(model, "mixed_precision", args.mixed_precision),
                "amp_dtype": args.amp_dtype,
                "feature_mode": getattr(model, "feature_mode", "standard"),
                "embedding_pipeline_version": EMBEDDING_PIPELINE_VERSION,
                "target_sample_rate": getattr(model, "sample_rate", actual_sample_rate),
            },
        )

    if args.results_csv is None:
        if args.dataset_name == "detection":
            with open(results_dir / "results_detection.txt", "a") as f:
                print(result_text, file=f)
        else:
            with open(results_dir / "results_classification.txt", "a") as f:
                print(result_text, file=f)


if __name__ == "__main__":
    main()
