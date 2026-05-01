import argparse
import random
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
    get_detection_label_vector,
    load_classification_splits,
    load_detection_splits,
)
from metrics import MeanAveragePrecision
from models import MFCC, Aves, BioLingual, Dolph2Vec, SpectralFeatures, Spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
        "--model",
        choices=[
            "dolph2vec",
            "aves_core",
            "aves_bio",
            "aves_ow",
            "biolingual",
            "mfcc",
            "spectrogram",
            "spectral_features",
        ],
        default="dolph2vec",
    )

    parser.add_argument("--target_sample_rate", default=44100, type=int)

    return parser.parse_args()


def embed_split(dataset, split_name: str, dataset_name: str, model):
    embeddings = []
    labels = []
    for row in tqdm(dataset, desc=f"processing {split_name}", total=len(dataset)):
        audio = row["audio"]
        if dataset_name == "detection":
            label = np.asarray(get_detection_label_vector(row), dtype=np.int64)
        else:
            label = int(row["label"])
        try:
            embedding = model(audio)
            embeddings.append(embedding.cpu())
            labels.append(label)
        except Exception as e:
            audio_path = audio.get("path") if isinstance(audio, dict) else None
            print(f"error processing {audio_path or '<in-memory-audio>'}: {e}")

    if not embeddings:
        raise ValueError(f"No embeddings were produced for split {split_name}.")

    x = np.stack([embedding.numpy() for embedding in embeddings])
    y = np.asarray(labels)
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
    return float(accuracy_score(y_eval, clf.predict(x_eval)))


def predict_for_metric(dataset_name: str, clf, x_eval) -> np.ndarray:
    if dataset_name == "detection":
        return detection_probabilities(clf, x_eval)
    return clf.predict(x_eval)


def score_predictions(dataset_name: str, y_true, y_pred_or_score) -> float:
    if dataset_name == "detection":
        return detection_map_from_scores(y_pred_or_score, y_true)
    return float(accuracy_score(y_true, y_pred_or_score))


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
    inverse_regs = (
        args.inverse_regs if args.inverse_regs is not None else [args.inverse_reg]
    )

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    name2model = {
        "aves_core": Aves,
        "aves_bio": Aves,
        "aves_ow": Aves,
        "biolingual": BioLingual,
        "dolph2vec": Dolph2Vec,
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
    elif args.model == "aves_ow":
        aves_model_path, aves_config_path = get_aves_paths("ow")
        actual_sample_rate = get_aves_sample_rate("ow")
    else:
        aves_model_path, aves_config_path = "", ""

    model_args = dict(
        sample_rate=actual_sample_rate,
        dolph2vec_config_path=dolph2vec_config_path,
        dolph2vec_model_path=dolph2vec_base,
    )

    model_args["aves_model_path"] = aves_model_path
    model_args["aves_config_path"] = aves_config_path

    model = name2model[args.model](**model_args)

    if args.dataset_name == "detection":
        dataset, _ = load_detection_splits()
    else:
        dataset, _ = load_classification_splits()

    x_train, y_train = embed_split(dataset["train"], "train", args.dataset_name, model)
    x_validation, y_validation = embed_split(
        dataset["validation"], "validation", args.dataset_name, model
    )
    x_test, y_test = embed_split(dataset["test"], "test", args.dataset_name, model)

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
    for seed in seeds:
        set_seed(seed)
        clf = fit_classifier(
            args.dataset_name, seed, best_c, x_final_train, y_final_train
        )
        y_pred_or_score = predict_for_metric(
            args.dataset_name, clf, x_test_for_evaluation
        )
        final_scores.append(
            score_predictions(args.dataset_name, y_test, y_pred_or_score)
        )
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
    metric_name = "mAP" if args.dataset_name == "detection" else "Accuracy"
    c_grid = ", ".join(str(c) for c in inverse_regs)
    seed_grid = ", ".join(str(seed) for seed in seeds)

    result_text = (
        f"Model {args.model} on dataset {args.dataset_name}:\n"
        f"Best C = {best_c} "
        f"(validation {metric_name} = {best_validation_score:.4f})\n"
        f"Logistic Regression Test {metric_name}: {mean_score:.4f} ± {std_score:.4f}"
    )
    print(result_text)

    if args.dataset_name == "detection":
        with open(results_dir / "results_detection.txt", "a") as f:
            print(result_text, file=f)
    else:
        with open(results_dir / "results_classification.txt", "a") as f:
            print(result_text, file=f)


if __name__ == "__main__":
    main()
