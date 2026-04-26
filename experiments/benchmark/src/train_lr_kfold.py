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
    load_classification_examples,
    load_detection_examples,
)
from metrics import MeanAveragePrecision
from models import MFCC, Aves, BioLingual, Dolph2Vec, SpectralFeatures, Spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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
    parser.add_argument("--kfold", default=5, type=int)

    parser.add_argument(
        "--no_normalize_data",
        dest="normalize_data",
        action="store_false",
        default=True,
        help="Disable feature standardization before logistic regression.",
    )

    parser.add_argument(
        "--dataset_name", 
        choices=["classification", 
                 "detection",
                 ]
    )

    parser.add_argument(
        "--model",
        choices=[
            "dolph2vec",
            "aves_core",
            "aves_bio",
            "biolingual",
            "mfcc",
            "spectrogram",
            "spectral_features",
        ],
        default="dolph2vec",
    )

    parser.add_argument("--target_sample_rate", default=44100, type=int)

    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    name2model = {
        "aves_core": Aves,
        "aves_bio": Aves,
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
        dataset, _ = load_detection_examples()
    else:
        dataset, _ = load_classification_examples()

    embeddings = []
    labels = []
    for row in tqdm(dataset, desc="processing audio files", total=len(dataset)):
        audio = row["audio"]
        if args.dataset_name == "detection":
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

    x_train = np.stack([embedding.numpy() for embedding in embeddings])
    y_train = np.array(labels)

    if args.dataset_name == "detection":
        kf = MultilabelStratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    else:
        kf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    accuracies = []
    map_scores = []

    for train_index, test_index in kf.split(x_train, y_train):
        X_tr, X_val = x_train[train_index], x_train[test_index]
        y_tr, y_val = y_train[train_index], y_train[test_index]

        if args.normalize_data:
            fold_scaler = StandardScaler()
            X_tr = fold_scaler.fit_transform(X_tr)
            X_val = fold_scaler.transform(X_val)

        if args.dataset_name == "detection":
            # Use MultiOutputClassifier for detection task
            base_clf = LogisticRegression(max_iter=LR_MAX_ITER, random_state=args.seed, C=args.inverse_reg)
            clf = MultiOutputClassifier(base_clf)
            clf.fit(X_tr, y_tr)
            
            y_score = clf.predict_proba(X_val)
            y_score = np.array([np.asarray(scores)[:, 1] for scores in y_score]).T
            
            map_metric = MeanAveragePrecision()
            map_metric.update(y_score, y_val)
            map_score = map_metric.get_primary_metric()
            map_scores.append(map_score)

        else:  # classification
            clf = LogisticRegression(max_iter=LR_MAX_ITER, random_state=args.seed, C=args.inverse_reg)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)

    if args.dataset_name == "detection":
        mean_map = np.mean(map_scores)
        std_map = np.std(map_scores)
        print(
            f"Model {args.model} on dataset {args.dataset_name} with C = {args.inverse_reg}:\n"
            f"Logistic Regression K-Fold mAP: {mean_map:.4f} ± {std_map:.4f}"
        )
        with open(results_dir / "results_detection.txt", "a") as f:
            print(
                f"Model {args.model} on dataset {args.dataset_name} with C = {args.inverse_reg}:\n"
                f"Logistic Regression K-Fold mAP: {mean_map:.4f} ± {std_map:.4f}",
                file=f,
            )
    else:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(
            f"Model {args.model} on dataset {args.dataset_name} with C = {args.inverse_reg}:\n"
            f"Logistic Regression K-Fold Accuracy: {mean_acc:.4f} ± {std_acc:.4f}"
        )
        with open(results_dir / "results_classification.txt", "a") as f:
            print(
                f"Model {args.model} on dataset {args.dataset_name} with C = {args.inverse_reg}:\n"
                f"Logistic Regression K-Fold Accuracy: {mean_acc:.4f} ± {std_acc:.4f}",
                file=f,
            )

if __name__ == "__main__":
    main()
