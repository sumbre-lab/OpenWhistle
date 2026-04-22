import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from conf import (
    aves_bio_config,
    aves_bio_model,
    aves_core_config,
    aves_core_model,
    dolph2vec_config_path,
    dolph2vec_base,
)
from models import MFCC, Aves, BioLingual, Dolph2Vec, SpectralFeatures, Spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm
from metrics import MeanAveragePrecision
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

LR_MAX_ITER = 10000


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

    parser.add_argument("--normalize_data", action="store_true", default=False)

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

    model_args = dict(
        sample_rate=args.target_sample_rate,
        dolph2vec_config_path=dolph2vec_config_path,
        dolph2vec_model_path=dolph2vec_base,
    )

    if args.model == "aves_bio":
        aves_model_path = aves_bio_model
        aves_config_path = aves_bio_config
    elif args.model == "aves_core":
        aves_model_path = aves_core_model
        aves_config_path = aves_core_config
    else:
        aves_model_path, aves_config_path = "", ""

    model_args["aves_model_path"] = aves_model_path
    model_args["aves_config_path"] = aves_config_path

    model = name2model[args.model](**model_args)

    dataset_name2path = {
        "classification": "data/classification/balanced/all.csv",
        "detection": "data/detection/all.csv",
    }

    data_path = dataset_name2path[args.dataset_name]

    df = pd.read_csv(data_path)

    # Multilabel columns only (exclude path/name and e.g. original_path)
    detection_label_cols = None
    if args.dataset_name == "detection":
        _detection_meta = {"path", "name", "original_path"}
        detection_label_cols = [c for c in df.columns if c not in _detection_meta]

    df["path"] = df["path"].str.replace(
        "/lustre/fsn1/projects/rech/vzf/uqe97pu/raw_data/all_categories/",
         #"/Volumes/jz_scratch/raw_data/all_categories/",
        #"/home/semenzin/jz_scratch/raw_data/all_categories/",
        "/media/DOLPHIN/HF_DolphinReef-labeled/",
        regex=False,
    )

    embeddings = []
    labels = []
    for i, row in tqdm(df.iterrows(), desc="processing audio files", total=len(df)):
        path = row["path"]
        if args.dataset_name == "detection":
            label = row[detection_label_cols].values.astype(int)
        else:
            label = row["label"]

        try:
            embedding = model(path)
            embeddings.append(embedding.cpu())
            labels.append(label)
        except Exception as e:
            print(f"error processing {path}: {e}")

    x_train = np.array(embeddings)
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
            y_score = np.array([np.array(x)[:, 1] for x in y_score]).T
            
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
