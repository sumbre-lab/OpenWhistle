import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from datasets import CsvAudioDataset
from metrics import Accuracy, MeanAveragePrecision
from models import (
    AvesClassifier,
    BiolingualClassifier,
    Dolph2VecClassifier,
    ResNetClassifier,
    VGGishClassifier,
)


Task = Literal["classification", "detection"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def spec2feats(spec: torch.Tensor) -> np.ndarray:
    # spec: (F, T)
    spec = torch.cat([spec.mean(dim=1), spec.std(dim=1), spec.min(dim=1)[0], spec.max(dim=1)[0]])
    return spec.numpy().reshape(-1)


@dataclass(frozen=True)
class LoadedData:
    df: pd.DataFrame
    task: Task
    # classification
    label_to_id: Optional[Dict[str, int]] = None
    num_labels: Optional[int] = None
    # detection
    detection_label_cols: Optional[List[str]] = None


def _infer_detection_label_cols(df: pd.DataFrame) -> List[str]:
    # common metadata columns
    meta_cols = {"path", "name", "original_path", "start", "end", "duration"}
    cols = [c for c in df.columns if c not in meta_cols]
    if not cols:
        raise ValueError("Could not infer detection label columns. Expected label columns besides metadata.")
    return cols


def load_csv(csv_path: str, task: Task) -> LoadedData:
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError(f"CSV must contain a 'path' column. Got columns: {list(df.columns)}")

    df["path"] = df["path"].str.replace(
        "/lustre/fsn1/projects/rech/vzf/uqe97pu/raw_data/all_categories/",
        "/media/DOLPHIN/HF_DolphinReef-labeled/",
        regex=False,
    )
    df["path"] = df["path"].str.replace(
        "data/detection/OW_detection/",
        "data/detection/",
        regex=False,
    )

    if task == "classification":
        if "label" not in df.columns:
            raise ValueError("Classification task expects a 'label' column.")
        labels = sorted(df["label"].astype(str).unique().tolist())
        label_to_id = {lbl: i for i, lbl in enumerate(labels)}
        return LoadedData(df=df, task=task, label_to_id=label_to_id, num_labels=len(labels))

    detection_label_cols = _infer_detection_label_cols(df)
    return LoadedData(df=df, task=task, detection_label_cols=detection_label_cols, num_labels=len(detection_label_cols))


def _metric_factory(task: Task):
    return Accuracy if task == "classification" else MeanAveragePrecision


def _feature_type_for_model(model_type: str) -> Literal["mfcc", "melspectrogram", "vggish", "waveform"]:
    if model_type == "vggish":
        return "vggish"
    if model_type.startswith("resnet"):
        return "melspectrogram"
    if model_type in {"biolingual", "aves", "dolph2vec"}:
        return "waveform"
    return "mfcc"


def _build_sklearn_model(model_type: str, seed: int):
    if model_type == "lr":
        return LogisticRegression(max_iter=10_000, random_state=seed)
    if model_type == "svm":
        return SVC(probability=False, random_state=seed)
    if model_type == "decisiontree":
        return DecisionTreeClassifier(random_state=seed)
    if model_type == "gbdt":
        return GradientBoostingClassifier(random_state=seed)
    if model_type == "xgboost":
        return XGBClassifier(n_jobs=4, random_state=seed)
    raise ValueError(f"Unknown sklearn model_type: {model_type}")


def _eval_sklearn(
    model,
    scaler: preprocessing.StandardScaler,
    dataloader: DataLoader,
    num_labels: int,
    task: Task,
):
    Metric = _metric_factory(task)
    metric = Metric()
    for x, y in dataloader:
        xs = [spec2feats(x[i]) for i in range(x.shape[0])]
        xs_scaled = scaler.transform(xs)
        if task == "classification":
            pred = model.predict(xs_scaled)
            pred_oh = F.one_hot(torch.tensor(pred), num_classes=num_labels)
            y_t = y if torch.is_tensor(y) else torch.tensor(y)
            metric.update(pred_oh, y_t)
        else:
            # MultiOutputClassifier.predict returns (N, L) hard labels; metric expects logits-like scores but
            # in our Metrics implementation, MeanAveragePrecision accepts probabilities too. We use predict_proba if possible.
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(xs_scaled)
                # MultiOutputClassifier returns list[L] of (N, 2); we take positive class
                if isinstance(model, MultiOutputClassifier):
                    probs = np.array([np.asarray(p)[:, 1] for p in probs]).T
                metric.update(probs, y.numpy() if hasattr(y, "numpy") else np.asarray(y))
            else:
                pred = model.predict(xs_scaled)
                metric.update(pred, y.numpy() if hasattr(y, "numpy") else np.asarray(y))
    return metric.get_primary_metric()


def _train_eval_sklearn_fold(
    model_type: str,
    task: Task,
    dataloader_train: DataLoader,
    dataloader_valid: DataLoader,
    num_labels: int,
    seed: int,
) -> float:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for x, y in dataloader_train:
        xs.extend(spec2feats(x[i]) for i in range(x.shape[0]))
        if task == "classification":
            ys.extend(np.asarray(y))
        else:
            ys.extend(y.numpy())

    scaler = preprocessing.StandardScaler().fit(xs)
    xs_scaled = scaler.transform(xs)

    model = _build_sklearn_model(model_type, seed=seed)
    if task == "detection":
        model = MultiOutputClassifier(model)
    model.fit(xs_scaled, ys)

    return _eval_sklearn(model=model, scaler=scaler, dataloader=dataloader_valid, num_labels=num_labels, task=task)


def _eval_pytorch_model(model, dataloader, metric_factory, device, desc: str, task: Task) -> float:
    model.eval()
    metric = metric_factory()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc, leave=False):
            x = x.to(device)
            y = y.to(device)
            _, logits = model(x, y)
            if task == "detection":
                metric.update(torch.sigmoid(logits), y)
            else:
                metric.update(logits, y)
    return metric.get_primary_metric()


def _train_eval_pytorch_fold(
    model_type: str,
    task: Task,
    dataloader_train: DataLoader,
    dataloader_valid: DataLoader,
    num_labels: int,
    sample_rate: int,
    device: torch.device,
    epochs: int,
    lr: float,
    classifier_type: str,
    freeze_feature_encoder: bool,
    fold_desc: str,
) -> float:
    Metric = _metric_factory(task)

    if model_type.startswith("resnet"):
        pretrained = model_type.endswith("pretrained")
        model = ResNetClassifier(
            model_type=model_type, pretrained=pretrained, num_classes=num_labels, multi_label=(task == "detection")
        ).to(device)
    elif model_type == "vggish":
        model = VGGishClassifier(sample_rate=sample_rate, num_classes=num_labels, multi_label=(task == "detection")).to(
            device
        )
    elif model_type == "biolingual":
        model = BiolingualClassifier(
            sample_rate=sample_rate,
            num_classes=num_labels,
            classifier_type=classifier_type,
            freeze_feature_encoder=freeze_feature_encoder,
            multi_label=(task == "detection"),
        ).to(device)
    elif model_type == "aves":
        model = AvesClassifier(
            sample_rate=sample_rate,
            num_classes=num_labels,
            classifier_type=classifier_type,
            freeze_feature_encoder=freeze_feature_encoder,
            multi_label=(task == "detection"),
        ).to(device)
    elif model_type == "dolph2vec":
        model = Dolph2VecClassifier(
            sample_rate=sample_rate,
            num_classes=num_labels,
            classifier_type=classifier_type,
            freeze_feature_encoder=freeze_feature_encoder,
            multi_label=(task == "detection"),
        ).to(device)
    else:
        raise ValueError(f"Unknown pytorch model_type: {model_type}")

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    metric_factory = Metric

    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(
            dataloader_train,
            desc=f"{fold_desc} train {epoch + 1}/{epochs}",
            leave=False,
        )
        for x, y in train_bar:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            loss, _ = model(x, y)
            loss.backward()
            optimizer.step()

    return _eval_pytorch_model(
        model=model,
        dataloader=dataloader_valid,
        metric_factory=metric_factory,
        device=device,
        desc=f"{fold_desc} valid",
        task=task,
    )


def _iter_model_types(requested: str) -> List[str]:
    all_types = [
        "lr",
        "svm",
        "decisiontree",
        "gbdt",
        "xgboost",
        "resnet18",
        "resnet18-pretrained",
        "resnet50",
        "resnet50-pretrained",
        "resnet152",
        "resnet152-pretrained",
        "vggish",
        "biolingual",
        "aves",
        "dolph2vec",
    ]
    if requested == "all":
        return all_types
    return [x.strip() for x in requested.split(",") if x.strip()]


TASK_TO_CSV = {
    "classification": "data/classification/balanced/all.csv",
    "detection": "data/detection/all.csv",
}

CLASSIFICATION_RESULTS_PATH = "logs/classification.txt"
DETECTION_RESULTS_PATH = "logs/detection.txt"


def _append_text_file(path: str, line: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "detection"], required=True)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="If omitted: 3s for classification, 0.5s for detection.",
    )
    parser.add_argument("--model-types", default="all", help="Comma-separated list or 'all'.")
    # pytorch
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--classifier-type", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--freeze-feature-encoder", action="store_true", default=False)
    parser.add_argument("--log-path", type=str)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    log_f = open(args.log_path, "w") if args.log_path else sys.stdout

    csv_path = TASK_TO_CSV[args.task]
    loaded = load_csv(csv_path, task=args.task)
    df = loaded.df
    num_labels = int(loaded.num_labels or 0)

    model_types = _iter_model_types(args.model_types)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare folds
    if args.task == "classification":
        y_all = df["label"].map(lambda x: loaded.label_to_id[str(x)]).to_numpy()
        splitter = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.zeros(len(df)), y_all))
    else:
        y_all = df[loaded.detection_label_cols].values.astype(int)
        splitter = MultilabelStratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.zeros(len(df)), y_all))

    for model_type in model_types:
        feature_type = _feature_type_for_model(model_type)
        sample_rate = 48_000 if model_type == "biolingual" else args.sample_rate
        max_duration = args.max_duration
        if max_duration is None:
            max_duration = 3.0 if args.task == "classification" else 0.5

        is_sklearn = model_type in {"lr", "svm", "decisiontree", "gbdt", "xgboost"}
        fold_scores: List[float] = []
        fold_iter = enumerate(splits)
        if not is_sklearn:
            fold_iter = tqdm(
                fold_iter,
                total=len(splits),
                desc=f"{model_type} k-fold",
                leave=True,
            )

        for fold_idx, (train_idx, valid_idx) in fold_iter:
            df_train = df.iloc[train_idx]
            df_valid = df.iloc[valid_idx]

            ds_train = CsvAudioDataset(
                df=df_train,
                task=args.task,
                feature_type=feature_type,
                sample_rate=sample_rate,
                max_duration=max_duration,
                label_to_id=loaded.label_to_id,
                detection_label_cols=loaded.detection_label_cols,
            )
            ds_valid = CsvAudioDataset(
                df=df_valid,
                task=args.task,
                feature_type=feature_type,
                sample_rate=sample_rate,
                max_duration=max_duration,
                label_to_id=loaded.label_to_id,
                detection_label_cols=loaded.detection_label_cols,
            )

            dl_train = DataLoader(
                dataset=ds_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            dl_valid = DataLoader(
                dataset=ds_valid,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            if is_sklearn:
                score = _train_eval_sklearn_fold(
                    model_type=model_type,
                    task=args.task,
                    dataloader_train=dl_train,
                    dataloader_valid=dl_valid,
                    num_labels=num_labels,
                    seed=args.seed + fold_idx,
                )
            else:
                score = _train_eval_pytorch_fold(
                    model_type=model_type,
                    task=args.task,
                    dataloader_train=dl_train,
                    dataloader_valid=dl_valid,
                    num_labels=num_labels,
                    sample_rate=sample_rate,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    classifier_type=args.classifier_type,
                    freeze_feature_encoder=args.freeze_feature_encoder,
                    fold_desc=f"{model_type} fold {fold_idx + 1}/{args.kfold}",
                )

            fold_scores.append(float(score))
            msg = f"{model_type} fold {fold_idx + 1}/{args.kfold}: metric={score:.4f}"
            if not is_sklearn and log_f is sys.stdout:
                tqdm.write(msg)
            else:
                print(msg, file=log_f)
            log_f.flush()

        if fold_scores:
            mean = float(np.mean(fold_scores))
            std = float(np.std(fold_scores))
            metric_name = "acc" if args.task == "classification" else "mAP"
            summary = f"{model_type}: {metric_name} {mean:.4f} ± {std:.4f} (k={args.kfold})"
            print(summary, file=log_f)
            if args.task == "classification":
                _append_text_file(CLASSIFICATION_RESULTS_PATH, summary)
            elif args.task == "detection":
                _append_text_file(DETECTION_RESULTS_PATH, summary)
            log_f.flush()

    if args.log_path:
        log_f.close()


if __name__ == "__main__":
    main()
