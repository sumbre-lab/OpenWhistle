from __future__ import annotations

from datasets import Audio, ClassLabel, Sequence, concatenate_datasets, load_dataset


CLASSIFICATION_DATASET_ID = "dolphinteam/OpenWhistle-Classification-Finetuning"
CLASSIFICATION_BALANCED_CONFIG_NAME = "balanced"
DETECTION_DATASET_ID = "dolphinteam/OpenWhistle-Detection-Finetuning"

DETECTION_ONE_HOT_COLUMNS = (
    "SW_Neo",
    "SW_Luna",
    "SW_Nikita",
    "SW_Nana",
    "SW_Yosefa",
    "SW_Dana",
    "NSW_1",
)


def _require_splits(ds, dataset_id: str, required_splits: tuple[str, ...]) -> list[str]:
    missing_splits = [split for split in required_splits if split not in ds]
    if missing_splits:
        raise ValueError(
            f"Dataset {dataset_id} is missing required splits: {missing_splits}."
        )
    return list(required_splits)


def _require_train_test_splits(ds, dataset_id: str) -> list[str]:
    splits = [split for split in ("train", "test") if split in ds]
    if not splits:
        raise ValueError(f"No train/test splits found for dataset {dataset_id}.")
    return splits


def _load_concat_splits(
    dataset_id: str,
    config_name: str | None = None,
    data_dir: str | None = None,
):
    ds = load_dataset(dataset_id, name=config_name, data_dir=data_dir)
    splits = _require_train_test_splits(ds, dataset_id)
    combined = concatenate_datasets([ds[split] for split in splits])
    return combined.cast_column("audio", Audio(decode=True))


def load_classification_examples(
    config_name: str = CLASSIFICATION_BALANCED_CONFIG_NAME,
):
    ds = load_dataset(CLASSIFICATION_DATASET_ID, name=config_name)
    first_split = next(iter(ds.values()))
    label_feature = first_split.features["label"]
    if not isinstance(label_feature, ClassLabel):
        raise TypeError("Classification dataset label must be a ClassLabel.")
    splits = _require_train_test_splits(ds, CLASSIFICATION_DATASET_ID)
    ds = concatenate_datasets([ds[split] for split in splits]).cast_column(
        "audio", Audio(decode=True)
    )
    return ds, ds.features["label"]


def load_classification_splits(
    config_name: str = CLASSIFICATION_BALANCED_CONFIG_NAME,
):
    ds = load_dataset(CLASSIFICATION_DATASET_ID, name=config_name)
    first_split = next(iter(ds.values()))
    label_feature = first_split.features["label"]
    if not isinstance(label_feature, ClassLabel):
        raise TypeError("Classification dataset label must be a ClassLabel.")
    _require_splits(ds, CLASSIFICATION_DATASET_ID, ("train", "validation", "test"))
    ds = ds.cast_column("audio", Audio(decode=True))
    return ds, label_feature


def get_detection_label_vector(row: dict) -> list[int]:
    if "labels" in row:
        return [int(value) for value in row["labels"]]
    if "label" in row and not isinstance(row["label"], (int, str)):
        return [int(value) for value in row["label"]]
    return [int(row[column]) for column in DETECTION_ONE_HOT_COLUMNS]


def get_detection_binary_label(row: dict) -> int:
    if "labels" in row:
        return int(any(get_detection_label_vector(row)))
    return int(any(int(row[column]) for column in DETECTION_ONE_HOT_COLUMNS))


def load_detection_examples():
    ds = _load_concat_splits(
        DETECTION_DATASET_ID,
        config_name="default",
        data_dir="data",
    )
    if "labels" not in ds.column_names and isinstance(ds.features.get("label"), Sequence):
        ds = ds.rename_column("label", "labels")

    if "labels" not in ds.column_names:
        raise ValueError(
            "Detection dataset must expose 'labels' as a fixed-length label vector."
        )

    removable_columns = [
        column
        for column in ("label", *DETECTION_ONE_HOT_COLUMNS)
        if column in ds.column_names
    ]
    if removable_columns:
        ds = ds.remove_columns(removable_columns)
    ds = ds.cast_column("audio", Audio(decode=True))
    return ds, DETECTION_ONE_HOT_COLUMNS


def load_detection_splits():
    ds = load_dataset(
        DETECTION_DATASET_ID,
        name="default",
        data_dir="data",
    )
    _require_splits(ds, DETECTION_DATASET_ID, ("train", "validation", "test"))

    first_split = next(iter(ds.values()))
    if "labels" not in first_split.column_names and isinstance(
        first_split.features.get("label"), Sequence
    ):
        ds = ds.rename_column("label", "labels")
        first_split = next(iter(ds.values()))

    if "labels" not in first_split.column_names:
        raise ValueError(
            "Detection dataset must expose 'labels' as a fixed-length label vector."
        )

    removable_columns = [
        column
        for column in ("label", *DETECTION_ONE_HOT_COLUMNS)
        if column in first_split.column_names
    ]
    if removable_columns:
        ds = ds.remove_columns(removable_columns)
    ds = ds.cast_column("audio", Audio(decode=True))
    return ds, DETECTION_ONE_HOT_COLUMNS
