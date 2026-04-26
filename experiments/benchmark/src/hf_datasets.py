from __future__ import annotations

from datasets import Audio, ClassLabel, concatenate_datasets, load_dataset


CLASSIFICATION_DATASET_ID = "dolphinteam/OpenWhistle-1.0-Classification-Finetuning"
CLASSIFICATION_BALANCED_CONFIG_NAME = "default"
DETECTION_DATASET_ID = "dolphinteam/OpenWhistle-1.0-Detection-Finetuning"

DETECTION_ONE_HOT_COLUMNS = (
    "SW_Neo",
    "SW_Luna",
    "SW_Nikita",
    "SW_Nana",
    "SW_Yosefa",
    "SW_Dana",
    "NSW_1",
)


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


def get_detection_label_vector(row: dict) -> list[int]:
    if "labels" in row:
        return [int(value) for value in row["labels"]]
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
    label_feature = ds.features.get("label")
    if label_feature is not None and not isinstance(label_feature, ClassLabel):
        raise TypeError("Detection dataset label must be a ClassLabel when present.")

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
