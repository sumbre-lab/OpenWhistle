from __future__ import annotations

from datasets import Audio, ClassLabel, concatenate_datasets, load_dataset


CLASSIFICATION_DATASET_ID = "dolphinteam/OpenWhistle-1.0-Classification-Finetuning"
CLASSIFICATION_BALANCED_CONFIG_NAME = "default"
DETECTION_DATASET_ID = "dolphinteam/OpenWhistle-1.0-Detection-Finetuning"
DETECTION_LABEL_VECTOR_COLUMN = "labels"

DETECTION_ONE_HOT_COLUMNS = (
    "SW_Neo",
    "SW_Luna",
    "SW_Nikita",
    "SW_Nana",
    "SW_Yosefa",
    "SW_Dana",
    "NSW_1",
)
DETECTION_SOURCE_LABEL_TO_ONE_HOT = {
    "Neo": "SW_Neo",
    "Luna": "SW_Luna",
    "Nikita": "SW_Nikita",
    "Nana": "SW_Nana",
    "Yosefa": "SW_Yosefa",
    "Dana": "SW_Dana",
    "NSW_1": "NSW_1",
    "noise": None,
}
DETECTION_ONE_HOT_TO_SOURCE_LABEL = {
    column: source_label
    for source_label, column in DETECTION_SOURCE_LABEL_TO_ONE_HOT.items()
    if column is not None
}


def _load_concat_splits(dataset_id: str, config_name: str | None = None):
    ds = load_dataset(dataset_id, name=config_name)
    splits = [split for split in ("train", "test") if split in ds]
    if not splits:
        raise ValueError(f"No train/test splits found for dataset {dataset_id}.")
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
    splits = [split for split in ("train", "test") if split in ds]
    if not splits:
        raise ValueError(
            f"No train/test splits found for dataset {CLASSIFICATION_DATASET_ID}."
        )
    ds = concatenate_datasets([ds[split] for split in splits]).cast_column(
        "audio", Audio(decode=True)
    )
    return ds, ds.features["label"]


def _source_label_to_vector(source_label: str) -> list[int]:
    if source_label not in DETECTION_SOURCE_LABEL_TO_ONE_HOT:
        raise ValueError(
            f"Unexpected detection source_label {source_label!r}; "
            "cannot reconstruct the label vector."
        )
    active_column = DETECTION_SOURCE_LABEL_TO_ONE_HOT[source_label]
    return [1 if column == active_column else 0 for column in DETECTION_ONE_HOT_COLUMNS]


def get_detection_label_vector(row: dict) -> list[int]:
    if DETECTION_LABEL_VECTOR_COLUMN in row:
        return [int(value) for value in row[DETECTION_LABEL_VECTOR_COLUMN]]
    return [int(row[column]) for column in DETECTION_ONE_HOT_COLUMNS]


def get_detection_binary_label(row: dict) -> int:
    if DETECTION_LABEL_VECTOR_COLUMN in row:
        return int(any(int(value) for value in row[DETECTION_LABEL_VECTOR_COLUMN]))
    return int(any(int(row[column]) for column in DETECTION_ONE_HOT_COLUMNS))


def _standardize_detection_example(example: dict) -> dict:
    standardized_example = dict(example)

    if DETECTION_LABEL_VECTOR_COLUMN in standardized_example:
        standardized_example[DETECTION_LABEL_VECTOR_COLUMN] = [
            int(value) for value in standardized_example[DETECTION_LABEL_VECTOR_COLUMN]
        ]
        return standardized_example

    has_one_hot_columns = all(column in example for column in DETECTION_ONE_HOT_COLUMNS)
    if has_one_hot_columns:
        active_columns = [
            column for column in DETECTION_ONE_HOT_COLUMNS if int(example[column]) == 1
        ]
        if len(active_columns) > 1:
            raise ValueError(
                "Detection example has multiple active source columns; expected at most one."
            )
        source_label = (
            DETECTION_ONE_HOT_TO_SOURCE_LABEL[active_columns[0]]
            if active_columns
            else standardized_example.get("source_label", "noise")
        )
        return {
            key: value
            for key, value in standardized_example.items()
            if key not in DETECTION_ONE_HOT_COLUMNS
        } | {
            "source_label": source_label,
            DETECTION_LABEL_VECTOR_COLUMN: [
                int(example[column]) for column in DETECTION_ONE_HOT_COLUMNS
            ],
        }

    if "source_label" in standardized_example:
        return {
            **standardized_example,
            DETECTION_LABEL_VECTOR_COLUMN: _source_label_to_vector(
                standardized_example["source_label"]
            ),
        }

    raise ValueError(
        "Detection example must expose source_label, labels, or the original detection CSV one-hot columns."
    )


def load_detection_examples():
    ds = _load_concat_splits(DETECTION_DATASET_ID, config_name="default")
    label_feature = ds.features.get("label")
    if label_feature is not None and not isinstance(label_feature, ClassLabel):
        raise TypeError("Detection dataset label must be a ClassLabel when present.")

    missing_one_hot_columns = [
        column for column in DETECTION_ONE_HOT_COLUMNS if column not in ds.column_names
    ]
    if (
        "source_label" not in ds.column_names
        and DETECTION_LABEL_VECTOR_COLUMN not in ds.column_names
        and missing_one_hot_columns
    ):
        raise ValueError(
            "Detection dataset is expected to match the original detection CSV schema "
            f"{DETECTION_ONE_HOT_COLUMNS}, or expose source_label/labels. "
            f"Missing columns: {missing_one_hot_columns}"
        )

    ds = ds.map(_standardize_detection_example)
    removable_columns = [
        column for column in ("label", *DETECTION_ONE_HOT_COLUMNS) if column in ds.column_names
    ]
    if removable_columns:
        ds = ds.remove_columns(removable_columns)
    return ds, DETECTION_ONE_HOT_COLUMNS