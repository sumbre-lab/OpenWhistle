"""Load raw Hub audio for the shared SoundFile waveform decoder.

Keep decoding off here: newer Datasets versions otherwise decode during row
iteration using TorchCodec before the benchmark can read the embedded bytes.
"""

from __future__ import annotations

from benchmark_revisions import DATASET_REVISIONS
from datasets import Audio, ClassLabel, Sequence, load_dataset

CLASSIFICATION_DATASET_ID = "dolphinteam/OpenWhistle-Classification-Finetuning"
CLASSIFICATION_BALANCED_CONFIG_NAME = "balanced"
DETECTION_DATASET_ID = "dolphinteam/OpenWhistle-Detection-Finetuning"

# ClassLabel order in the pinned classification dataset snapshot.
_UNBALANCED_CLASS_NAMES = (
    "NSW_3",
    "NSW_2",
    "NSW_1",
    "SW_Dana",
    "SW_Luna",
    "SW_Nana",
    "SW_Neo",
    "SW_Nikita",
    "SW_Shy",
    "SW_Yosefa",
)
CLASSIFICATION_LABEL_NAMES = {
    "unbalanced": _UNBALANCED_CLASS_NAMES,
    "balanced": ("NSW_1", "SW_Luna", "SW_Nana", "SW_Neo", "SW_Nikita", "SW_Yosefa"),
}

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


def load_classification_splits(
    config_name: str = CLASSIFICATION_BALANCED_CONFIG_NAME,
):
    if config_name not in CLASSIFICATION_LABEL_NAMES:
        raise ValueError(
            "Classification supports only unbalanced and balanced configurations."
        )
    ds = load_dataset(
        CLASSIFICATION_DATASET_ID,
        name=config_name,
        revision=DATASET_REVISIONS[CLASSIFICATION_DATASET_ID],
    )
    first_split = next(iter(ds.values()))
    label_feature = first_split.features["label"]
    if not isinstance(label_feature, ClassLabel):
        raise TypeError("Classification dataset label must be a ClassLabel.")
    _require_splits(ds, CLASSIFICATION_DATASET_ID, ("train", "validation", "test"))
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds, label_feature


def get_detection_label_vector(row: dict) -> list[int]:
    if "labels" in row:
        return [int(value) for value in row["labels"]]
    if "label" in row and not isinstance(row["label"], (int, str)):
        return [int(value) for value in row["label"]]
    return [int(row[column]) for column in DETECTION_ONE_HOT_COLUMNS]


def load_detection_splits():
    ds = load_dataset(
        DETECTION_DATASET_ID,
        name="default",
        data_dir="data",
        revision=DATASET_REVISIONS[DETECTION_DATASET_ID],
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
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds, DETECTION_ONE_HOT_COLUMNS
