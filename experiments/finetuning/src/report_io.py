"""Write portable evaluation reports without local paths or unused arguments."""

import json


def clean_report(report):
    """Keep scientific results intact and remove unused reporting metadata."""
    def without_null_fields(value):
        if isinstance(value, dict):
            return {key: without_null_fields(child) for key, child in value.items()
                    if child is not None}
        if isinstance(value, list):
            return [without_null_fields(child) for child in value]
        return value

    cleaned = without_null_fields(report)
    arguments = cleaned.get("training_args", {})
    # Model identity is recorded at the top level. Native AVES ignores the
    # parser's default BioLingual model_id, so copying it would be misleading.
    for key in ("output_dir", "model_id", "num_bootstrap"):
        arguments.pop(key, None)
    if arguments.get("class_weight") == "none":
        arguments.pop("class_weight")
    return cleaned


def write_report(path, report):
    path.write_text(json.dumps(clean_report(report), indent=2, allow_nan=False) + "\n")
