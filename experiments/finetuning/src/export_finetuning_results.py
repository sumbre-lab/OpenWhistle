"""Export one fine-tuning CSV and the 18 detailed evaluation reports."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from run_finetuning import ROOT, MODELS, TASKS
from report_io import write_report
from verify_finetuning_results import validate_manifest, verify


def export(run_dir, output_dir):
    source, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing releases are not overwritten.")
    manifest = json.loads((source / "run_config.json").read_text())
    validate_manifest(manifest)
    verify(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finetuning-release-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shutil.copy2(source / "results.csv", staging / "results.csv")
        for model in MODELS:
            for task in TASKS:
                relative = Path(model[0]) / task / "seed_42"
                (staging / relative).mkdir(parents=True)
                report = json.loads((source / relative / "results.json").read_text())
                write_report(staging / relative / "results.json", report)
        verify(staging)
        staging.rename(output)
    print(f"Exported verified fine-tuning release: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=ROOT / "runs/manuscript")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results/manuscript")
    args = parser.parse_args()
    export(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
