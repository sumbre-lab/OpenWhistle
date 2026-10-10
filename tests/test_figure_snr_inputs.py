"""Manuscript panel F must use completed, matching full-corpus SNR inputs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from datasets_figures.scripts.classification_overview import load_snr_comparison


class FigureSNRInputTests(unittest.TestCase):
    def prepare(self, folder):
        protocol = {"limit_per_dataset": 0, "datasets": {}}
        groups = {
            "pretraining": ("dolphinteam/OpenWhistle-Pretraining", "default", ["train", "validation"]),
            "classification_all": ("dolphinteam/OpenWhistle-Classification-Finetuning", "all",
                                   ["train", "validation", "test"]),
        }
        for group, (repo, config, splits) in groups.items():
            path = folder / f"snr_{group}.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["split", "index", "snr_db", "status"])
                writer.writeheader()
                for i, split in enumerate(splits):
                    writer.writerow(dict(split=split, index=0, snr_db=i + 2, status="ok"))
                writer.writerow(dict(split="train", index=1, snr_db="nan", status="too_short"))
            protocol["datasets"][group] = dict(repo=repo, config=config, splits=splits,
                                              revision="fixed-commit", processed=len(splits) + 1,
                                              status_counts={"ok": len(splits), "too_short": 1})
        (folder / "snr_protocol.json").write_text(json.dumps(protocol))
        return protocol

    def test_full_inputs_exclude_invalid_clip_and_preserve_group_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.prepare(folder)
            pretraining, classification, protocol = load_snr_comparison(folder)
            self.assertEqual(pretraining.tolist(), [2, 3])
            self.assertEqual(classification.tolist(), [2, 3, 4])

    def test_diagnostic_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            protocol = self.prepare(folder)
            protocol["limit_per_dataset"] = 3
            (folder / "snr_protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "diagnostic subset"):
                load_snr_comparison(folder)

    def test_truncated_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.prepare(folder)
            path = folder / "snr_pretraining.csv"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "completed protocol"):
                load_snr_comparison(folder)

    def test_duplicate_row_cannot_replace_a_missing_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.prepare(folder)
            path = folder / "snr_classification_all.csv"
            text = path.read_text().replace("train,1,nan,too_short", "train,0,nan,too_short")
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "duplicate SNR row"):
                load_snr_comparison(folder)


if __name__ == "__main__":
    unittest.main()
