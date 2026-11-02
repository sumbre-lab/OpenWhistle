"""Verify the benchmark launcher uses the published evaluation settings."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkLauncherTests(unittest.TestCase):
    def test_protocol_overrides_are_rejected_without_creating_a_run(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-run"
            for option, value in (
                ("--batch_size", "64"),
                ("--seed", "0"),
                ("--wav2vec2_model_id", "test/alternative-model"),
            ):
                with self.subTest(option=option):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "experiments/benchmark/src/run_benchmark.py"),
                            "--output_dir",
                            str(output),
                            option,
                            value,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("unrecognized arguments", result.stderr)
                    self.assertFalse(output.exists())

    def test_main_launcher_uses_published_protocol_from_any_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "experiments/benchmark/src/run_benchmark.py"),
                    "--dry_run",
                    "--output_dir",
                    str(output),
                ],
                cwd=folder,
                capture_output=True,
                text=True,
                check=True,
            )
            commands = result.stdout.splitlines()
            self.assertEqual(len(commands), 28)
            self.assertFalse(output.exists())
            for command in commands[:-1]:
                self.assertIn("--embedding_batch_size 1", command)
                self.assertIn("--seed 42 --num_seeds 1 --num_bootstrap 1000", command)
            self.assertEqual(
                sum(
                    "--hf_model_id dolphinteam/OpenWhistle-Wav2Vec2.0" in c
                    for c in commands
                ),
                3,
            )


if __name__ == "__main__":
    unittest.main()
