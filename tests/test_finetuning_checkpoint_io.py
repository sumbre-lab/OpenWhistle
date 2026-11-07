"""Check that interrupted checkpoint writes preserve the last valid model."""
import errno
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/finetuning/src"))
import checkpoint_io  # noqa: E402


class CheckpointIOTests(unittest.TestCase):
    def test_successfully_replaces_and_reloads_the_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            path.write_bytes(b"previous checkpoint")
            checkpoint_io.save_checkpoint({"epoch": 10, "weight": torch.arange(4)}, path)
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["epoch"], 10)
            torch.testing.assert_close(saved["weight"], torch.arange(4))
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_disk_full_preserves_old_checkpoint_and_removes_incomplete_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            checkpoint_io.save_checkpoint({"epoch": 5}, path)
            previous = path.read_bytes()

            def disk_full(state, handle):
                handle.write(b"partial checkpoint")
                raise OSError(errno.ENOSPC, "No space left on device")

            with patch.object(checkpoint_io.torch, "save", side_effect=disk_full):
                with self.assertRaises(OSError) as error:
                    checkpoint_io.save_checkpoint({"epoch": 6}, path)
            self.assertEqual(error.exception.errno, errno.ENOSPC)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_rename_does_not_publish_or_leave_a_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            with patch.object(checkpoint_io.os, "replace", side_effect=OSError("write failure")):
                with self.assertRaises(OSError):
                    checkpoint_io.save_checkpoint({"epoch": 10}, path)
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
