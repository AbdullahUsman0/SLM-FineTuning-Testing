import tempfile
import unittest
from pathlib import Path

from training.train_lora import latest_checkpoint


class TrainingHelperTests(unittest.TestCase):
    def test_latest_checkpoint_uses_highest_numeric_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint-17").mkdir()
            (root / "checkpoint-68").mkdir()
            (root / "checkpoint-invalid").mkdir()
            self.assertEqual(latest_checkpoint(root), root / "checkpoint-68")

    def test_latest_checkpoint_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(latest_checkpoint(Path(temporary)))


if __name__ == "__main__":
    unittest.main()
