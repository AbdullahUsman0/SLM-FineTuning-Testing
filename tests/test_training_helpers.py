import tempfile
import unittest
from pathlib import Path

from training.train_lora import latest_checkpoint, to_prompt_completion


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

    def test_prompt_completion_rows_pass_through_without_metadata(self):
        prompt = [{"role": "user", "content": "input"}]
        completion = [{"role": "assistant", "content": "output"}]
        self.assertEqual(
            to_prompt_completion(
                {"prompt": prompt, "completion": completion, "metadata": {"task": "extract"}}
            ),
            {"prompt": prompt, "completion": completion},
        )

    def test_legacy_messages_are_still_supported(self):
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "input"},
            {"role": "assistant", "content": "output"},
        ]
        self.assertEqual(
            to_prompt_completion({"messages": messages}),
            {"prompt": messages[:-1], "completion": [messages[-1]]},
        )


if __name__ == "__main__":
    unittest.main()
