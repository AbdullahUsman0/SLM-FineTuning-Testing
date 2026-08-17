import json
import unittest

from local_slm_lab.corpus_v2 import (
    build_corpus_v2,
    build_sft_examples_v2,
    validate_corpus_v2,
)


class CorpusV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus_v2()
        cls.sft = build_sft_examples_v2(cls.corpus)

    def test_v2_has_320_scenarios_and_valid_splits(self):
        validate_corpus_v2(self.corpus, self.sft)
        self.assertEqual(len(self.corpus), 320)

    def test_yes_and_roman_urdu_are_in_training_data(self):
        messages = [
            json.loads(item["messages"][1]["content"])["current_message"].lower()
            for item in self.sft
            if item["metadata"]["task"] == "extract"
            and item["metadata"]["split"] == "train"
        ]
        self.assertIn("yes", messages)
        self.assertTrue(any("forecast banana hai" in message for message in messages))

    def test_v2_uses_compact_prompt_profile(self):
        extract = next(item for item in self.sft if item["metadata"]["task"] == "extract")
        payload = json.loads(extract["messages"][1]["content"])
        self.assertLessEqual(len(payload["slot_definitions"]), 24)
        self.assertIn("selected_slot", payload)

    def test_test_scenarios_never_enter_sft(self):
        self.assertFalse(any(item["metadata"]["split"] == "test" for item in self.sft))


if __name__ == "__main__":
    unittest.main()
