import json
import unittest

from local_slm_lab.corpus_v3 import (
    build_corpus_v3,
    build_sft_examples_v3,
    validate_corpus_v3,
)


class CorpusV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus_v3()
        cls.sft = build_sft_examples_v3(cls.corpus)

    def test_v3_is_extraction_only_prompt_completion(self):
        validate_corpus_v3(self.corpus, self.sft)
        self.assertTrue(all(item["metadata"]["task"] == "extract" for item in self.sft))
        self.assertTrue(all("prompt" in item and "completion" in item for item in self.sft))
        self.assertTrue(all("messages" not in item for item in self.sft))

    def test_selected_slot_answers_and_nonanswers_are_covered(self):
        variants = {item["metadata"]["variant"] for item in self.sft}
        self.assertIn("selected_short", variants)
        self.assertIn("selected_sentence", variants)
        self.assertIn("selected_non_answer", variants)

    def test_selected_slot_context_matches_training_label(self):
        augmented = next(
            item for item in self.sft if item["metadata"]["variant"] == "selected_short"
        )
        payload = json.loads(augmented["prompt"][1]["content"])
        self.assertEqual(payload["selected_slot"], augmented["metadata"]["selected_slot"])

    def test_test_scenarios_never_enter_sft(self):
        self.assertFalse(any(item["metadata"]["split"] == "test" for item in self.sft))


if __name__ == "__main__":
    unittest.main()
