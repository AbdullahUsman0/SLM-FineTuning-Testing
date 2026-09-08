import json
import unittest

from local_slm_lab.corpus_v4 import (
    build_corpus_v4,
    build_sft_examples_v4,
    validate_corpus_v4,
)


class CorpusV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus_v4()
        cls.sft = build_sft_examples_v4(cls.corpus)

    def test_v4_is_extraction_only_prompt_completion(self):
        validate_corpus_v4(self.corpus, self.sft)
        self.assertTrue(all(item["metadata"]["task"] == "extract" for item in self.sft))
        self.assertTrue(all("prompt" in item and "completion" in item for item in self.sft))
        self.assertTrue(all("messages" not in item for item in self.sft))

    def test_v4_augmentation_variants_are_present(self):
        variants = {item["metadata"]["variant"] for item in self.sft}
        expected = {
            "reviewed_base",
            "paraphrase_1",
            "paraphrase_2",
            "paraphrase_3",
            "selected_short",
            "selected_sentence",
            "selected_roman_urdu",
            "selected_multi_slot",
            "selected_non_answer",
            "cluster_answer",
        }
        self.assertTrue(expected.issubset(variants), f"missing variants: {expected - variants}")

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
