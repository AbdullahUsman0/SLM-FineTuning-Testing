import unittest

from local_slm_lab.corpus import build_corpus, build_sft_examples, validate_corpus


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus()

    def test_corpus_is_valid_and_has_200_scenarios(self):
        validate_corpus(self.corpus)
        self.assertEqual(len(self.corpus), 200)

    def test_splits_are_domain_disjoint(self):
        domains = {
            split: {item["domain"] for item in self.corpus if item["split"] == split}
            for split in ("train", "validation", "test")
        }
        self.assertFalse(domains["train"] & domains["validation"])
        self.assertFalse(domains["train"] & domains["test"])
        self.assertFalse(domains["validation"] & domains["test"])

    def test_test_scenarios_never_enter_sft_examples(self):
        examples = build_sft_examples(self.corpus)
        self.assertNotIn("test", {item["metadata"]["split"] for item in examples})

    def test_every_question_has_exactly_one_question_mark(self):
        questions = [item["ideal_question"] for item in self.corpus if item["ideal_question"]]
        self.assertTrue(questions)
        self.assertTrue(all(question.count("?") == 1 for question in questions))

