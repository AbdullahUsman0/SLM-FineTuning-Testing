import unittest

from local_slm_lab.evaluate import check_answer


class EvaluationCheckTests(unittest.TestCase):
    def test_detects_forbidden_future_target_request(self):
        case = {
            "must_contain": ["bitcoin"],
            "must_not_contain": ["prices for the next 7 days"],
        }
        checks = check_answer(
            case,
            "Please provide Bitcoin prices for the next 7 days.",
        )
        self.assertFalse(checks["avoids_forbidden_text"])

    def test_enforces_one_question_and_no_numbered_list(self):
        case = {"max_question_marks": 1, "max_numbered_items": 0}
        checks = check_answer(case, "1. What target?\n2. What horizon?")
        self.assertFalse(checks["question_limit"])
        self.assertFalse(checks["numbered_item_limit"])


if __name__ == "__main__":
    unittest.main()
