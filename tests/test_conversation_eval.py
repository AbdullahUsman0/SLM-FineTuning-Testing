import unittest

from local_slm_lab.conversation_eval import score_conversations


class ConversationMetricTests(unittest.TestCase):
    def test_progress_and_repeat_rates_are_behavioral(self):
        records = [
            {
                "success": True,
                "intent_correct": True,
                "turns": [
                    {
                        "state_progressed": True,
                        "repeated_previous_question": False,
                        "traces": [{"operation": "extract", "parsed": True}],
                    },
                    {
                        "state_progressed": True,
                        "repeated_previous_question": False,
                        "traces": [{"operation": "ask", "parsed": True}],
                    },
                ],
            }
        ]
        metrics = score_conversations(records)
        self.assertEqual(metrics["state_progression_rate"], 1.0)
        self.assertEqual(metrics["consecutive_question_repeat_rate"], 0.0)
        self.assertEqual(metrics["structured_output_validity_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
