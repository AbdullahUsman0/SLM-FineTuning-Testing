import unittest

from local_slm_lab.conversation_eval import score_conversations


class ConversationMetricTests(unittest.TestCase):
    def test_progress_and_repeat_rates_are_behavioral(self):
        records = [
            {
                "success": True,
                "intent_correct": True,
                "slots_exact": True,
                "assistant_safe": True,
                "expected_final_slots": {"intent": "create_forecast"},
                "final_slots": {"intent": "create_forecast"},
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
        self.assertEqual(metrics["structured_model_calls"], 2)
        self.assertEqual(metrics["exact_final_state_accuracy"], 1.0)
        self.assertEqual(metrics["slot_micro_f1"], 1.0)
        self.assertEqual(metrics["unexpected_slot_inference_rate"], 0.0)

    def test_unexpected_slot_is_not_counted_as_success(self):
        records = [
            {
                "success": False,
                "intent_correct": True,
                "slots_exact": False,
                "assistant_safe": True,
                "expected_final_slots": {"intent": "create_forecast"},
                "final_slots": {
                    "intent": "create_forecast",
                    "target_column": "invented",
                },
                "turns": [
                    {
                        "state_progressed": True,
                        "repeated_previous_question": False,
                        "traces": [{"operation": "extract", "parsed": True}],
                    }
                ],
            }
        ]
        metrics = score_conversations(records)
        self.assertEqual(metrics["case_success_rate"], 0.0)
        self.assertEqual(metrics["exact_final_state_accuracy"], 0.0)
        self.assertEqual(metrics["slot_micro_recall"], 1.0)
        self.assertEqual(metrics["slot_micro_precision"], 0.5)
        self.assertEqual(metrics["unexpected_slot_inference_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
