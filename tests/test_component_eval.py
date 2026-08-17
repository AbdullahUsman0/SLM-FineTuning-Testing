import unittest

from local_slm_lab.component_eval import CallRecord, score_records


def extraction(intent="create_forecast", updates=None, correction=False):
    return {
        "intent": intent,
        "intent_confidence": 1.0,
        "updates": updates or [],
        "correction_detected": correction,
        "unsupported_claims": [],
    }


class ComponentMetricTests(unittest.TestCase):
    def test_perfect_records_score_one(self):
        value = extraction(
            updates=[
                {
                    "slot_id": "target_column",
                    "candidate_value": "sales",
                    "status": "provided",
                    "confidence": 1.0,
                    "evidence_text": "sales",
                }
            ]
        )
        records = [
            CallRecord(
                scenario_id="one",
                category="complete",
                task="extract",
                turn_index=1,
                expected=value,
                predicted=value,
                latency_ms=10,
                error=None,
                checks={
                    "intent": True,
                    "joint_extraction": True,
                    "correction": True,
                    "has_forbidden_contract": False,
                    "avoids_forbidden_slots": True,
                },
            ),
            CallRecord(
                scenario_id="one",
                category="complete",
                task="ask",
                turn_index=None,
                expected={"question": "What is the target?"},
                predicted={"question": "What is the target?"},
                latency_ms=20,
                error=None,
                checks={
                    "question_contract": True,
                    "one_question": True,
                    "slot_relevance": True,
                },
            ),
        ]
        metrics = score_records(records)
        self.assertEqual(metrics["intent_accuracy"], 1.0)
        self.assertEqual(metrics["slot_micro_f1"], 1.0)
        self.assertEqual(metrics["question_contract_accuracy"], 1.0)

    def test_provider_failure_lowers_success_rate(self):
        record = CallRecord(
            scenario_id="one",
            category="complete",
            task="extract",
            turn_index=1,
            expected=extraction(),
            predicted=None,
            latency_ms=5,
            error="failed",
            checks={},
        )
        self.assertEqual(score_records([record])["provider_success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
