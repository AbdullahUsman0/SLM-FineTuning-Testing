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
                    "question_exact_match": True,
                    "question_content_match": True,
                    "question_content_token_f1": 1.0,
                },
            ),
        ]
        metrics = score_records(records)
        self.assertEqual(metrics["intent_accuracy"], 1.0)
        self.assertEqual(metrics["slot_micro_f1"], 1.0)
        self.assertEqual(metrics["question_contract_accuracy"], 1.0)
        self.assertEqual(metrics["slot_id_micro_f1"], 1.0)
        self.assertEqual(metrics["nonempty_update_accuracy"], 1.0)
        self.assertEqual(metrics["empty_update_collapse_rate"], 0.0)
        self.assertEqual(metrics["question_exact_match_accuracy"], 1.0)

    def test_empty_updates_are_measured_as_collapse(self):
        expected = extraction(
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
        record = CallRecord(
            scenario_id="collapse",
            category="complete",
            task="extract",
            turn_index=1,
            expected=expected,
            predicted=extraction(),
            latency_ms=5,
            error=None,
            checks={"intent": True, "joint_extraction": False, "correction": True},
        )
        metrics = score_records([record])
        self.assertEqual(metrics["nonempty_update_accuracy"], 0.0)
        self.assertEqual(metrics["empty_update_collapse_rate"], 1.0)
        self.assertEqual(metrics["slot_id_micro_recall"], 0.0)

    def test_wrong_domain_question_fails_content_metrics(self):
        record = CallRecord(
            scenario_id="wrong-domain",
            category="missing_required",
            task="ask",
            turn_index=None,
            expected={"question": "How often is factory production output observed?"},
            predicted={"question": "How often is electricity demand observed?"},
            latency_ms=5,
            error=None,
            checks={
                "question_contract": True,
                "one_question": True,
                "slot_relevance": True,
                "question_exact_match": False,
                "question_content_match": False,
                "question_content_token_f1": 0.3333,
            },
        )
        metrics = score_records([record])
        self.assertEqual(metrics["question_contract_accuracy"], 1.0)
        self.assertEqual(metrics["question_exact_match_accuracy"], 0.0)
        self.assertEqual(metrics["question_content_accuracy"], 0.0)

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
