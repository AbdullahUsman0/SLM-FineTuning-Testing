import json
import unittest

from local_slm_lab.v5_prompts import build_v5_extractor_input
from local_slm_lab.v5_provider import strict_json_object, validate_raw_output
from forecasting_assistant.domain.models import ExtractorResult, QuestionOutput
from forecasting_assistant.domain.schema import create_initial_state, load_schema


class V5PromptTests(unittest.TestCase):
    def test_all_schema_slots_available_on_first_turn(self):
        schema = load_schema()
        payload = json.loads(build_v5_extractor_input("I will upload a CSV file.", create_initial_state(schema), schema))
        self.assertEqual({s["slot_id"] for s in payload["slot_definitions"]}, {s.slot_id for s in schema.slots})
        file_format = next(s for s in payload["slot_definitions"] if s["slot_id"] == "file_format")
        self.assertIn("csv", file_format["allowed_values"])

    def test_output_parser_does_not_repair_model_errors(self):
        for invalid in ('```json\n{}\n```', '{"question":"What?"', '{} trailing', '{"a":1,"a":2}', '{"a":NaN}', '[]'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                strict_json_object(invalid)

    def test_extra_keys_are_schema_failure(self):
        with self.assertRaises(ValueError):
            validate_raw_output({"question": "What?", "extra": 1}, QuestionOutput)

    def test_exact_wire_contract_and_duplicate_updates(self):
        update = {"slot_id": "file_format", "candidate_value": '"csv"', "status": "provided", "confidence": 1.0, "evidence_text": "CSV"}
        raw = {"intent": "create_forecast", "intent_confidence": 1.0, "updates": [update], "correction_detected": False, "unsupported_claims": []}
        self.assertEqual(validate_raw_output(raw, ExtractorResult).updates[0].candidate_value, "csv")
        raw["updates"] = [update, update]
        with self.assertRaises(ValueError):
            validate_raw_output(raw, ExtractorResult)
        raw["updates"] = [{**update, "candidate_value": {"invalid": "wire representation"}}]
        with self.assertRaises(ValueError):
            validate_raw_output(raw, ExtractorResult)


if __name__ == "__main__":
    unittest.main()
