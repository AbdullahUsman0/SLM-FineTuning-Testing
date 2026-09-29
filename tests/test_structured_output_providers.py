import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from local_slm_lab.structured_output_providers import (
    V5OpenAIStructuredProvider,
    V5RetryTransformersProvider,
    V5RuleProvider,
    V5SlotWiseTransformersProvider,
    ControlProbe,
    SlotProbe,
    _wire_result,
    openai_compatible_schema,
)
from forecasting_assistant.domain.models import ExtractorResult, Intent
from forecasting_assistant.domain.schema import create_initial_state, load_schema


def wire(updates=None):
    return json.dumps({
        "intent": "create_forecast", "intent_confidence": 1.0,
        "updates": updates or [], "correction_detected": False, "unsupported_claims": [],
    })


class RetryProviderTests(unittest.TestCase):
    def provider(self, outputs):
        provider = object.__new__(V5RetryTransformersProvider)
        provider._traces = []
        iterator = iter(outputs)
        provider._generate_raw = lambda system, user, logits_processor=None: next(iterator)
        return provider

    def test_first_attempt_is_unchanged_and_success_does_not_retry(self):
        provider = self.provider([(wire(), 10, 5, False)])
        result = provider._structured("extract", "system", "original-user", ExtractorResult)
        self.assertEqual(result.intent, Intent.CREATE_FORECAST)
        trace = provider.drain_traces()[0]
        self.assertEqual(trace["model_calls"], 1)
        self.assertEqual(trace["retry_count"], 0)
        self.assertTrue(trace["first_pass_schema_valid"])

    def test_one_retry_records_both_attempts_and_total_tokens(self):
        prompts = []
        outputs = iter([("not-json", 10, 2, False), (wire(), 30, 5, False)])
        provider = object.__new__(V5RetryTransformersProvider)
        provider._traces = []
        def generate(system, user, logits_processor=None):
            prompts.append((system, user))
            return next(outputs)
        provider._generate_raw = generate
        provider._structured("extract", "same-system", "same-user", ExtractorResult)
        trace = provider.drain_traces()[0]
        self.assertEqual(prompts[0], ("same-system", "same-user"))
        self.assertIn("VALIDATION_RETRY", prompts[1][1])
        self.assertEqual(trace["model_calls"], 2)
        self.assertEqual(trace["prompt_tokens"], 40)
        self.assertEqual(trace["completion_tokens"], 7)
        self.assertTrue(trace["retry_success"])

    def test_second_failure_is_not_retried_again(self):
        provider = self.provider([("bad", 1, 1, False), ("still bad", 1, 1, False)])
        with self.assertRaisesRegex(RuntimeError, "after one"):
            provider._structured("extract", "system", "user", ExtractorResult)
        self.assertEqual(provider.drain_traces()[0]["model_calls"], 2)


class RuleProviderTests(unittest.TestCase):
    def test_rules_are_conservative_typed_and_evidence_grounded(self):
        schema = load_schema()
        provider = V5RuleProvider(schema)
        message = "Forecast the next 12 weeks from sales.csv using weekly data and return as JSON with RMSE."
        result = asyncio.run(provider.extract(message, create_initial_state(schema)))
        values = {item.slot_id: item.candidate_value for item in result.updates}
        self.assertEqual(values["forecast_horizon"], {"periods": 12, "unit": "week"})
        self.assertEqual(values["frequency"], {"periods": 1, "unit": "week"})
        self.assertEqual(values["source_reference"], "sales.csv")
        self.assertEqual(values["output_format"], "json")
        self.assertEqual(values["primary_metric"], "rmse")
        for update in result.updates:
            self.assertIn(update.evidence_text, message)
        trace = provider.drain_traces()[0]
        self.assertEqual(trace["model_calls"], 0)
        self.assertTrue(trace["raw_schema_valid"])

    def test_wire_result_json_encodes_candidate_values(self):
        result = ExtractorResult.model_validate(json.loads(wire([{
            "slot_id": "backtest_folds", "candidate_value": "3", "status": "provided",
            "confidence": 1.0, "evidence_text": "3",
        }])))
        self.assertEqual(_wire_result(result)["updates"][0]["candidate_value"], "3")


class SlotWiseProviderTests(unittest.TestCase):
    def test_control_and_slot_probes_are_combined_with_visible_call_count(self):
        provider = object.__new__(V5SlotWiseTransformersProvider)
        provider._traces = []
        provider.schema = SimpleNamespace(slots=[
            SimpleNamespace(slot_id="intent", description="Intent", allowed_values=()),
            SimpleNamespace(slot_id="target_column", description="Target", allowed_values=()),
        ])
        outputs = iter([
            (ControlProbe(intent="create_forecast", intent_confidence=1,
                          correction_detected=False, unsupported_claims=[]), {"prompt_tokens": 5, "completion_tokens": 3}),
            (SlotProbe(mentioned=True, candidate_value='"sales"', status="provided",
                       confidence=1, evidence_text="sales"), {"prompt_tokens": 7, "completion_tokens": 4}),
        ])
        provider._probe = lambda system, user, output_type: next(outputs)
        with patch("local_slm_lab.structured_output_providers.build_v5_extractor_input",
                   return_value='{"current_message":"sales"}'):
            result = asyncio.run(provider.extract("sales", object()))
        self.assertEqual(result.updates[0].candidate_value, "sales")
        trace = provider.drain_traces()[0]
        self.assertEqual(trace["model_calls"], 2)
        self.assertEqual(trace["strategy"], "slot_wise")
        self.assertTrue(trace["raw_schema_valid"])


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


class OpenAIProviderTests(unittest.TestCase):
    def test_openai_schema_removes_unsupported_annotations(self):
        value = {"$schema": "draft", "$comment": "note", "default": [],
                 "type": "object", "properties": {"x": {"type": "string"}}}
        clean = openai_compatible_schema(value)
        self.assertEqual(clean, {"type": "object", "properties": {"x": {"type": "string"}}})

    def test_responses_request_uses_strict_schema_and_local_validation(self):
        response = SimpleNamespace(output_text=wire(), usage=SimpleNamespace(input_tokens=10, output_tokens=5))
        responses = FakeResponses(response)
        client = SimpleNamespace(responses=responses)
        schema = load_schema()
        output_schema = ExtractorResult.model_json_schema(mode="validation")
        provider = V5OpenAIStructuredProvider(
            "unused-test-key", "explicit-snapshot", schema,
            {"extract": output_schema, "ask": {"type": "object"}}, client=client,
        )
        result = asyncio.run(provider.extract("Forecast sales", create_initial_state(schema)))
        self.assertEqual(result.intent, Intent.CREATE_FORECAST)
        request = responses.requests[0]
        self.assertFalse(request["store"])
        self.assertEqual(request["max_output_tokens"], 1024)
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        trace = provider.drain_traces()[0]
        self.assertEqual(trace["prompt_tokens"], 10)
        self.assertTrue(trace["raw_schema_valid"])


if __name__ == "__main__":
    unittest.main()
