import asyncio
import sys
import unittest
from pathlib import Path


FPY_SRC = Path(__file__).resolve().parents[2] / "fpy" / "src"
if str(FPY_SRC) not in sys.path:
    sys.path.insert(0, str(FPY_SRC))

from forecasting_assistant.domain.models import ExtractorResult, Intent, SlotStatus
from forecasting_assistant.domain.schema import create_initial_state, load_schema
from local_slm_lab.client import LocalModelConfig
from local_slm_lab.providers import LocalFineTunedProvider


class RecordingFineTunedProvider(LocalFineTunedProvider):
    def __init__(self):
        super().__init__(LocalModelConfig(), load_schema())
        self.calls = []

    def _raw_completion(self, system, user):
        self.calls.append((system, user))
        return (
            '{"intent":"create_forecast","intent_confidence":1.0,'
            '"updates":[],"correction_detected":false,"unsupported_claims":[]}'
        )


class WrongSlotFineTunedProvider(RecordingFineTunedProvider):
    def _raw_completion(self, system, user):
        self.calls.append((system, user))
        return (
            '{"intent":"create_forecast","intent_confidence":1.0,'
            '"updates":[{"slot_id":"business_goal","candidate_value":"\\"date\\"",'
            '"status":"provided","confidence":1.0,"evidence_text":"date"}],'
            '"correction_detected":false,"unsupported_claims":[]}'
        )


class GenericProblemFineTunedProvider(RecordingFineTunedProvider):
    def _raw_completion(self, system, user):
        self.calls.append((system, user))
        return (
            '{"intent":"create_forecast","intent_confidence":1.0,'
            '"updates":[{"slot_id":"problem_statement",'
            '"candidate_value":"\\"forecasting\\"","status":"provided",'
            '"confidence":1.0,"evidence_text":"forecasting"}],'
            '"correction_detected":false,"unsupported_claims":[]}'
        )


class LocalFineTunedProviderTests(unittest.TestCase):
    def test_short_yes_uses_deterministic_recovery_without_server(self):
        provider = RecordingFineTunedProvider()
        result = asyncio.run(
            provider.extract("yes", create_initial_state(provider.schema))
        )
        self.assertEqual(result.intent, Intent.CREATE_FORECAST)
        self.assertEqual(provider.calls, [])

    def test_extraction_uses_compact_sft_prompt(self):
        provider = RecordingFineTunedProvider()
        result = asyncio.run(
            provider.extract(
                "Forecast Bitcoin prices", create_initial_state(provider.schema)
            )
        )
        self.assertEqual(result.intent, Intent.CREATE_FORECAST)
        self.assertEqual(len(provider.calls), 1)
        system, user = provider.calls[0]
        self.assertIn("candidate_value must itself be JSON encoded as text", system)
        self.assertIn('"current_message": "Forecast Bitcoin prices"', user)

    def test_short_answer_recovers_the_selected_string_slot(self):
        provider = RecordingFineTunedProvider()
        state = create_initial_state(provider.schema)
        state.intent = Intent.CREATE_FORECAST
        state.slots["intent"].value = Intent.CREATE_FORECAST.value
        state.slots["intent"].status = SlotStatus.PROVIDED
        state.slots["intent"].evidence_text = "yes"

        result = asyncio.run(provider.extract("btc_usd_close", state))

        updates = {update.slot_id: update.candidate_value for update in result.updates}
        self.assertEqual(updates["target_column"], "btc_usd_close")
        self.assertEqual(result.intent, Intent.CREATE_FORECAST)
        self.assertEqual(provider.calls, [])

    def test_short_answer_recovery_rejects_non_answers(self):
        provider = RecordingFineTunedProvider()
        state = create_initial_state(provider.schema)
        state.intent = Intent.CREATE_FORECAST
        state.slots["intent"].value = Intent.CREATE_FORECAST.value
        state.slots["intent"].status = SlotStatus.PROVIDED
        state.slots["intent"].evidence_text = "yes"

        result = asyncio.run(provider.extract("I don't know", state))

        self.assertEqual(result.updates, [])

    def test_sentence_is_not_forced_into_the_selected_slot(self):
        provider = RecordingFineTunedProvider()
        state = create_initial_state(provider.schema)
        state.intent = Intent.CREATE_FORECAST
        state.slots["intent"].value = Intent.CREATE_FORECAST.value
        state.slots["intent"].status = SlotStatus.PROVIDED
        state.slots["intent"].evidence_text = "yes"

        result = asyncio.run(
            provider.extract("I want to forecast factory production output.", state)
        )

        self.assertEqual(result.updates, [])

    def test_short_sentence_with_pronoun_is_not_forced_into_slot(self):
        provider = RecordingFineTunedProvider()
        state = create_initial_state(provider.schema)
        state.intent = Intent.CREATE_FORECAST
        state.slots["intent"].value = Intent.CREATE_FORECAST.value
        state.slots["intent"].status = SlotStatus.PROVIDED
        state.slots["intent"].evidence_text = "yes"

        result = asyncio.run(provider.extract("I will upload production_output.csv.", state))

        self.assertEqual(result.updates, [])

    def test_generic_problem_statement_is_rejected(self):
        provider = GenericProblemFineTunedProvider()
        state = create_initial_state(provider.schema)
        state.intent = Intent.CREATE_FORECAST
        state.slots["intent"].value = Intent.CREATE_FORECAST.value
        state.slots["intent"].status = SlotStatus.PROVIDED
        state.slots["intent"].evidence_text = "yes"
        for slot_id, value in {
            "target_column": "btc_used",
            "time_column": "date",
            "frequency": {"periods": 1, "unit": "day"},
            "forecast_horizon": {"periods": 7, "unit": "day"},
        }.items():
            state.slots[slot_id].value = value
            state.slots[slot_id].status = SlotStatus.PROVIDED
            state.slots[slot_id].evidence_text = str(value)

        result = asyncio.run(provider.extract("forecasting", state))

        self.assertEqual(result.updates, [])

    def test_short_answer_discards_an_unrelated_model_update(self):
        provider = WrongSlotFineTunedProvider()
        state = create_initial_state(provider.schema)
        state.intent = Intent.CREATE_FORECAST
        state.slots["intent"].value = Intent.CREATE_FORECAST.value
        state.slots["intent"].status = SlotStatus.PROVIDED
        state.slots["intent"].evidence_text = "yes"
        state.slots["target_column"].value = "btc_usd_close"
        state.slots["target_column"].status = SlotStatus.PROVIDED
        state.slots["target_column"].evidence_text = "btc_usd_close"

        wrong_result = ExtractorResult.model_validate_json(
            provider._raw_completion("system", "user")
        )
        result = provider._slot_answer_recovery("date", state, wrong_result)

        self.assertIsNotNone(result)
        updates = {update.slot_id: update.candidate_value for update in result.updates}
        self.assertEqual(updates, {"time_column": "date"})


if __name__ == "__main__":
    unittest.main()
