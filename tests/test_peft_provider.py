import unittest
import json
from pathlib import Path

from local_slm_lab.peft_provider import (
    RUN_ROOT,
    TransformersPeftProvider,
    resolve_adapter_path,
)
from local_slm_lab.slm_prompts import build_slm_extractor_input

from forecasting_assistant.domain.models import DialogueTurn, Intent
from forecasting_assistant.domain.schema import create_initial_state, load_schema


class PeftArtifactRoutingTests(unittest.TestCase):
    def test_base_has_no_adapter(self):
        self.assertIsNone(resolve_adapter_path("base"))

    def test_best_and_final_are_separate(self):
        self.assertEqual(resolve_adapter_path("best"), RUN_ROOT / "best-adapter")
        self.assertEqual(resolve_adapter_path("final"), RUN_ROOT / "checkpoint-165")
        self.assertNotEqual(resolve_adapter_path("best"), resolve_adapter_path("final"))

    def test_custom_requires_path(self):
        with self.assertRaises(ValueError):
            resolve_adapter_path("custom")
        custom = resolve_adapter_path("custom", Path("adapter"))
        self.assertTrue(custom.is_absolute())


class PeftGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema()
        self.provider = object.__new__(TransformersPeftProvider)
        self.provider.schema = self.schema

    def test_json_object_accepts_fenced_or_prefixed_json(self):
        self.assertEqual(
            TransformersPeftProvider._json_object('```json\n{"question":"What?"}\n```'),
            {"question": "What?"},
        )

    def test_batch_decode_falls_back_to_nested_tokenizer(self):
        class Tokenizer:
            def batch_decode(self, token_ids, *, skip_special_tokens):
                self.call = (token_ids, skip_special_tokens)
                return ["decoded"]

        class Processor:
            tokenizer = Tokenizer()

        token_ids = object()
        self.assertEqual(
            TransformersPeftProvider._batch_decode(Processor(), token_ids),
            ["decoded"],
        )
        self.assertEqual(Processor.tokenizer.call, (token_ids, True))
        self.assertEqual(
            TransformersPeftProvider._json_object('result: {"question":"What?"} done'),
            {"question": "What?"},
        )

    def test_yes_recovers_create_forecast_intent(self):
        result = self.provider._intent_recovery("yes", create_initial_state(self.schema))
        self.assertIsNotNone(result)
        self.assertEqual(result.intent, Intent.CREATE_FORECAST)
        self.assertEqual(result.updates[0].slot_id, "intent")

    def test_roman_urdu_intent_is_recovered(self):
        result = self.provider._intent_recovery(
            "Haan mujhe Bitcoin ka forecast banana hai",
            create_initial_state(self.schema),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.intent, Intent.CREATE_FORECAST)

    def test_compact_prompt_includes_previous_question_and_at_most_24_slots(self):
        state = create_initial_state(self.schema)
        state.turns.append(
            DialogueTurn(
                turn_number=1,
                user_message="start",
                assistant_message="Do you want a forecast?",
            )
        )
        payload = json.loads(build_slm_extractor_input("yes", state, self.schema))
        self.assertEqual(payload["selected_slot"], "intent")
        self.assertEqual(payload["last_assistant_question"], "Do you want a forecast?")
        self.assertLessEqual(len(payload["slot_definitions"]), 24)
        self.assertLess(len(payload["slot_definitions"]), len(self.schema.slots))


if __name__ == "__main__":
    unittest.main()
