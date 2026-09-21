import unittest
import json
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

from local_slm_lab.peft_provider import (
    RUN_ROOT,
    TransformersPeftProvider,
    adapter_model_class_name,
    load_peft_adapter,
    resolve_adapter_path,
    select_model_class,
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

    def test_adapter_model_class_is_read_from_peft_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "adapter_config.json").write_text(json.dumps({
                "auto_mapping": {
                    "base_model_class": "Qwen3_5ForConditionalGeneration",
                    "parent_library": "transformers.models.qwen3_5.modeling_qwen3_5",
                }
            }), encoding="utf-8")
            self.assertEqual(
                adapter_model_class_name(path), "Qwen3_5ForConditionalGeneration"
            )

    def test_model_class_selection_rejects_training_architecture_drift(self):
        module = SimpleNamespace(Qwen3_5ForConditionalGeneration=object())
        with self.assertRaisesRegex(ValueError, "does not match"):
            select_model_class(
                module, ["Qwen3_5ForConditionalGeneration"], "Qwen3_5ForCausalLM"
            )

    def test_model_class_selection_uses_recorded_architecture(self):
        expected = object()
        module = SimpleNamespace(Qwen3_5ForConditionalGeneration=expected)
        self.assertIs(
            select_model_class(module, ["Qwen3_5ForConditionalGeneration"], None),
            expected,
        )

    def test_adapter_load_rejects_peft_missing_key_warning(self):
        class FakePeftModel:
            @staticmethod
            def from_pretrained(base, path):
                warnings.warn("Found missing adapter keys while loading the checkpoint")
                return object()

        with self.assertRaisesRegex(RuntimeError, "did not load"):
            load_peft_adapter(FakePeftModel, object(), Path("adapter"))


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

    def test_duration_normalization_from_scalar_and_evidence(self):
        dur = TransformersPeftProvider._parse_duration_candidate(
            "6", "next 6 months", "I want to forecast monthly sales for the next 6 months"
        )
        self.assertEqual(dur, {"periods": 6, "unit": "month"})

    def test_duration_normalization_from_nested_json(self):
        dur = TransformersPeftProvider._parse_duration_candidate(
            '{"periods": 6.0, "unit": "month"}', "", ""
        )
        self.assertEqual(dur, {"periods": 6, "unit": "month"})

    def test_clean_and_normalize_updates_filters_prompt_leak_evidence(self):
        from forecasting_assistant.domain.models import ExtractorResult, SlotUpdate, SlotStatus
        raw_res = ExtractorResult(
            intent=Intent.CREATE_FORECAST,
            intent_confidence=1.0,
            updates=[
                SlotUpdate(
                    slot_id="forecast_horizon",
                    candidate_value="6",
                    status=SlotStatus.PROVIDED,
                    confidence=1.0,
                    evidence_text="next 6 months",
                ),
                SlotUpdate(
                    slot_id="frequency",
                    candidate_value='{"periods": 6.0, "unit": "month"}',
                    status=SlotStatus.PROVIDED,
                    confidence=1.0,
                    evidence_text='confirmed_slots: ["frequency"]',
                ),
            ],
        )
        cleaned = self.provider._clean_and_normalize_updates(
            raw_res, "I want to forecast monthly sales for the next 6 months"
        )
        self.assertEqual(
            cleaned.updates[0].candidate_value,
            {"periods": 6, "unit": "month"}
        )
        self.assertEqual(
            cleaned.updates[1].candidate_value,
            {"periods": 6, "unit": "month"}
        )
        self.assertEqual(
            cleaned.updates[1].evidence_text,
            "I want to forecast monthly sales for the next 6 months"
        )


if __name__ == "__main__":
    unittest.main()
