import unittest
from uuid import UUID

from local_slm_lab.slm_engine import LocalSLMElicitationEngine
from forecasting_assistant.domain.models import (
    DialogueState,
    ExtractorResult,
    ForecastingSpecification,
    Intent,
)
from forecasting_assistant.domain.schema import load_schema


class MemoryRepository:
    def __init__(self):
        self.states: dict[UUID, DialogueState] = {}
        self.events: list[tuple[str, dict]] = []

    def load_state(self, dialogue_id):
        state = self.states.get(dialogue_id)
        return None if state is None else state.model_copy(deep=True)

    def save_state(self, state):
        self.states[state.dialogue_id] = state.model_copy(deep=True)

    def append_event(self, dialogue_id, event_type, payload):
        self.events.append((event_type, payload))

    def save_specification(self, specification: ForecastingSpecification):
        pass


class FailingProvider:
    async def extract(self, message, state):
        raise RuntimeError("invalid JSON")

    async def ask(self, request):
        raise RuntimeError("invalid JSON")


class RepeatedIntentProvider:
    async def extract(self, message, state):
        return ExtractorResult(
            intent=Intent.CREATE_FORECAST,
            intent_confidence=1.0,
            updates=[],
        )

    async def ask(self, request):
        raise AssertionError("local engine should use the schema question")


class RepeatProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_engine_uses_schema_question_without_model_generation(self):
        repository = MemoryRepository()
        engine = LocalSLMElicitationEngine(load_schema(), FailingProvider(), repository)
        state = engine.start_dialogue()
        request = engine._question_request(state, "intent", "missing")

        question = await engine._ask(request)

        self.assertEqual(question, "Do you want the system to create a time-series forecast?")

    async def test_identical_question_is_replaced_on_second_turn(self):
        repository = MemoryRepository()
        engine = LocalSLMElicitationEngine(load_schema(), FailingProvider(), repository)
        dialogue = engine.start_dialogue()

        first = await engine.handle_user_message(dialogue.dialogue_id, "unclear")
        second = await engine.handle_user_message(dialogue.dialogue_id, "still unclear")

        self.assertNotEqual(second.assistant_message, first.assistant_message)
        self.assertIn("could not reliably record", second.assistant_message.lower())
        self.assertTrue(
            any(event == "repeated_question_prevented" for event, _ in repository.events)
        )

    async def test_unchanged_intent_keeps_original_evidence(self):
        repository = MemoryRepository()
        engine = LocalSLMElicitationEngine(load_schema(), RepeatedIntentProvider(), repository)
        dialogue = engine.start_dialogue()

        first = await engine.handle_user_message(dialogue.dialogue_id, "yes")
        second = await engine.handle_user_message(dialogue.dialogue_id, "uploaded")

        self.assertEqual(first.state.slots["intent"].evidence_text, "yes")
        self.assertEqual(second.state.slots["intent"].evidence_text, "yes")
        self.assertEqual(second.state.slots["intent"].source_turn, 1)


if __name__ == "__main__":
    unittest.main()
