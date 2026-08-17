"""SLM-only orchestration safeguards; the production fpy engine stays unchanged."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
if str(FPY_SRC) not in sys.path:
    sys.path.insert(0, str(FPY_SRC))

from forecasting_assistant.application.clarification import select_next_slot  # noqa: E402
from forecasting_assistant.application.orchestrator import ElicitationEngine  # noqa: E402
from forecasting_assistant.domain.models import TurnResult  # noqa: E402


class LocalSLMElicitationEngine(ElicitationEngine):
    """Prevent an unchanged SLM state from producing an endless identical question."""

    async def handle_user_message(self, dialogue_id, message: str) -> TurnResult:
        result = await super().handle_user_message(dialogue_id, message)
        state = result.state.model_copy(deep=True)
        if len(state.turns) < 2:
            return result

        current = (state.turns[-1].assistant_message or "").strip()
        previous = (state.turns[-2].assistant_message or "").strip()
        if not current or current != previous:
            return result

        candidate = select_next_slot(self._schema, state)
        if candidate is None:
            replacement = "I could not reliably record that answer. Please restate it in one short sentence."
        else:
            definition = self._schema.get(candidate.slot_id)
            label = definition.description.rstrip(".")
            if candidate.slot_id == "intent":
                replacement = (
                    "I could not reliably record your forecasting intent. "
                    "Please reply 'yes, create a forecast' or 'no'."
                )
            elif definition.allowed_values:
                choices = ", ".join(definition.allowed_values)
                replacement = f"I could not record {label}. Please choose one of: {choices}."
            else:
                replacement = f"I could not reliably record {label}. Please answer in one short phrase."

        state.turns[-1].assistant_message = replacement
        self._repository.save_state(state)
        self._repository.append_event(
            dialogue_id,
            "repeated_question_prevented",
            {"turn_number": state.turns[-1].turn_number, "replacement": replacement},
        )
        return result.model_copy(update={"state": state, "assistant_message": replacement})

