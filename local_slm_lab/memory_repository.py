"""In-memory repository used only by isolated SLM chat and evaluation runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
if str(FPY_SRC) not in sys.path:
    sys.path.insert(0, str(FPY_SRC))

from forecasting_assistant.domain.models import DialogueState, ForecastingSpecification


class MemoryRepository:
    def __init__(self) -> None:
        self.states: dict[UUID, DialogueState] = {}
        self.events: list[dict[str, Any]] = []
        self.specifications: dict[UUID, ForecastingSpecification] = {}

    def load_state(self, dialogue_id: UUID) -> DialogueState | None:
        state = self.states.get(dialogue_id)
        return None if state is None else state.model_copy(deep=True)

    def save_state(self, state: DialogueState) -> None:
        self.states[state.dialogue_id] = state.model_copy(deep=True)

    def append_event(
        self, dialogue_id: UUID, event_type: str, payload: dict[str, Any]
    ) -> None:
        self.events.append(
            {"dialogue_id": str(dialogue_id), "event_type": event_type, "payload": payload}
        )

    def save_specification(self, specification: ForecastingSpecification) -> None:
        self.specifications[specification.dialogue_id] = specification.model_copy(deep=True)
