"""End-to-end behavioral evaluation for conversational SLM state progression."""

from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
if str(FPY_SRC) not in sys.path:
    sys.path.insert(0, str(FPY_SRC))

from forecasting_assistant.domain.models import Intent, SlotStatus
from forecasting_assistant.domain.schema import ForecastingSchema
from local_slm_lab.memory_repository import MemoryRepository
from local_slm_lab.slm_engine import LocalSLMElicitationEngine


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mentioned_count(state) -> int:
    return sum(slot.status != SlotStatus.UNMENTIONED for slot in state.slots.values())


def score_conversations(records: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [turn for record in records for turn in record["turns"]]
    trace_items = [
        trace
        for turn in turns
        for trace in turn.get("traces", [])
        if trace.get("operation") in {"extract", "ask"}
    ]
    repeat_denominator = sum(max(0, len(record["turns"]) - 1) for record in records)
    return {
        "case_success_rate": round(
            sum(record["success"] for record in records) / len(records), 4
        )
        if records
        else 0.0,
        "intent_accuracy": round(
            sum(record["intent_correct"] for record in records) / len(records), 4
        )
        if records
        else 0.0,
        "state_progression_rate": round(
            sum(turn["state_progressed"] for turn in turns) / len(turns), 4
        )
        if turns
        else 0.0,
        "consecutive_question_repeat_rate": round(
            sum(turn["repeated_previous_question"] for turn in turns) / repeat_denominator,
            4,
        )
        if repeat_denominator
        else 0.0,
        "structured_output_validity_rate": round(
            sum(trace.get("parsed", False) for trace in trace_items) / len(trace_items), 4
        )
        if trace_items
        else 0.0,
        "average_turns": round(statistics.fmean(len(record["turns"]) for record in records), 2)
        if records
        else 0.0,
        "cases": len(records),
        "turns": len(turns),
    }


async def evaluate_conversations(
    provider,
    cases: list[dict[str, Any]],
    schema: ForecastingSchema,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in cases:
        repository = MemoryRepository()
        engine = LocalSLMElicitationEngine(schema, provider, repository)
        dialogue = engine.start_dialogue()
        turns: list[dict[str, Any]] = []
        previous_question: str | None = None
        for message in case["messages"]:
            before = engine.get_state(dialogue.dialogue_id)
            result = await engine.handle_user_message(dialogue.dialogue_id, message)
            after = result.state
            traces = provider.drain_traces() if hasattr(provider, "drain_traces") else []
            assistant = result.assistant_message.strip()
            turns.append(
                {
                    "user": message,
                    "assistant": assistant,
                    "state_progressed": (
                        _mentioned_count(after) > _mentioned_count(before)
                        or after.intent != before.intent
                    ),
                    "repeated_previous_question": bool(
                        previous_question and assistant == previous_question
                    ),
                    "unresolved_slots": result.readiness.unresolved_slots,
                    "traces": traces,
                }
            )
            previous_question = assistant

        final_state = engine.get_state(dialogue.dialogue_id)
        final_slots = {
            slot_id: slot.value
            for slot_id, slot in final_state.slots.items()
            if slot.value is not None and slot.status != SlotStatus.UNMENTIONED
        }
        expected_intent = Intent(case["expected_intent"])
        expected_slots = case.get("expected_final_slots", {})
        intent_correct = final_state.intent == expected_intent
        slots_correct = all(final_slots.get(key) == value for key, value in expected_slots.items())
        records.append(
            {
                "case_id": case["case_id"],
                "expected_intent": expected_intent.value,
                "predicted_intent": final_state.intent.value,
                "intent_correct": intent_correct,
                "expected_final_slots": expected_slots,
                "final_slots": final_slots,
                "slots_correct": slots_correct,
                "success": intent_correct and slots_correct,
                "turns": turns,
            }
        )

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "provider": provider.name,
        "model": provider.model,
        "metrics": score_conversations(records),
        "records": records,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
