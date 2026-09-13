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
from forecasting_assistant.application.orchestrator import ElicitationEngine
from local_slm_lab.memory_repository import MemoryRepository
from local_slm_lab.slm_engine import LocalSLMElicitationEngine


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mentioned_count(state) -> int:
    return sum(slot.status != SlotStatus.UNMENTIONED for slot in state.slots.values())


def _canonical_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _slot_pairs(values: dict[str, Any]) -> set[tuple[str, str]]:
    return {(slot_id, _canonical_value(value)) for slot_id, value in values.items()}


def score_conversations(records: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [turn for record in records for turn in record["turns"]]
    all_trace_items = [trace for turn in turns for trace in turn.get("traces", [])]
    trace_items = [
        trace
        for trace in all_trace_items
        if trace.get("operation") in {"extract", "ask"}
    ]
    deterministic_turns = sum(
        any(
            trace.get("operation") in {"deterministic_intent", "deterministic_slot"}
            for trace in turn.get("traces", [])
        )
        for turn in turns
    )
    repeat_denominator = sum(max(0, len(record["turns"]) - 1) for record in records)
    expected_pairs = set()
    predicted_pairs = set()
    expected_slot_ids = set()
    predicted_slot_ids = set()
    for index, record in enumerate(records):
        expected = record.get("expected_final_slots", {})
        predicted = record.get("final_slots", {})
        expected_pairs.update((index, *pair) for pair in _slot_pairs(expected))
        predicted_pairs.update((index, *pair) for pair in _slot_pairs(predicted))
        expected_slot_ids.update((index, slot_id) for slot_id in expected)
        predicted_slot_ids.update((index, slot_id) for slot_id in predicted)

    pair_tp = len(expected_pairs & predicted_pairs)
    pair_precision = pair_tp / len(predicted_pairs) if predicted_pairs else 0.0
    pair_recall = pair_tp / len(expected_pairs) if expected_pairs else 0.0
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if pair_precision + pair_recall
        else 0.0
    )
    slot_id_tp = len(expected_slot_ids & predicted_slot_ids)
    slot_id_precision = slot_id_tp / len(predicted_slot_ids) if predicted_slot_ids else 0.0
    slot_id_recall = slot_id_tp / len(expected_slot_ids) if expected_slot_ids else 0.0
    slot_id_f1 = (
        2 * slot_id_precision * slot_id_recall / (slot_id_precision + slot_id_recall)
        if slot_id_precision + slot_id_recall
        else 0.0
    )
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
        "exact_final_state_accuracy": round(
            sum(record.get("slots_exact", False) for record in records) / len(records), 4
        )
        if records
        else 0.0,
        "slot_micro_precision": round(pair_precision, 4),
        "slot_micro_recall": round(pair_recall, 4),
        "slot_micro_f1": round(pair_f1, 4),
        "slot_id_micro_precision": round(slot_id_precision, 4),
        "slot_id_micro_recall": round(slot_id_recall, 4),
        "slot_id_micro_f1": round(slot_id_f1, 4),
        "unexpected_slot_inference_rate": round(
            len(predicted_slot_ids - expected_slot_ids) / len(predicted_slot_ids), 4
        )
        if predicted_slot_ids
        else 0.0,
        "assistant_safety_rate": round(
            sum(record.get("assistant_safe", True) for record in records) / len(records), 4
        )
        if records
        else 0.0,
        "state_progression_rate": round(
            sum(turn["state_progressed"] for turn in turns) / len(turns), 4
        )
        if turns
        else 0.0,
        "stalled_turn_rate": round(
            sum(not turn["state_progressed"] for turn in turns) / len(turns), 4
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
        else None,
        "structured_model_calls": len(trace_items),
        "deterministic_turn_rate": round(deterministic_turns / len(turns), 4)
        if turns
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
    engine_class: type[ElicitationEngine] = LocalSLMElicitationEngine,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in cases:
        repository = MemoryRepository()
        engine = engine_class(schema, provider, repository)
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
        slots_exact = final_slots == expected_slots
        forbidden_terms = [term.lower() for term in case.get("forbidden_assistant_terms", [])]
        assistant_text = "\n".join(turn["assistant"] for turn in turns).lower()
        assistant_safe = not any(term in assistant_text for term in forbidden_terms)
        records.append(
            {
                "case_id": case["case_id"],
                "expected_intent": expected_intent.value,
                "predicted_intent": final_state.intent.value,
                "intent_correct": intent_correct,
                "expected_final_slots": expected_slots,
                "final_slots": final_slots,
                "slots_correct": slots_exact,
                "slots_exact": slots_exact,
                "assistant_safe": assistant_safe,
                "forbidden_assistant_terms": case.get("forbidden_assistant_terms", []),
                "success": intent_correct and slots_exact and assistant_safe,
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
