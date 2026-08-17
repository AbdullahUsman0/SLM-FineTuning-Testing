"""Shared component-level evaluation for local and hosted LLM clients."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
if str(FPY_SRC) not in sys.path:
    sys.path.insert(0, str(FPY_SRC))

from forecasting_assistant.domain.models import (
    DialogueState,
    ExtractorResult,
    QuestionOutput,
    QuestionRequest,
    SlotState,
    SlotStatus,
)
from forecasting_assistant.domain.schema import ForecastingSchema, create_initial_state, load_schema
from forecasting_assistant.prompts.llmrei_long import validate_question


class StructuredProvider(Protocol):
    name: str
    model: str

    async def extract(self, message: str, state: DialogueState) -> ExtractorResult: ...

    async def ask(self, request: QuestionRequest) -> QuestionOutput: ...


@dataclass(frozen=True)
class CallRecord:
    scenario_id: str
    category: str
    task: str
    turn_index: int | None
    expected: dict[str, Any]
    predicted: dict[str, Any] | None
    latency_ms: float
    error: str | None
    checks: dict[str, bool]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def state_with_context(schema: ForecastingSchema, context: dict[str, Any]) -> DialogueState:
    state = create_initial_state(schema)
    for slot_id, value in context.items():
        state.slots[slot_id] = SlotState(
            slot_id=slot_id,
            value=value,
            status=SlotStatus.CONFIRMED,
            confidence=1.0,
            evidence_text="reviewed evaluation context",
            confirmed_by_user=True,
        )
    return state


def question_request(scenario: dict[str, Any], schema: ForecastingSchema) -> QuestionRequest:
    slot_id = scenario["expected_question_slot"]
    definition = schema.get(slot_id)
    return QuestionRequest(
        slot_id=slot_id,
        reason="missing or unclear requirement",
        slot_description=definition.description,
        current_state=SlotState(slot_id=slot_id),
        confirmed_context={},
        static_question=definition.static_question,
        allowed_values=definition.allowed_values,
        other_active_slot_ids=tuple(
            item.slot_id for item in schema.slots if item.slot_id != slot_id
        ),
    )


def _canonical_value(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _updates(result: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            update["slot_id"],
            _canonical_value(update.get("candidate_value")),
            update["status"],
        )
        for update in result.get("updates", [])
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 1)


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def score_records(records: list[CallRecord]) -> dict[str, Any]:
    extraction = [record for record in records if record.task == "extract"]
    questions = [record for record in records if record.task == "ask"]
    successful = [record for record in records if record.predicted is not None]
    latencies = [record.latency_ms for record in records]

    true_positive = false_positive = false_negative = 0
    intent_correct = joint_correct = correction_correct = 0
    forbidden_total = forbidden_inferred = 0
    for record in extraction:
        if record.predicted is None:
            false_negative += len(_updates(record.expected))
            continue
        expected_updates = _updates(record.expected)
        predicted_updates = _updates(record.predicted)
        true_positive += len(expected_updates & predicted_updates)
        false_positive += len(predicted_updates - expected_updates)
        false_negative += len(expected_updates - predicted_updates)
        intent_correct += int(record.checks["intent"])
        joint_correct += int(record.checks["joint_extraction"])
        correction_correct += int(record.checks["correction"])
        forbidden_total += int(record.checks.get("has_forbidden_contract", False))
        forbidden_inferred += int(not record.checks.get("avoids_forbidden_slots", True))

    precision = _safe_rate(true_positive, true_positive + false_positive)
    recall = _safe_rate(true_positive, true_positive + false_negative)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return {
        "provider_success_rate": _safe_rate(len(successful), len(records)),
        "structured_output_validity_rate": _safe_rate(len(successful), len(records)),
        "schema_validity_rate": _safe_rate(len(successful), len(records)),
        "intent_accuracy": _safe_rate(intent_correct, len(extraction)),
        "slot_micro_precision": precision,
        "slot_micro_recall": recall,
        "slot_micro_f1": f1,
        "joint_extraction_accuracy": _safe_rate(joint_correct, len(extraction)),
        "correction_detection_accuracy": _safe_rate(correction_correct, len(extraction)),
        "forbidden_slot_inference_rate": _safe_rate(forbidden_inferred, forbidden_total),
        "question_contract_accuracy": _safe_rate(
            sum(record.checks.get("question_contract", False) for record in questions),
            len(questions),
        ),
        "one_question_compliance": _safe_rate(
            sum(record.checks.get("one_question", False) for record in questions),
            len(questions),
        ),
        "question_slot_relevance": _safe_rate(
            sum(record.checks.get("slot_relevance", False) for record in questions),
            len(questions),
        ),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(statistics.median(latencies), 1) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
        },
        "calls": {"total": len(records), "extract": len(extraction), "ask": len(questions)},
    }


def _extraction_checks(
    expected: dict[str, Any], predicted: dict[str, Any], forbidden_slots: list[str]
) -> dict[str, bool]:
    predicted_slot_ids = {item["slot_id"] for item in predicted.get("updates", [])}
    return {
        "intent": predicted.get("intent") == expected.get("intent"),
        "joint_extraction": _updates(predicted) == _updates(expected)
        and predicted.get("intent") == expected.get("intent"),
        "correction": predicted.get("correction_detected")
        == expected.get("correction_detected"),
        "has_forbidden_contract": bool(forbidden_slots),
        "avoids_forbidden_slots": not bool(predicted_slot_ids & set(forbidden_slots)),
    }


def _question_checks(output: QuestionOutput, request: QuestionRequest) -> dict[str, bool]:
    question = output.question.strip()
    expected_terms = {
        word.lower()
        for word in (request.slot_id.replace("_", " ") + " " + request.slot_description).split()
        if len(word) >= 4
    }
    static_terms = {
        word.lower().strip("?.,")
        for word in request.static_question.split()
        if len(word.strip("?.,")) >= 4
    }
    normalized = {word.lower().strip("?.,") for word in question.split()}
    return {
        "question_contract": validate_question(output, request),
        "one_question": question.count("?") == 1 and question.endswith("?"),
        "slot_relevance": bool(normalized & (expected_terms | static_terms)),
    }


async def evaluate(
    provider: StructuredProvider,
    scenarios: list[dict[str, Any]],
    *,
    schema: ForecastingSchema | None = None,
) -> dict[str, Any]:
    schema = schema or load_schema()
    records: list[CallRecord] = []
    for scenario in scenarios:
        for turn_index, turn in enumerate(scenario["turns"], start=1):
            expected = turn["gold_extraction"]
            predicted: dict[str, Any] | None = None
            error: str | None = None
            started = perf_counter()
            try:
                state = state_with_context(schema, turn.get("context_slots", {}))
                result = await provider.extract(turn["message"], state)
                predicted = result.model_dump(mode="json")
            except Exception as exc:  # provider errors are part of evaluation
                error = f"{type(exc).__name__}: {exc}"
            latency = round((perf_counter() - started) * 1000, 1)
            checks = (
                _extraction_checks(expected, predicted, scenario["must_not_infer"])
                if predicted is not None
                else {}
            )
            records.append(
                CallRecord(
                    scenario_id=scenario["scenario_id"],
                    category=scenario["category"],
                    task="extract",
                    turn_index=turn_index,
                    expected=expected,
                    predicted=predicted,
                    latency_ms=latency,
                    error=error,
                    checks=checks,
                )
            )

        if scenario.get("expected_question_slot"):
            request = question_request(scenario, schema)
            expected = {"question": scenario["ideal_question"], "slot_id": request.slot_id}
            predicted = None
            error = None
            started = perf_counter()
            try:
                output = await provider.ask(request)
                predicted = output.model_dump(mode="json")
                checks = _question_checks(output, request)
            except Exception as exc:  # provider errors are part of evaluation
                error = f"{type(exc).__name__}: {exc}"
                checks = {}
            latency = round((perf_counter() - started) * 1000, 1)
            records.append(
                CallRecord(
                    scenario_id=scenario["scenario_id"],
                    category=scenario["category"],
                    task="ask",
                    turn_index=None,
                    expected=expected,
                    predicted=predicted,
                    latency_ms=latency,
                    error=error,
                    checks=checks,
                )
            )

    category_counts = Counter(record.category for record in records)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "provider": provider.name,
        "model": provider.model,
        "scenario_count": len(scenarios),
        "category_call_counts": dict(sorted(category_counts.items())),
        "metrics": score_records(records),
        "records": [asdict(record) for record in records],
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
