"""Raw v5 component scoring, guarded loading, and paired scenario bootstrap.

Extraction is teacher-forced (gold reducer replay plus explicit reviewed context).
Questions are oracle-conditioned component tests, NOT end-to-end dialogue tests.
No provider recovery, output repair, secret-bearing exception messages, or final
label access is performed on import.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
FPY = ROOT.parent / "fpy"
if str(FPY / "src") not in sys.path:
    sys.path.insert(0, str(FPY / "src"))

from forecasting_assistant.application.state_reducer import apply_extraction
from forecasting_assistant.domain.models import (
    DialogueState, DialogueTurn, ExtractorResult, Intent, QuestionOutput,
    QuestionRequest, SlotState, SlotStatus,
)
from forecasting_assistant.domain.schema import create_initial_state, load_schema
from forecasting_assistant.prompts.llmrei_long import validate_question

SCORER_VERSION = "v5-raw-exact-1"
FPY_PIN = "04d52c015d1e3ecdefe92b87116f209361509b4b"
PROTOCOL = {
    "extraction_context": "teacher_forced_gold_reducer_replay_with_explicit_context_overlays",
    "last_assistant_question": "explicit turn/scenario question; never predicted",
    "question_context": "oracle_gold_final_slots_excluding_requested_slot",
    "inclusive_f1": "exact update tuples including intent slot; top-level intent scored separately",
    "canonical_value": "decode candidate_value JSON text once, preserve types and list order",
    "invalid_prediction": "zero TP, all gold FN, structurally readable emitted tuples FP",
    "zero_denominator": "P/R/F1=0; other rates=null; empty-gold scenarios disclosed separately",
    "transition": "same pinned reducer on gold/prediction; intent and all slot values/status/confirmation/errors",
    "question_contract": "pinned fpy validator, not semantic answer quality",
    "correction_tuple_accuracy": "gold explicit-correction calls; flag true and all nonintent update tuples exact",
    "empty_update_collapse": "nonempty-gold extraction calls with failed/invalid or empty output; successful-empty also reported",
    "variant_context": "declared parent branches replay independently; unbranched turns replay sequentially",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_error(exc: BaseException) -> dict[str, Any]:
    """Never serialize exception messages, bodies, requests, or headers."""
    result: dict[str, Any] = {"type": type(exc).__name__}
    current: BaseException | None = exc
    for _ in range(5):
        if current is None:
            break
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            result["status_code"] = status
        current = current.__cause__
    return result


def strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError("non-finite JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def raw_validity(raw: str | None, task: str) -> tuple[bool | None, bool | None, Any]:
    """Validate syntax independently of the strict wire contract (no Pydantic coercion)."""
    if raw is None:
        return None, None, None
    try:
        value = strict_json(raw)
    except (ValueError, TypeError):
        return False, False, None
    if not isinstance(value, dict):
        return True, False, value
    if task == "ask":
        valid = (set(value) == {"question"} and isinstance(value["question"], str)
                 and 1 <= len(value["question"]) <= 300)
        return True, valid, value
    fields = {"intent", "intent_confidence", "updates", "correction_detected", "unsupported_claims"}
    update_fields = {"slot_id", "candidate_value", "status", "confidence", "evidence_text"}
    number = lambda x: type(x) in (int, float) and 0 <= x <= 1
    valid = (
        set(value) == fields
        and isinstance(value.get("intent"), str)
        and value.get("intent") in {item.value for item in Intent}
        and number(value.get("intent_confidence"))
        and type(value.get("correction_detected")) is bool
        and isinstance(value.get("unsupported_claims"), list)
        and all(isinstance(x, str) for x in value.get("unsupported_claims", []))
        and isinstance(value.get("updates"), list)
    )
    if valid:
        for update in value["updates"]:
            if not (isinstance(update, dict) and set(update) == update_fields
                    and isinstance(update["slot_id"], str) and bool(update["slot_id"])
                    and isinstance(update["candidate_value"], str)
                    and isinstance(update["status"], str)
                    and update["status"] in {item.value for item in SlotStatus}
                    and number(update["confidence"]) and isinstance(update["evidence_text"], str)):
                valid = False
                break
            try:
                strict_json(update["candidate_value"])
            except (ValueError, TypeError):
                valid = False
                break
        if _duplicates(value):
            valid = False
    return True, bool(valid), value


def _slot_ids(result: Any) -> list[str]:
    if not isinstance(result, dict) or not isinstance(result.get("updates"), list):
        return []
    return [u["slot_id"] for u in result["updates"]
            if isinstance(u, dict) and isinstance(u.get("slot_id"), str)]


def _duplicates(result: dict) -> bool:
    ids = _slot_ids(result)
    return len(ids) != len(set(ids))


def tuples(result: dict | None) -> list[tuple[str, str, str]]:
    """Input is the decoded model dump, not wire candidate_value strings."""
    if result is None:
        return []
    return [(u["slot_id"], dumps(u.get("candidate_value")), u["status"])
            for u in result.get("updates", [])]


def _decoded(result: dict) -> dict:
    return ExtractorResult.model_validate(result).model_dump(mode="json")


def _forbidden(scenario: dict, turn: dict) -> list[str]:
    return sorted(set(scenario.get("must_not_infer", [])) | set(turn.get("must_not_infer", []))
                  | set(scenario.get("forbidden_slots", [])) | set(turn.get("forbidden_slots", [])))


def validate_scenarios(scenarios: list[dict], schema=None) -> None:
    schema = schema or load_schema()
    known = {s.slot_id for s in schema.slots}
    seen = set()
    if not scenarios:
        raise ValueError("empty cases")
    for scenario in scenarios:
        sid = scenario.get("scenario_id", scenario.get("id"))
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("missing or duplicate scenario ID")
        if "id" in scenario and "scenario_id" in scenario and scenario["id"] != sid:
            raise ValueError("conflicting scenario ID aliases")
        seen.add(sid)
        if not isinstance(scenario.get("turns"), list) or not scenario["turns"]:
            raise ValueError("scenario requires extraction turns")
        if set(scenario.get("gold_final_slots", {})) - known:
            raise ValueError("unknown gold final slot")
        for turn in scenario["turns"]:
            gold = turn["gold_extraction"]
            if _duplicates(gold):
                raise ValueError("duplicate gold slot ID")
            parsed = _decoded(gold)
            ids = set(_slot_ids(parsed))
            forbidden = set(_forbidden(scenario, turn))
            if (ids | forbidden) - known:
                raise ValueError("unknown gold or forbidden slot")
            if ids & forbidden:
                raise ValueError("gold/forbidden contradiction")
            if not isinstance(turn.get("message"), str):
                raise ValueError("turn requires message")
        qids = []
        for index, question in enumerate(question_cases(scenario)):
            qids.append(str(question.get("id", index)))
            slot = question.get("slot_id", question.get("expected_question_slot"))
            if slot not in known:
                raise ValueError("unknown question slot")
            QuestionOutput(question=question.get("ideal_question", question.get("question")))
        if len(qids) != len(set(qids)):
            raise ValueError("duplicate question case ID")


def question_cases(scenario: dict) -> list[dict]:
    if "question_cases" in scenario:
        return scenario["question_cases"]
    if scenario.get("expected_question_slot"):
        return [{"slot_id": scenario["expected_question_slot"], "ideal_question": scenario["ideal_question"]}]
    return []


def _overlay(state: DialogueState, values: dict) -> None:
    for slot_id, value in values.items():
        if slot_id not in state.slots:
            raise ValueError("unknown context slot")
        # Canonical initial/context slots map IDs to values, not serialized SlotStates.
        state.slots[slot_id] = SlotState(
            slot_id=slot_id, value=value, status=SlotStatus.CONFIRMED, confidence=1,
            confirmed_by_user=True, evidence_text="synthetic reviewed context",
        )
        if slot_id == "intent":
            state.intent = Intent(value)


def _snapshot_value(value: Any) -> Any:
    # datetime slots normalize to Python datetime/date objects; emit the same
    # ISO 8601 text pydantic mode="json" and now() use so records stay serializable.
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def state_snapshot(state: DialogueState) -> dict:
    return {
        "intent": state.intent.value,
        "slots": {key: {"value": _snapshot_value(value.value), "status": value.status.value,
                        "confirmed_by_user": value.confirmed_by_user,
                        "validation_errors": value.validation_errors}
                  for key, value in sorted(state.slots.items())},
        "turns": [{"user_message": t.user_message, "assistant_message": t.assistant_message}
                  for t in state.turns],
    }


def build_call_plan(scenarios: list[dict], schema=None) -> list[dict]:
    """Preflight ALL gold and transitions before the first provider call."""
    schema = schema or load_schema()
    validate_scenarios(scenarios, schema)
    plan = []
    for scenario in scenarios:
        sid = scenario.get("scenario_id", scenario.get("id"))
        cluster = scenario.get("source_scenario_id", scenario.get("cluster_id", sid))
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("invalid source scenario cluster")
        state = create_initial_state(schema)
        _overlay(state, scenario.get("initial_slots", {}))
        initial = state.model_copy(deep=True)
        branch_states = {}
        for index, turn in enumerate(scenario["turns"], 1):
            # Canonical v5 paraphrases share a parent; they are not sequential replies.
            if "parent_id" in turn:
                parent = turn["parent_id"]
                if parent is None or parent == sid:
                    state = initial.model_copy(deep=True)
                else:
                    parent_key = parent.removeprefix(sid + "/")
                    if parent_key not in branch_states:
                        raise ValueError("unknown/forward turn parent")
                    state = branch_states[parent_key].model_copy(deep=True)
            _overlay(state, turn.get("context_slots", {}))
            previous = turn.get("last_assistant_question", scenario.get("last_assistant_question") if index == 1 else None)
            if previous:
                state.turns.append(DialogueTurn(turn_number=len(state.turns) + 1,
                                               user_message="[earlier turn]", assistant_message=previous))
            expected_model = ExtractorResult.model_validate(turn["gold_extraction"])
            after = apply_extraction(state, expected_model, schema, index, turn["message"])
            context = {"message": turn["message"], "state": state_snapshot(state)}
            plan.append({"key": [sid, "extract", index], "scenario_id": sid,
                         "cluster_id": cluster, "category": scenario.get("category", "uncategorized"),
                         "task": "extract", "expected": expected_model.model_dump(mode="json"),
                         "forbidden_slots": _forbidden(scenario, turn), "context": context,
                         "state": state.model_copy(deep=True), "gold_after": state_snapshot(after)})
            state = after
            # No predicted answers enter the next turn, including after failures.
            state.turns.append(DialogueTurn(turn_number=len(state.turns) + 1, user_message=turn["message"]))
            turn_key = turn.get("turn_id", str(index))
            if turn_key in branch_states:
                raise ValueError("duplicate turn ID")
            branch_states[turn_key] = state.model_copy(deep=True)
        for index, question in enumerate(question_cases(scenario)):
            slot_id = question.get("slot_id", question.get("expected_question_slot"))
            definition = schema.get(slot_id)
            request = QuestionRequest(
                slot_id=slot_id, reason=question.get("reason", "missing or unclear requirement"),
                slot_description=definition.description, current_state=SlotState(slot_id=slot_id),
                confirmed_context={k: v for k, v in scenario.get("gold_final_slots", {}).items() if k != slot_id},
                static_question=definition.static_question, allowed_values=definition.allowed_values,
                other_active_slot_ids=tuple(s.slot_id for s in schema.slots if s.slot_id != slot_id),
            )
            plan.append({"key": [sid, "ask", str(question.get("id", index))], "scenario_id": sid,
                         "cluster_id": cluster, "category": scenario.get("category", "uncategorized"),
                         "task": "ask", "expected": {"slot_id": slot_id, "question": question.get("ideal_question", question.get("question"))},
                         "forbidden_slots": [], "context": request.model_dump(mode="json"), "request": request})
    return plan


def _trace(provider: Any, task: str) -> dict:
    traces = provider.drain_traces() if hasattr(provider, "drain_traces") else []
    clean = []
    for trace in traces:
        clean.append({k: trace.get(k) for k in (
            "operation", "raw_output", "raw_json_valid", "raw_schema_valid", "prompt_tokens", "completion_tokens", "parsed"
        )})
        # Do not trust historical/provider error strings to be secret-free.
        clean[-1]["error"] = "provider_trace_error" if trace.get("error") else None
    relevant = [t for t in clean if t["operation"] == task]
    selected = relevant[-1] if len(relevant) == 1 else {}
    raw = selected.get("raw_output")
    syntax, schema, value = raw_validity(raw, task)
    return {"traces": clean, "raw_output": raw, "raw_json_valid": syntax,
            "raw_schema_valid": schema, "raw_value": value,
            "prompt_tokens": selected.get("prompt_tokens"), "completion_tokens": selected.get("completion_tokens"),
            "trace_contract_valid": len(relevant) == 1 and len(clean) == 1 if traces else None}


async def evaluate(provider: Any, scenarios: list[dict], *, schema=None,
                   metadata: dict | None = None, on_record: Callable[[dict], None] | None = None,
                   plan: list[dict] | None = None) -> dict:
    schema = schema or load_schema()
    plan = build_call_plan(scenarios, schema) if plan is None else plan
    records = []
    started_at = now()
    for call in plan:
        record = {k: call[k] for k in ("key", "scenario_id", "cluster_id", "category", "task", "expected", "forbidden_slots", "context")}
        record["context_sha256"] = digest(record["context"])
        record["gold_sha256"] = digest({"expected": record["expected"], "forbidden_slots": record["forbidden_slots"]})
        record.update(predicted=None, provider_success=False, error=None, transition_correct=False, started_at=now())
        if call["task"] == "extract":
            record["gold_after"] = call["gold_after"]
        started = perf_counter()
        try:
            if call["task"] == "extract":
                output = await provider.extract(call["context"]["message"], call["state"].model_copy(deep=True))
            else:
                output = await provider.ask(call["request"].model_copy(deep=True))
            record["predicted"] = output.model_dump(mode="json")
            record["provider_success"] = True
        except Exception as exc:
            record["error"] = safe_error(exc)
        record["latency_ms"] = (perf_counter() - started) * 1000
        record["finished_at"] = now()
        record.update(_trace(provider, call["task"]))
        record["emitted"] = record["predicted"]
        if isinstance(record["raw_value"], dict) and call["task"] == "extract":
            raw_updates = record["raw_value"].get("updates")
            if isinstance(raw_updates, list):
                emitted = []
                for update in raw_updates:
                    if not (isinstance(update, dict) and isinstance(update.get("slot_id"), str)
                            and isinstance(update.get("status"), str) and "candidate_value" in update):
                        continue
                    value = update["candidate_value"]
                    if isinstance(value, str):
                        try:
                            value = strict_json(value)
                        except ValueError:
                            pass
                    emitted.append(dict(update, candidate_value=value))
                record["emitted"] = {"updates": emitted}
        record["duplicate_slot_ids"] = bool(call["task"] == "extract" and record["emitted"] and _duplicates(record["emitted"]))
        record["prediction_valid"] = (record["provider_success"] and record["raw_schema_valid"] is not False
                                      and not record["duplicate_slot_ids"] and record["trace_contract_valid"] is not False)
        if call["task"] == "extract" and record["prediction_valid"]:
            try:
                if not isinstance(output, ExtractorResult):
                    raise TypeError("provider must return ExtractorResult")
                after = apply_extraction(call["state"], output, schema, call["key"][2], call["context"]["message"])
                record["predicted_after"] = state_snapshot(after)
                record["gold_after"] = call["gold_after"]
                record["transition_correct"] = record["predicted_after"] == record["gold_after"]
            except Exception as exc:
                record["transition_error"] = safe_error(exc)
        elif call["task"] == "ask":
            record["question_contract"] = bool(record["prediction_valid"] and validate_question(output, call["request"]))
            record["question_exact_match"] = bool(record["prediction_valid"] and record["predicted"]["question"] == call["expected"]["question"])
        records.append(record)
        if on_record:
            on_record(record)
    return {"report_version": SCORER_VERSION, "status": "complete", "started_at": started_at, "finished_at": now(),
            "protocol": PROTOCOL, "metadata": metadata or {}, "scenario_count": len(scenarios),
            "scenario_ids": [s.get("scenario_id", s.get("id")) for s in scenarios],
            "metrics": score_records(records, [s.slot_id for s in schema.slots]),
            "category_metrics": {category: score_records([r for r in records if r["category"] == category])
                                 for category in sorted({r["category"] for r in records})}, "records": records}


def confusion(record: dict, *, nonintent: bool = False) -> tuple[int, int, int]:
    gold = Counter(t for t in tuples(record["expected"]) if not nonintent or t[0] != "intent")
    emitted = Counter(t for t in tuples(record.get("emitted", record.get("predicted"))) if not nonintent or t[0] != "intent")
    valid = record.get("prediction_valid", False)
    if _duplicates(record["expected"]):
        raise ValueError("duplicate gold slot ID")
    if set(_slot_ids(record["expected"])) & set(record.get("forbidden_slots", [])):
        raise ValueError("gold/forbidden contradiction")
    if _duplicates(record.get("emitted") or record.get("predicted") or {}):
        valid = False
    overlap = sum((gold & emitted).values()) if valid else 0
    return overlap, sum(emitted.values()) - overlap, sum(gold.values()) - overlap


def prf(tp: int, fp: int, fn: int) -> dict:
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "support": int(tp + fn),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def rate(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def score_records(records: list[dict], slot_ids=()) -> dict:
    extraction = [r for r in records if r["task"] == "extract"]
    questions = [r for r in records if r["task"] == "ask"]
    total = {"inclusive": [0, 0, 0], "nonintent": [0, 0, 0]}
    by_slot = defaultdict(lambda: [0, 0, 0])
    for slot in slot_ids:
        by_slot[slot]
    scenarios = defaultdict(list)
    for r in extraction:
        scenarios[r["scenario_id"]].append(r)
        for label in total:
            for i, count in enumerate(confusion(r, nonintent=label == "nonintent")):
                total[label][i] += count
        ids = set(_slot_ids(r["expected"])) | set(_slot_ids(r.get("emitted") or {}))
        for slot in ids:
            one = dict(r)
            one["expected"] = dict(r["expected"], updates=[u for u in r["expected"]["updates"] if u["slot_id"] == slot])
            one["emitted"] = {"updates": [u for u in (r.get("emitted") or {}).get("updates", []) if u["slot_id"] == slot]}
            if _duplicates(r.get("emitted") or {}):
                one["prediction_valid"] = False
            for i, count in enumerate(confusion(one)):
                by_slot[slot][i] += count
    def good(r):
        return bool(r.get("prediction_valid") and not _duplicates(r.get("emitted") or {}))
    def field_correct(r, field):
        return good(r) and r["predicted"].get(field) == r["expected"].get(field)
    def has_nonintent(r):
        return any(t[0] != "intent" for t in tuples(r["expected"]))
    nonempty_scenarios = [rs for rs in scenarios.values() if any(has_nonintent(r) for r in rs)]
    def all_gold(rs):
        return all(good(r) and confusion(r, nonintent=True)[2] == 0 for r in rs)
    def exact(rs):
        return all(good(r) and confusion(r, nonintent=True)[1:] == (0, 0) for r in rs)
    corrections = [r for r in extraction if r["expected"]["correction_detected"]]
    forbidden_calls = [r for r in extraction if r.get("forbidden_slots")]
    successful_forbidden = [r for r in forbidden_calls if r["provider_success"]]
    def observed_ids(r):
        return set(_slot_ids(r.get("raw_value"))) | set(_slot_ids(r.get("emitted")))
    def forbidden_count(r):
        return len(observed_ids(r) & set(r.get("forbidden_slots", [])))
    hallucinated = [len(observed_ids(r) - set(_slot_ids(r["expected"]))) for r in extraction]
    nonempty = [r for r in extraction if tuples(r["expected"])]
    latencies = sorted(r["latency_ms"] for r in records)
    metrics = {label: prf(*values) for label, values in total.items()}
    metrics.update({
        "per_slot": {slot: prf(*values) for slot, values in sorted(by_slot.items())},
        "scenario_all_gold_nonintent": rate(sum(all_gold(rs) for rs in scenarios.values()), len(scenarios)),
        "scenario_all_gold_nonintent_nonempty": rate(sum(all_gold(rs) for rs in nonempty_scenarios), len(nonempty_scenarios)),
        "scenario_exact_nonintent": rate(sum(exact(rs) for rs in scenarios.values()), len(scenarios)),
        "empty_gold_nonintent_scenarios": len(scenarios) - len(nonempty_scenarios),
        "intent_accuracy": rate(sum(field_correct(r, "intent") for r in extraction), len(extraction)),
        "correction_flag_accuracy": rate(sum(field_correct(r, "correction_detected") for r in extraction), len(extraction)),
        "correction_tuple_accuracy": rate(sum(field_correct(r, "correction_detected") and confusion(r, nonintent=True)[1:] == (0, 0) for r in corrections), len(corrections)),
        "reducer_transition_accuracy": rate(sum(r.get("transition_correct", False) and good(r) for r in extraction), len(extraction)),
        "forbidden_call_rate": rate(sum(bool(forbidden_count(r)) for r in forbidden_calls), len(forbidden_calls)),
        "forbidden_slot_rate": rate(sum(forbidden_count(r) for r in forbidden_calls), sum(len(r["forbidden_slots"]) for r in forbidden_calls)),
        "forbidden_call_rate_conditional_success": rate(sum(bool(forbidden_count(r)) for r in successful_forbidden), len(successful_forbidden)),
        "forbidden_failed_calls": sum(not r["provider_success"] for r in forbidden_calls),
        "hallucinated_slot_call_rate": rate(sum(bool(n) for n in hallucinated), len(extraction)),
        "hallucinated_slot_call_rate_conditional_success": rate(
            sum(bool(observed_ids(r) - set(_slot_ids(r["expected"]))) for r in extraction if r["provider_success"]),
            sum(r["provider_success"] for r in extraction)),
        "hallucinated_slot_count": sum(hallucinated),
        "empty_update_collapse": rate(sum(not good(r) or not tuples(r.get("emitted")) for r in nonempty), len(nonempty)),
        "successful_empty_update_collapse": rate(sum(good(r) and not tuples(r.get("emitted")) for r in nonempty), len(nonempty)),
        "duplicate_prediction_calls": sum(r.get("duplicate_slot_ids", False) for r in extraction),
        "provider_success": rate(sum(r["provider_success"] for r in records), len(records)),
        "question_contract": rate(sum(r.get("question_contract", False) for r in questions), len(questions)),
        "question_exact_match": rate(sum(r.get("question_exact_match", False) for r in questions), len(questions)),
        "calls": {"total": len(records), "extract": len(extraction), "ask": len(questions)},
        "latency_ms": {"mean": sum(latencies) / len(latencies) if latencies else None,
                       "p50": latencies[(len(latencies) - 1) // 2] if latencies else None,
                       "p95": latencies[max(0, (95 * len(latencies) + 99) // 100 - 1)] if latencies else None},
    })
    for field in ("raw_json_valid", "raw_schema_valid"):
        measured = [r for r in records if r.get(field) is not None]
        metrics[field] = dict(rate(sum(r.get(field) is True for r in records), len(records)),
                              measured_calls=len(measured), unavailable_calls=len(records) - len(measured),
                              measured_rate=sum(r[field] is True for r in measured) / len(measured) if measured else None)
    metrics["tokens"] = {field: {"total_observed": sum(r.get(field) or 0 for r in records),
                                 "measured_calls": sum(r.get(field) is not None for r in records),
                                 "unavailable_calls": sum(r.get(field) is None for r in records)}
                        for field in ("prompt_tokens", "completion_tokens")}
    metrics["forbidden_interpretation"] = "observed forbidden emissions only; failed/unparseable calls are NOT certified safe"
    return metrics


def _is_sealed(path: Path) -> bool:
    parts = {part.lower() for part in path.resolve().parts}
    return bool(parts & {"v5-sealed", "sealed", "final", "final.jsonl", "test.jsonl"})


def _date(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("dated selection/review required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
        raise ValueError("selection dates require timezone and cannot be future")
    return parsed


def authorize_final(manifest: dict | None, *, reviewed_final: bool, candidate: str | None, settings: dict | None) -> dict:
    if not reviewed_final or not candidate or not manifest or not settings:
        raise PermissionError("final requires reviewed-final gate, manifest, and candidate declaration")
    if manifest.get("manifest_version") != "v5-selection-1" or manifest.get("selection_split") != "validation":
        raise ValueError("validation-only selection manifest required")
    selected_at = _date(manifest.get("selected_at"))
    review = manifest.get("review", {})
    is_hash = lambda value: isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))
    placeholders = {"todo", "pending", "placeholder", "unknown", "tbd", "none"}
    if (review.get("completed") is not True or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"].strip() or review["reviewer"].strip().lower() in placeholders
            or not is_hash(review.get("artifact_sha256"))):
        raise ValueError("completed named review and hashed review artifact required")
    if _date(review.get("completed_at")) > selected_at:
        raise ValueError("review must precede frozen selection")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("final_cases_sha256", "")):
        raise ValueError("exact final cases hash required")
    candidates = manifest.get("candidates", [])
    ids = [c.get("candidate_id") for c in candidates]
    if not ids or len(set(ids)) != len(ids) or any(not i for i in ids):
        raise ValueError("unique candidate declarations required")
    required = {"provider", "model", "revision", "adapter", "dtype", "device", "max_new_tokens", "decoding", "prompt_sha256", "scorer_sha256", "dependency_sha256", "adapter_sha256"}
    for entry in candidates:
        declared = entry.get("settings", {})
        if set(declared) != required or not is_hash(entry.get("validation_report_sha256")):
            raise ValueError("each candidate needs exact settings and validation report hash")
        if (not isinstance(declared.get("model"), str) or not declared["model"].strip()
                or declared["model"].lower() in placeholders or declared["provider"] not in {"base", "lora", "openai"}):
            raise ValueError("explicit model/provider required")
        if any(not is_hash(declared[k]) for k in ("prompt_sha256", "scorer_sha256", "dependency_sha256")):
            raise ValueError("candidate artifact hashes must be exact")
        if not isinstance(declared["decoding"], dict) or not declared["decoding"]:
            raise ValueError("candidate decoding settings required")
        if declared["provider"] != "openai":
            if (not isinstance(declared["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", declared["revision"])
                    or declared["dtype"] not in {"float32", "float16", "bfloat16"}
                    or not isinstance(declared["device"], str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", declared["device"])
                    or type(declared["max_new_tokens"]) is not int or declared["max_new_tokens"] < 1):
                raise ValueError("final local runtime and immutable revision must be explicit")
        if declared["provider"] == "lora":
            if not declared["adapter"] or not isinstance(declared["adapter_sha256"], dict) or not declared["adapter_sha256"]:
                raise ValueError("exact LoRA adapter artifacts required")
            if any(not is_hash(v) for v in declared["adapter_sha256"].values()):
                raise ValueError("invalid adapter artifact hash")
        elif declared["adapter"] is not None or declared["adapter_sha256"] is not None:
            raise ValueError("non-LoRA candidate cannot declare an adapter")
    if candidate not in ids or candidates[ids.index(candidate)]["settings"] != settings:
        raise ValueError("candidate settings differ from frozen declaration")
    if manifest.get("selected_candidate_id") not in ids:
        raise ValueError("selected candidate must be declared")
    if not manifest.get("all_candidates_declared") is True:
        raise ValueError("explicit all-candidate declaration required")
    return manifest


def load_cases(path: Path, *, split="dev", manifest_path: Path | None = None,
               reviewed_final=False, candidate=None, settings=None) -> tuple[list[dict], dict]:
    path = path.resolve()
    if split != "final" and _is_sealed(path):
        raise PermissionError("sealed final cannot be loaded as development data")
    manifest = None
    if split == "final":
        # Authorization intentionally precedes even stat/hash/open of cases.
        if not reviewed_final or not manifest_path or not candidate or not settings:
            raise PermissionError("final access not authorized")
        manifest = authorize_final(strict_json(manifest_path.read_text(encoding="utf-8")),
                                   reviewed_final=reviewed_final, candidate=candidate, settings=settings)
    payload = path.read_bytes()
    cases_hash = hashlib.sha256(payload).hexdigest()
    if manifest and cases_hash != manifest["final_cases_sha256"]:
        raise ValueError("final cases hash differs from frozen manifest")
    scenarios = [strict_json(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    for row in scenarios:
        label = row.get("split")
        if split != "final" and label in {"final", "test"}:
            raise PermissionError("final-labelled data is not development data")
        if split in {"train", "validation", "smoke"} and label is not None and label != split:
            raise ValueError("declared split differs from cases")
    validate_scenarios(scenarios)
    return scenarios, {"path": str(path), "sha256": cases_hash, "split": split,
                       "selection_manifest_sha256": sha256(manifest_path) if manifest else None,
                       "candidate_id": candidate if manifest else None}


def claim_final_run(cases_hash: str, candidate: str, output: Path, *, marker_root: Path | None = None,
                    settings_sha256: str | None = None) -> Path:
    """A marker is never removed, including failed/partial runs. No silent retries."""
    root = marker_root or ROOT / "training-runs" / "v5-sealed" / "v5-20260919-r1"
    root.mkdir(parents=True, exist_ok=True)
    identity = {"cases": cases_hash, "settings": settings_sha256} if settings_sha256 else {"cases": cases_hash, "candidate": candidate}
    marker = root / ("run-" + digest(identity) + ".json")
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"claimed_at": now(), "cases_sha256": cases_hash, "candidate_id": candidate,
                   "settings_sha256": settings_sha256,
                   "output": str(output), "status": "claimed_no_automatic_retry"}, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return marker


def write_report(report: dict, output: Path) -> None:
    # Parents must already exist: a typo must not create an unintended run directory.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def provenance(provider: str) -> dict:
    fpy_revision = subprocess.check_output(["git", "-C", str(FPY), "rev-parse", "HEAD"], text=True).strip()
    if fpy_revision != FPY_PIN:
        raise ValueError("fpy dependency revision differs from v5 pin")
    dependency_files = sorted((FPY / "src" / "forecasting_assistant").rglob("*.py"))
    dependency_hashes = {str(p.relative_to(FPY)): sha256(p) for p in dependency_files}
    local_files = [ROOT / "local_slm_lab" / name for name in (
        "v5_eval.py", "peft_provider.py", "providers.py", "slm_prompts.py", "v5_prompts.py", "v5_provider.py")]
    local_files.append(ROOT / "scripts/evaluate-v5.py")
    local_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in local_files if p.is_file()}
    if provider == "openai":
        prompt_files = [FPY / "src/forecasting_assistant/prompts" / name for name in ("extractor.py", "llmrei_long.py")]
    else:
        prompt_files = [ROOT / "local_slm_lab/v5_prompts.py", ROOT / "local_slm_lab/slm_prompts.py"]
    prompt_hashes = {p.name: sha256(p) for p in prompt_files}
    versions = {}
    for package in ("pydantic", "numpy", "torch", "transformers", "peft", "openai", "tokenizers", "safetensors", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"fpy_revision": fpy_revision, "dependency_files": dependency_hashes,
            "dependency_sha256": digest({"fpy": dependency_hashes, "local": local_hashes}), "local_files": local_hashes,
            "prompt_files": prompt_hashes, "prompt_sha256": digest(prompt_hashes),
            "scorer_sha256": sha256(Path(__file__)), "versions": versions,
            "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                         "cpu_count": os.cpu_count(), "python": sys.version},
            "repository_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}


def runtime_metadata(provider: Any) -> dict:
    model = getattr(provider, "_model", None)
    generation = getattr(model, "generation_config", None)
    result = {"actual_device": str(getattr(provider, "_input_device", "unavailable")),
              "actual_dtype": str(getattr(model, "dtype", "unavailable")),
              "resolved_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
              "chat_template_sha256": getattr(provider, "chat_template_sha256", None),
              "generation_config": generation.to_dict() if generation is not None else None}
    torch = getattr(provider, "_torch", None)
    if torch is not None:
        result["cuda_version"] = torch.version.cuda
        result["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        result["device_map"] = {k: str(v) for k, v in (getattr(model, "hf_device_map", None) or {}).items()}
    return result


def instrument_openai(provider: Any) -> bool:
    """Wrap only response.parse's result; never retain SDK request kwargs/credentials."""
    try:
        responses = provider._client._client.responses
        original = responses.parse
        traces = []

        async def parse(**kwargs):
            task = "extract" if kwargs.get("text_format") is ExtractorResult else "ask"
            try:
                response = await original(**kwargs)
            except Exception as exc:
                traces.append({"operation": task, "raw_output": None, "error": safe_error(exc), "parsed": False})
                raise
            raw = getattr(response, "output_text", None)
            syntax, schema, _ = raw_validity(raw, task)
            usage = getattr(response, "usage", None)
            traces.append({"operation": task, "raw_output": raw, "raw_json_valid": syntax,
                           "raw_schema_valid": schema, "prompt_tokens": getattr(usage, "input_tokens", None),
                           "completion_tokens": getattr(usage, "output_tokens", None),
                           "parsed": getattr(response, "output_parsed", None) is not None, "error": None})
            return response

        def drain():
            result = list(traces)
            traces.clear()
            return result

        responses.parse = parse
        provider.drain_traces = drain
        return True
    except (AttributeError, TypeError):
        return False


def _paired_records(left: dict, right: dict, allow_runtime_difference: bool) -> tuple[list[tuple[dict, dict]], list[str]]:
    if left.get("status") != "complete" or right.get("status") != "complete":
        raise ValueError("partial reports cannot be compared")
    if left.get("report_version") != SCORER_VERSION or right.get("report_version") != SCORER_VERSION:
        raise ValueError("incompatible scorer version")
    if left.get("protocol") != right.get("protocol"):
        raise ValueError("evaluation protocols differ")
    if sorted(left["scenario_ids"]) != sorted(right["scenario_ids"]) or len(set(left["scenario_ids"])) != len(left["scenario_ids"]):
        raise ValueError("scenario sets differ or contain duplicates")
    for report in (left, right):
        if {r["scenario_id"] for r in report["records"]} != set(report["scenario_ids"]):
            raise ValueError("missing scenario records")
    lm, rm = left["metadata"], right["metadata"]
    if lm["input"]["sha256"] != rm["input"]["sha256"] or lm["input"]["split"] != rm["input"]["split"]:
        raise ValueError("input hashes or splits differ")
    ls, rs = lm["settings"], rm["settings"]
    for key in ("scorer_sha256", "dependency_sha256"):
        if ls[key] != rs[key]:
            raise ValueError("scorer/dependency hashes differ")
    differences = []
    if ls["provider"] != "openai" and rs["provider"] != "openai":
        for key in ("model", "revision", "dtype", "device", "prompt_sha256", "decoding", "max_new_tokens"):
            if ls[key] != rs[key]:
                differences.append(key)
        for key in ("actual_device", "actual_dtype", "resolved_revision", "gpu_names", "device_map", "chat_template_sha256", "generation_config", "cuda_version"):
            if lm.get("runtime", {}).get(key) != rm.get("runtime", {}).get(key):
                differences.append("runtime." + key)
        if lm.get("provenance", {}).get("versions") != rm.get("provenance", {}).get("versions"):
            differences.append("dependency_versions")
        if lm.get("model_artifacts") != rm.get("model_artifacts"):
            differences.append("model_artifact_hashes")
        if differences and not allow_runtime_difference:
            raise ValueError("local runtime/model/prompt mismatch; explicit override required")
    else:
        differences.append("openai_separate_provider_prompt_and_runtime")
    def index(report):
        values = {tuple(r["key"]): r for r in report["records"]}
        if len(values) != len(report["records"]):
            raise ValueError("duplicate call key")
        return values
    li, ri = index(left), index(right)
    if set(li) != set(ri):
        raise ValueError("paired call keys differ")
    pairs = []
    for key in sorted(li, key=str):
        a, b = li[key], ri[key]
        for field in ("scenario_id", "cluster_id", "task", "expected", "forbidden_slots", "context", "context_sha256", "gold_sha256"):
            if a.get(field) != b.get(field):
                raise ValueError("paired gold/context/cluster differs")
        if a["task"] == "extract":
            confusion(a)
            confusion(b)
        pairs.append((a, b))
    return pairs, differences


def paired_bootstrap(left: dict, right: dict, *, samples=10000, seed=42, allow_runtime_difference=False) -> dict:
    """Paired right-minus-left micro F1; resample SOURCE clusters, not calls."""
    if samples < 10000:
        raise ValueError("at least 10000 bootstrap samples required")
    import numpy as np
    pairs, differences = _paired_records(left, right, allow_runtime_difference)
    clusters = sorted({a["cluster_id"] for a, _ in pairs})
    if not clusters:
        raise ValueError("no clusters")
    index = {key: i for i, key in enumerate(clusters)}
    counts = np.zeros((len(clusters), 2, 2, 3), dtype=np.int64)
    for a, b in pairs:
        if a["task"] != "extract":
            continue
        for provider_index, record in enumerate((a, b)):
            for metric_index in (0, 1):
                counts[index[a["cluster_id"]], provider_index, metric_index] += confusion(record, nonintent=bool(metric_index))
    def f1(values):
        tp, fp, fn = values[..., 0], values[..., 1], values[..., 2]
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    point = f1(counts.sum(axis=0))
    rng = np.random.default_rng(seed)
    deltas = np.empty((samples, 2))
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        selected = rng.integers(0, len(clusters), size=(size, len(clusters)))
        scores = f1(counts[selected].sum(axis=1))
        deltas[start:start + size] = scores[:, 1, :] - scores[:, 0, :]
    result = {"comparison_version": SCORER_VERSION, "created_at": now(), "direction": "right_minus_left",
              "samples": samples, "seed": seed, "cluster_count": len(clusters), "cluster_unit": "source_scenario_id/cluster_id; otherwise scenario_id",
              "interval": "paired percentile 95%; exact integer-count micro F1 recomputed per sample",
              "runtime_differences": differences, "runtime_difference_override": bool(differences and allow_runtime_difference)}
    for i, metric in enumerate(("inclusive", "nonintent")):
        result[metric] = {"left_f1": float(point[0, i]), "right_f1": float(point[1, i]),
                          "delta": float(point[1, i] - point[0, i]),
                          "ci95": [float(v) for v in np.quantile(deltas[:, i], [0.025, 0.975])]}
    return result


def select_validation_candidate(base: dict, candidates: dict[str, dict]) -> dict:
    """Return a recommendation only; never auto-authorize final review/selection."""
    if base["metadata"]["input"]["split"] != "validation" or base["metadata"]["settings"]["provider"] != "base":
        raise ValueError("selection requires a validation base report")
    baseline = score_records(base["records"])
    results = {}
    for name, candidate in candidates.items():
        if candidate["metadata"]["input"]["split"] != "validation" or candidate["metadata"]["settings"]["provider"] != "lora":
            raise ValueError("adapter selection is validation-only")
        _paired_records(base, candidate, False)
        metrics = score_records(candidate["records"])
        reasons = []
        for metric, tolerance in (("forbidden_call_rate", 0), ("forbidden_slot_rate", 0),
                                  ("raw_json_valid", .01), ("raw_schema_valid", .01),
                                  ("correction_flag_accuracy", .01), ("correction_tuple_accuracy", .01)):
            before, after = baseline[metric]["rate"], metrics[metric]["rate"]
            if before is None or after is None:
                reasons.append(metric + " unavailable")
            elif metric in {"raw_json_valid", "raw_schema_valid"} and (baseline[metric]["unavailable_calls"] or metrics[metric]["unavailable_calls"]):
                reasons.append(metric + " unavailable")
            elif (after > before + 1e-12 if metric.startswith("forbidden_") else after < before - tolerance - 1e-12):
                reasons.append(metric + " regression")
        results[name] = {"eligible": not reasons, "nonintent_f1": metrics["nonintent"]["f1"],
                         "delta_vs_base": metrics["nonintent"]["f1"] - baseline["nonintent"]["f1"], "reasons": reasons}
    eligible = [name for name in results if results[name]["eligible"]]
    winner = max(eligible, key=lambda name: (results[name]["nonintent_f1"], name)) if eligible else None
    return {"primary_metric": "validation_nonintent_micro_f1", "loss_best_is_not_selection": True,
            "recommended_candidate": winner, "candidates": results, "review_completed": False,
            "final_authorized": False, "tie_break": "lexicographically greatest candidate ID"}
