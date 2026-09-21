"""Offline, fact-first v5 corpus. Final labels are written only to a sealed root.

The component records deliberately retain corpus.py's scenario/turn/extraction
shape. Evidence spans, atomic facts and review provenance are additional fields.
No old examples, model calls, training, or purported human decisions are used.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from local_slm_lab.corpus import FPY_ROOT, PROJECT_ROOT, _sha256, _update
from local_slm_lab.component_eval import question_request, state_with_context
from local_slm_lab.slm_prompts import build_slm_question_input, build_slm_question_instructions
from local_slm_lab.v5_prompts import (
    PROMPT_VERSION, build_v5_extractor_input, build_v5_extractor_instructions,
)
from forecasting_assistant.application.clarification import select_next_slot
from forecasting_assistant.application.normalization import normalize_value
from forecasting_assistant.application.state_reducer import apply_extraction
from forecasting_assistant.application.validation import validate_slot
from forecasting_assistant.domain.models import ExtractorResult, Intent, QuestionOutput, SlotState
from forecasting_assistant.domain.schema import load_schema
from forecasting_assistant.prompts.llmrei_long import validate_question

VERSION = "v5-20260919-r1"
FPY_COMMIT = "04d52c015d1e3ecdefe92b87116f209361509b4b"
REVIEW_STATUS = "machine_validated_pending_human"
DOMAINS = (
    ("sales_demand", "orders sold", "orders"),
    ("inventory", "stock withdrawals", "items"),
    ("finance_budget", "operating expenditure", "fictional credits"),
    ("staffing", "unfilled shifts", "shifts"),
    ("energy", "electricity consumption", "kWh"),
    ("weather", "rainfall totals", "millimetres"),
    ("transport", "passenger boardings", "passengers"),
    ("synthetic_health_ops", "simulated appointment arrivals", "appointments"),
    ("education", "course enrolments", "enrolments"),
    ("web_traffic", "page visits", "visits"),
    ("supply_chain", "outbound consignments", "consignments"),
)
BOTTLENECKS = ("file_format", "target_description", "problem_statement", "source_mode", "source_reference")
DURATION_UNITS = {"second", "minute", "hour", "day", "week", "month", "quarter", "year", "forecast_horizon"}
# Each template is a statement, not a slot-name/value dump. Quotes delimit exact
# user wording, including case, punctuation, and non-ASCII characters.
CLAUSES = {
    "intent": "Please create a time-series forecast",
    "problem_statement": "The prediction task is {v}",
    "business_goal": "Our operational goal is {v}",
    "stakeholder_role": "My role in this project is {v}",
    "decision_supported": "This forecast will inform the decision {v}",
    "success_criteria": "I will consider the result useful if it meets {v}",
    "target_column": "The target field is named {v}",
    "target_description": "The target represents {v}",
    "target_unit": "The target is measured in {v}",
    "target_bounds": "Valid target bounds are {v}",
    "aggregation_method": "Aggregate observations using {v}",
    "time_column": "The timestamp field is named {v}",
    "frequency": "Observations arrive every {v}",
    "timezone": "The timestamps use timezone {v}",
    "calendar_type": "Use the {v} calendar",
    "forecast_horizon": "Predict the next {v}",
    "forecast_start": "The first forecast period begins at {v}",
    "data_cutoff": "Do not use observations after {v}",
    "lead_time": "Allow a lead time of {v}",
    "dataset_type": "The dataset structure is {v}",
    "series_id_columns": "Each series is identified by the columns {v}",
    "hierarchy_columns": "The ordered hierarchy columns are {v}",
    "aggregation_level": "Return results at the level {v}",
    "scope_filters": "Limit the scope using these explicit filters: {v}",
    "geography": "Include only these fictional areas: {v}",
    "source_mode": "Access the data through {v}",
    "source_reference": "The non-secret source identifier is {v}",
    "source_provider": "The fictional data provider is {v}",
    "file_format": "The file format is {v}",
    "sheet_or_table": "Read the worksheet or resource named {v}",
    "authentication_reference": "Use the stored credential reference {v}, not a raw credential",
    "refresh_frequency": "The source is refreshed every {v}",
    "history_start": "The historical observations begin at {v}",
    "history_end": "The historical observations end at {v}",
    "expected_history_length": "The available historical coverage is {v}",
    "minimum_training_points": "Require at least {v} training observations",
    "known_regime_changes": "The recorded structural breaks are {v}",
    "missing_timestamp_policy": "For missing periods, use the policy {v}",
    "missing_target_policy": "For missing target values, use the policy {v}",
    "duplicate_policy": "For duplicate observations, use the policy {v}",
    "outlier_policy": "For unusual values, use the policy {v}",
    "invalid_value_policy": "For invalid values, use the policy {v}",
    "minimum_coverage": "Require non-missing coverage of at least {v} percent",
    "seasonal_periods": "The seasonal cycles in observation periods are {v}",
    "holidays": "Use these fictional holiday calendars: {v}",
    "special_events": "The known special events are {v}",
    "past_covariates": "These regressors are observed only through the cutoff: {v}",
    "known_future_covariates": "These regressors are known for future periods: {v}",
    "static_features": "The fixed attributes of each series are {v}",
    "covariate_availability": "The future regressor availability records are {v}",
    "external_covariate_sources": "The external regressor source mappings are {v}",
    "forecast_type": "The requested forecast output type is {v}",
    "prediction_interval_levels": "Use prediction interval coverage percentages {v}",
    "quantiles": "Return forecast quantiles at probabilities {v}",
    "scenario_forecasts": "The alternative future scenarios are {v}",
    "rounding_rule": "Round the output according to {v}",
    "output_granularity": "The output resolution must be {v}",
    "primary_metric": "Rank forecasts by the primary metric {v}",
    "secondary_metrics": "Also report accuracy using {v}",
    "validation_strategy": "Use {v} validation",
    "backtest_folds": "Use {v} historical backtest windows",
    "test_window": "Each holdout window spans {v}",
    "baseline_model": "Include {v} as the baseline comparison",
    "acceptable_error": "The acceptable error threshold is {v}",
    "inference_mode": "Generate predictions in {v} mode",
    "prediction_frequency": "Generate new forecasts every {v}",
    "retraining_frequency": "Retrain the model every {v}",
    "latency_requirement": "The maximum response time is {v}",
    "output_format": "Serialize the results as {v}",
    "destination": "Deliver completed forecasts to {v}",
    "privacy_constraints": "Enforce these privacy restrictions: {v}",
    "license": "The source data usage license is {v}",
    "explainability_level": "The required explanation level is {v}",
}
BOOL_CLAUSES = {
    "allow_negative_values": ("Negative target values are not valid", "Negative target values are valid"),
    "known_seasonality": ("I do not expect a repeating seasonal pattern", "I expect a repeating seasonal pattern"),
    "business_days_only": ("Do not exclude weekends or non-business days", "Exclude weekends and non-business days"),
    "contains_sensitive_data": ("The source contains no personal, confidential, or regulated data", "The source contains confidential synthetic records"),
    "provenance_required": ("Detailed data lineage is not required", "Detailed data lineage is required"),
    "human_approval_required": ("Human approval before modeling is not required", "Human approval before modeling is required"),
}
QUESTION_SLOTS = ("business_goal", "success_criteria", "target_unit", "target_column", "time_column", "frequency", "forecast_horizon", "problem_statement", "target_description", "source_reference", "destination", "license", "stakeholder_role")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=lambda x: x.isoformat())


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((dumps(value) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows) -> None:
    """Unlike legacy write_text, byte writes guarantee LF on Windows too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write((dumps(row) + "\n").encode("utf-8"))


def category_for(index: int, count: int) -> str:
    # Per 140: 74 medium, 24 dense, 16 multi-turn, 18 short, 8 abstention.
    position = index * 140 // count
    if position < 74:
        return "medium"
    if position < 98:
        return "dense"
    if position < 114:
        return "correction" if position % 2 == 0 else "confirmed_conflict"
    if position < 132:
        return "selected_short"
    return "abstention"


def freeze_plan(per_domain: int = 140, version: str = VERSION) -> dict:
    if not 5 <= per_domain <= 140:
        raise ValueError("per_domain must be between 5 and 140")
    rows = []
    for domain, _, _ in DOMAINS:
        for i in range(per_domain):
            rows.append({"scenario_id": f"{version}/{domain}/{i:03d}", "domain": domain,
                         "index": i, "category": category_for(i, per_domain)})
    # Freeze IDs before any source values or utterances are generated. Health is
    # entirely final; remaining quotas are hash-selected, not cherry-picked.
    n = len(rows)
    n_train, n_validation = round(n * .70), round(n * .15)
    dev = sorted((r for r in rows if r["domain"] != "synthetic_health_ops"),
                 key=lambda r: digest(["split-seed-v5-1", r["scenario_id"]]))
    assignments = {r["scenario_id"]: ("train" if i < n_train else "validation" if i < n_train + n_validation else "final")
                   for i, r in enumerate(dev)}
    for row in rows:
        row["split"] = assignments.get(row["scenario_id"], "final")
        row["template_family"] = "heldout_handover" if row["split"] == "final" else "direct_requirements"
    return {"version": version, "schema_version": "1.0.0", "fpy_commit": FPY_COMMIT,
            "prompt_version": PROMPT_VERSION, "seed": "split-seed-v5-1", "per_domain": per_domain,
            "heldout_domain": "synthetic_health_ops", "assignments": rows,
            "status": "split_ids_frozen_before_generation_and_training"}


def source_values(domain: str, index: int, schema=None) -> dict:
    schema = schema or load_schema()
    _, target, unit = next(d for d in DOMAINS if d[0] == domain)
    # Each site is a fictional operational scope, not an appended sample ID.
    site = f"{('Alder', 'Birch', 'Cedar', 'Juniper', 'Willow')[index % 5]} facility {index + 1}"
    desc = f"{target} at {site} — simulated only"
    values = {
        "intent": "create_forecast", "problem_statement": f"Predict {target} for {site}; retain zero days",
        "business_goal": f"Plan capacity for {site} without overtime",
        "stakeholder_role": "simulation planning coordinator", "decision_supported": f"Allocate the {site} reserve",
        "success_criteria": f"MAE below {index % 11 + 2} {unit}", "target_column": f"{domain}_value",
        "target_description": desc, "target_unit": unit, "target_bounds": {"min": 0, "max": 10000 + index},
        "time_column": "observed_at", "timezone": "UTC", "forecast_start": "2026-10-01T00:00:00+00:00",
        "data_cutoff": "2026-09-30T00:00:00+00:00", "history_start": "2023-01-01T00:00:00+00:00",
        "history_end": "2026-09-30T00:00:00+00:00", "series_id_columns": ["facility_id"],
        "hierarchy_columns": ["region_id", "facility_id"], "aggregation_level": "facility",
        "scope_filters": [{"column": "facility_id", "operator": "eq", "value": site}],
        "geography": [f"Fictional {site} district"], "source_reference": f"simulated_{domain}_{index + 1}.csv",
        "source_provider": "Fictional Simulation Bureau", "sheet_or_table": "observations",
        "authentication_reference": f"secret://fictional/{domain}/reader", "minimum_training_points": 30 + index,
        "known_regime_changes": [{"date": "2025-01-01", "description": "simulated relocation"}],
        "minimum_coverage": 80 + index % 20, "seasonal_periods": [7, 28], "holidays": ["fictional_founders_day"],
        "special_events": [{"date": "2026-10-05", "name": "simulated open day"}],
        "past_covariates": ["recorded_load"], "known_future_covariates": ["planned_open_hours"],
        "static_features": ["facility_size"],
        "covariate_availability": [{"name": "planned_open_hours", "available": "before forecast creation"}],
        "external_covariate_sources": [{"name": "planned_open_hours", "source": "fictional_roster"}],
        "prediction_interval_levels": [80, 95], "quantiles": [.1, .5, .9],
        "scenario_forecasts": [{"name": "restricted", "capacity_multiplier": .8}],
        "rounding_rule": {"decimal_places": 2}, "output_granularity": "daily per facility",
        "secondary_metrics": ["rmse", "smape"], "backtest_folds": 3 + index % 5,
        "acceptable_error": {"metric": "mae", "maximum": index % 11 + 2},
        "destination": f"fictional://planning/{domain}/{index + 1}",
        "privacy_constraints": ["role-based access", "no external sharing"], "license": "fictional research-only license",
    }
    for definition in schema.slots:
        slot = definition.slot_id
        if slot in values:
            continue
        if definition.value_type == "enum":
            values[slot] = definition.allowed_values[index % len(definition.allowed_values)]
        elif definition.value_type == "boolean":
            values[slot] = bool(index % 2)
        elif definition.value_type == "duration":
            values[slot] = {"periods": 1 if slot == "frequency" else 2 + index % 25,
                            "unit": "second" if slot == "latency_requirement" else "day"}
        else:
            raise ValueError(f"missing source value rule: {slot}")
    # Coherent source/container/output combinations; no file-to-column inference.
    values["file_format"] = ("csv", "xlsx", "parquet", "json")[index % 4]
    values["source_reference"] = f"simulated_{domain}_{index + 1}.{values['file_format']}"
    values["source_mode"] = ("upload", "upload", "api", "database", "catalog")[index % 5]
    if values["source_mode"] != "upload":
        values["source_reference"] = f"fictional-{values['source_mode']}://{domain}/{index + 1}"
    values["calendar_type"] = "business_day" if values["business_days_only"] else "calendar"
    values["allow_negative_values"] = False
    values["dataset_type"] = "single_series"
    values["forecast_type"] = "point"
    return values


def atom(sid: str, turn_id: str, slot: str, value: Any, role: str = "update") -> dict:
    return {"source_fact_id": f"{sid}/{turn_id}/{slot}", "turn_id": turn_id,
            "slot_id": slot, "value": value, "status": "provided", "role": role}


def surface(value: Any) -> str:
    if isinstance(value, dict) and set(value) == {"periods", "unit"}:
        return f"{value['periods']} {value['unit']}{'' if value['periods'] == 1 else 's'}"
    return dumps(value)


def clause(fact: dict, variant: int, family: str) -> str:
    slot, value = fact["slot_id"], fact["value"]
    if slot in BOOL_CLAUSES:
        text = BOOL_CLAUSES[slot][int(value)]
    else:
        text = CLAUSES[slot].format(v=surface(value))
    if family == "heldout_handover":
        # Reserved reported-speech family; never used in development.
        return ("The handover states: ", "According to the incoming planner's brief: ",
                "The handover instruction reads: ")[variant] + text + "."
    if variant == 1:
        return "For this request: " + text[0].lower() + text[1:] + "."
    if variant == 2:
        return "Please record this requirement: " + text[0].lower() + text[1:] + "."
    return text + "."


def render_turn(facts: list[dict], *, variant: int, family: str, category: str,
                selected_slot: str | None = None, boundary: str = "unknown") -> tuple[str, list[dict]]:
    ordered = list(facts)
    if variant == 1:
        ordered.reverse()
    elif variant == 2 and ordered:
        ordered = ordered[1:] + ordered[:1]
    parts = []
    if category == "correction":
        parts.append(("Correction to my confirmed settings.", "Please replace my earlier setting.",
                      "I am explicitly changing my prior requirement.")[variant])
    if category == "prior_confirmation":
        parts.append("I confirm the following settings.")
    if category == "abstention":
        options = {
            "unknown": ("I don't know the remaining requirements.", "Those details are not known yet.", "I cannot provide those details yet."),
            "unsupported": ("Write a poem instead; do not create a forecast.", "Do not forecast; I want you to write a poem instead.", "My request is a poem, not a forecast."),
            "not_forecasting": ("I only want a definition, not a forecast.", "Explain the term only; do not forecast anything.", "No forecasting is requested; just define the term."),
            "privacy": ("I do not know the access policy; do not invent credentials.", "The access restrictions are unknown; do not fabricate credentials.", "I cannot specify the privacy policy; no credentials should be invented."),
        }
        parts.append(options[boundary][variant])
    spans = []
    for fact in ordered:
        if category == "selected_short":
            value = fact["value"]
            answer = ("yes" if value else "no") if isinstance(value, bool) else surface(value)
            text = (answer + ".", "My answer is " + answer + ".", "Please use " + answer + ".")[variant]
            if family == "heldout_handover":
                text = ("The planner's answer: ", "The handover answer is ", "The incoming planner confirms ")[variant] + answer + "."
        else:
            text = clause(fact, variant, family)
        start = len(" ".join(parts)) + bool(parts)
        parts.append(text)
        spans.append({"source_fact_id": fact["source_fact_id"], "slot_id": fact["slot_id"],
                      "start": start, "end": start + len(text), "text": text})
    if not facts and family == "heldout_handover":
        parts[0] = "The incoming planner says: " + parts[0]
    return " ".join(parts), spans


def turn_state(turn: dict, schema):
    state = state_with_context(schema, turn.get("context_slots", {}))
    # component_eval reconstructs confirmed context but leaves the intent field
    # ambiguous. Explicitly retain established intent for v5 SFT/reducer checks.
    if turn.get("context_slots", {}).get("intent") == "create_forecast":
        state.intent = Intent.CREATE_FORECAST
    return state


def make_turn(row: dict, facts: list[dict], context: dict, variant: int, category: str,
              turn_id: str, parent: str | None, selected_slot=None, boundary="unknown") -> dict:
    message, spans = render_turn(facts, variant=variant, family=row["template_family"],
                                 category=category, selected_slot=selected_slot, boundary=boundary)
    intent = boundary if category == "abstention" and boundary in {"unsupported", "not_forecasting"} else "create_forecast"
    by_id = {f["source_fact_id"]: f for f in facts}
    output = {"intent": intent, "intent_confidence": 1.0,
              "updates": [_update(s["slot_id"], by_id[s["source_fact_id"]]["value"], s["text"]) for s in spans],
              "correction_detected": category == "correction", "unsupported_claims": []}
    return {"message": message, "gold_extraction": output, "context_slots": context,
            "source_fact_ids": [f["source_fact_id"] for f in facts], "evidence_spans": spans,
            "turn_id": turn_id, "parent_id": parent, "variant": variant,
            "render_category": category, "selected_slot": selected_slot, "boundary": boundary,
            "template_family": row["template_family"], "origin": "fictional_synthetic_fact_table",
            "language": "en", "reviewer_status": REVIEW_STATUS}


def build_scenario(row: dict, version: str, schema=None) -> dict:
    schema = schema or load_schema()
    sid, i, category = row["scenario_id"], row["index"], row["category"]
    values = source_values(row["domain"], i, schema)
    context, prior_facts, selected = {}, [], None
    all_slots = [s.slot_id for s in schema.slots if s.slot_id != "intent"]
    # The pinned prompt sanitizer redacts embedded secret:// references. Do not
    # supervise a value whose exact evidence is absent from the model input.
    eligible_slots = [s for s in all_slots if s != "authentication_reference"]
    if category in {"medium", "dense"}:
        n = 3 + i % 6 if category == "medium" else 9 + i % 6
        # Two bottlenecks per ordinary case, one intrinsically site-specific.
        slots = ["target_description" if i % 2 else "source_reference", BOTTLENECKS[i % 5]]
        slots = list(dict.fromkeys(slots))
        offset = (i * 7 + next(j for j, d in enumerate(DOMAINS) if d[0] == row["domain"]) * 13) % len(eligible_slots)
        for j in range(len(eligible_slots)):
            if len(slots) >= n:
                break
            slot = eligible_slots[(offset + j) % len(eligible_slots)]
            if slot not in slots:
                slots.append(slot)
        if "seasonal_periods" in slots:
            values["known_seasonality"] = True
        if "hierarchy_columns" in slots:
            values["dataset_type"] = "hierarchical"
        elif "series_id_columns" in slots:
            values["dataset_type"] = "panel"
        if "quantiles" in slots or "prediction_interval_levels" in slots:
            values["forecast_type"] = "probabilistic"
        slots.insert(0, "intent")
    elif category in {"correction", "confirmed_conflict"}:
        prior_facts = [atom(sid, "t1", s, values[s]) for s in ("intent", "target_description", "forecast_horizon")]
        context = {f["slot_id"]: f["value"] for f in prior_facts}
        values["forecast_horizon"] = {"periods": 40 + i, "unit": "day"}
        slots = ["forecast_horizon"]
    elif category == "selected_short":
        selected = ("target_column", "time_column", "forecast_horizon", "frequency", "file_format", "source_reference", "target_unit", "contains_sensitive_data")[i % 8]
        if selected == "file_format":
            values["source_mode"] = "upload"
            values["source_reference"] = f"simulated_{row['domain']}_{i + 1}.{values['file_format']}"
        context = {"intent": values["intent"], "target_description": values["target_description"]}
        if selected != "source_reference":
            context["source_reference"] = values["source_reference"]
        # Only supply prior explicitly confirmed facts needed to make the
        # actual fpy selector choose this short-answer slot, not a fake hint.
        for _ in schema.slots:
            choice = select_next_slot(schema, state_with_context(schema, context))
            if choice is not None and choice.slot_id == selected:
                break
            if choice is None:
                raise ValueError("selected short answer cannot be reached")
            context[choice.slot_id] = values[choice.slot_id]
        else:
            raise ValueError("selected short answer did not converge")
        slots = [selected]
    else:
        slots = []
        context = {"intent": "create_forecast", "target_description": values["target_description"]}
    main_id = "t2" if prior_facts else "t1"
    main_facts = [atom(sid, main_id, s, values[s]) for s in slots]
    context_facts = [atom(sid, "context", s, v, "confirmed_context") for s, v in context.items()]
    turns = []
    if prior_facts:
        turns.append(make_turn(row, prior_facts, {}, 0, "prior_confirmation", "t1-v0", None))
        # A fixture confirmation event is explicit and separately represented.
        turns[-1]["confirmation_event"] = "user confirms the stated t1 values before t2"
    for variant in range(3):
        turns.append(make_turn(row, main_facts, context, variant, category, f"{main_id}-v{variant}",
                               f"{sid}/t1-v0" if prior_facts else sid, selected,
                               ("unknown", "unsupported", "not_forecasting", "privacy")[i % 4]))
    last = turns[-1]
    final_state = apply_extraction(turn_state(last, schema), ExtractorResult.model_validate(last["gold_extraction"]), schema, 2 if prior_facts else 1, last["message"])
    gold = {s: json.loads(dumps(v.value)) for s, v in final_state.slots.items() if v.value is not None}
    if category == "confirmed_conflict":
        question_slot = "forecast_horizon"
    elif last["gold_extraction"]["intent"] != "create_forecast":
        question_slot = "intent"
    else:
        question_slot = next(s for s in QUESTION_SLOTS if s not in gold)
    mentioned = {f["slot_id"] for f in main_facts + prior_facts}
    return {"scenario_id": sid, "source_fact_id": f"{sid}/fact-table", "corpus_version": version,
            "schema_version": schema.version, "split": row["split"], "domain": row["domain"],
            "category": category, "template_family": row["template_family"], "language": "en",
            "origin": "fictional_synthetic_fact_table", "reviewer_status": REVIEW_STATUS,
            "parent_id": None, "turns": turns, "source_facts": prior_facts + main_facts + context_facts,
            "gold_intent": last["gold_extraction"]["intent"], "gold_final_slots": gold,
            "gold_final_statuses": {s: v.status.value for s, v in final_state.slots.items() if v.value is not None},
            "expected_question_slot": question_slot, "ideal_question": schema.get(question_slot).static_question,
            "must_not_infer": sorted(set(all_slots) - mentioned),
            "behavior_tags": [category, "literal_evidence", "no_invention", "synthetic_only"],
            "question_context_confirmation_event": "user confirms nonconflicting supplied values before the question fixture",
            "context_provenance": "explicit t1 confirmation" if prior_facts else "fictional previously confirmed facts"}


def canonical_check(slot_id: str, value: Any, schema) -> None:
    definition = schema.get(slot_id)
    kind = definition.value_type
    number = lambda x: type(x) in (int, float) and math.isfinite(x)
    valid = False
    if kind in {"string", "enum", "timezone", "datetime", "secret_reference"}:
        valid = isinstance(value, str) and bool(value.strip())
    elif kind == "boolean":
        valid = type(value) is bool
    elif kind == "integer":
        valid = type(value) is int and value > 0
    elif kind == "percentage":
        valid = number(value) and 0 <= value <= 100
    elif kind == "duration":
        valid = isinstance(value, dict) and set(value) == {"periods", "unit"} and number(value["periods"]) and value["periods"] > 0 and value["unit"] in DURATION_UNITS
    elif kind == "object":
        valid = isinstance(value, dict) and bool(value)
    elif kind.endswith("_list"):
        valid = isinstance(value, list) and bool(value)
        if valid:
            subtype = kind[:-5]
            checks = {"object": lambda x: isinstance(x, dict) and bool(x),
                      "string": lambda x: isinstance(x, str) and bool(x.strip()),
                      "enum": lambda x: isinstance(x, str) and x in {"mae", "rmse", "mase", "smape"},
                      "integer": lambda x: type(x) is int and x > 0,
                      "percentage": lambda x: number(x) and 0 <= x <= 100,
                      "probability": lambda x: number(x) and 0 <= x <= 1}
            valid = all(checks[subtype](item) for item in value)
    if not valid:
        raise ValueError(f"noncanonical type or range for {slot_id}")
    if definition.allowed_values and kind == "enum" and value not in definition.allowed_values:
        raise ValueError(f"unsupported enum for {slot_id}")
    issues = validate_slot(definition, SlotState(slot_id=slot_id, value=value, status="provided", evidence_text="source fact"))
    if issues:
        raise ValueError(f"invalid canonical value for {slot_id}")


def validate_scenario(scenario: dict, schema=None) -> None:
    """Validate without printing source facts or labels (also used for final)."""
    schema = schema or load_schema()
    sid = scenario["scenario_id"]
    if scenario["schema_version"] != schema.version or scenario["source_fact_id"] != f"{sid}/fact-table":
        raise ValueError("schema or source identity mismatch")
    turn_ids = [t["turn_id"] for t in scenario["turns"]]
    if len(set(turn_ids)) != len(turn_ids):
        raise ValueError("duplicate turn identity")
    for turn in scenario["turns"]:
        if turn["parent_id"] not in {None, sid, f"{sid}/t1-v0"}:
            raise ValueError("turn parent crosses source group")
        if any(turn[k] != scenario[k] for k in ("template_family", "origin", "language", "reviewer_status")):
            raise ValueError("turn provenance differs from source")
    facts = {f["source_fact_id"]: f for f in scenario["source_facts"]}
    if any(not fid.startswith(sid + "/") for fid in facts):
        raise ValueError("source fact belongs to another scenario")
    if len(facts) != len(scenario["source_facts"]):
        raise ValueError("duplicate source fact ID")
    for fact in facts.values():
        canonical_check(fact["slot_id"], fact["value"], schema)
        if fact["status"] != "provided":
            raise ValueError("fact status is not explicit provided")
    main_atoms = []
    for turn in scenario["turns"]:
        selected_facts = [facts[fid] for fid in turn["source_fact_ids"]]
        context = {f["slot_id"]: f["value"] for f in facts.values() if f["role"] == "confirmed_context"}
        if turn["render_category"] == "prior_confirmation":
            context = {}
        if turn["context_slots"] != context:
            raise ValueError("context/fact mismatch")
        expected_message, expected_spans = render_turn(selected_facts, variant=turn["variant"], family=scenario["template_family"],
            category=turn["render_category"], selected_slot=turn["selected_slot"], boundary=turn["boundary"])
        if turn["message"] != expected_message or turn["evidence_spans"] != expected_spans:
            raise ValueError("evidence/text/atom mismatch (including polarity or numbers)")
        result = ExtractorResult.model_validate(turn["gold_extraction"])
        if set(turn["gold_extraction"]) != {"intent", "intent_confidence", "updates", "correction_detected", "unsupported_claims"}:
            raise ValueError("unexpected extraction fields")
        if result.unsupported_claims:
            raise ValueError("unbacked unsupported claims")
        expected_intent = turn["boundary"] if turn["render_category"] == "abstention" and turn["boundary"] in {"unsupported", "not_forecasting"} else "create_forecast"
        if result.intent.value != expected_intent or result.correction_detected != (turn["render_category"] == "correction"):
            raise ValueError("intent/correction mismatch")
        expected = {f["slot_id"]: f["value"] for f in selected_facts}
        actual = {u.slot_id: u.candidate_value for u in result.updates}
        if len(actual) != len(result.updates) or dumps(actual) != dumps(expected):
            raise ValueError("fact-label equivalence violated")
        by_slot = {span["slot_id"]: span for span in turn["evidence_spans"]}
        for raw, update in zip(turn["gold_extraction"]["updates"], result.updates):
            if set(raw) != {"slot_id", "candidate_value", "status", "confidence", "evidence_text"} or not isinstance(raw["candidate_value"], str):
                raise ValueError("noncanonical extraction serialization")
            if update.status.value != "provided":
                raise ValueError("absence/unknown must not be provided")
            canonical_check(update.slot_id, update.candidate_value, schema)
            span = by_slot[update.slot_id]
            if not update.evidence_text or turn["message"][span["start"]:span["end"]] != update.evidence_text or span["text"] != update.evidence_text:
                raise ValueError("invalid evidence span")
        state = turn_state(turn, schema)
        before = state.model_dump()
        if turn["selected_slot"]:
            choice = select_next_slot(schema, state)
            if choice is None or choice.slot_id != turn["selected_slot"]:
                raise ValueError("short answer not for actual selected slot")
        reduced = apply_extraction(state, result, schema, 1, turn["message"])
        if state.model_dump() != before:
            raise ValueError("reducer mutated input state")
        for slot, original in state.slots.items():
            new = reduced.slots[slot]
            if slot not in actual and slot != "intent" and new != original:
                raise ValueError("unmentioned slot changed")
            if slot in actual:
                normal = normalize_value(schema.get(slot), actual[slot])
                conflict = original.confirmed_by_user and original.value != normal and not result.correction_detected
                if conflict:
                    if new.status.value != "conflicting" or new.value != original.value or not new.confirmed_by_user:
                        raise ValueError("confirmed conflict overwrote prior value")
                elif new.value != normal or new.status.value not in {"provided", "confirmed"}:
                    raise ValueError("provided/correction reducer invariant failed")
        payload = json.loads(build_v5_extractor_input(turn["message"], state, schema))
        if payload["current_message"] != turn["message"]:
            raise ValueError("prompt altered exact source wording")
        if [d["slot_id"] for d in payload["slot_definitions"]] != [s.slot_id for s in schema.slots]:
            raise ValueError("prompt omitted schema slots")
        for sent, definition in zip(payload["slot_definitions"], schema.slots):
            if sent["value_type"] != definition.value_type or sent.get("allowed_values", []) != list(definition.allowed_values):
                raise ValueError("schema definition mismatch")
        if turn["render_category"] != "prior_confirmation":
            main_atoms.append(dumps(expected))
    if len(main_atoms) != 3 or len(set(main_atoms)) != 1:
        raise ValueError("variants must preserve the same atomic facts")
    prior = [f for f in facts.values() if f["turn_id"] == "t1" and f["role"] == "update"]
    if scenario["category"] in {"correction", "confirmed_conflict"}:
        expected_context = {f["slot_id"]: f["value"] for f in prior}
        if scenario["turns"][-1]["context_slots"] != expected_context or not scenario["turns"][0].get("confirmation_event"):
            raise ValueError("cross-turn confirmation provenance mismatch")
    gold = {s: json.loads(dumps(v.value)) for s, v in reduced.slots.items() if v.value is not None}
    statuses = {s: v.status.value for s, v in reduced.slots.items() if v.value is not None}
    if dumps(gold) != dumps(scenario["gold_final_slots"]) or statuses != scenario["gold_final_statuses"]:
        raise ValueError("gold final state does not match reducer")
    if scenario["gold_intent"] != result.intent.value:
        raise ValueError("gold final intent mismatch")
    updated_slots = {u["slot_id"] for t in scenario["turns"] for u in t["gold_extraction"]["updates"]}
    if updated_slots & set(scenario["must_not_infer"]):
        raise ValueError("forbidden slot is labeled")
    request = question_request(scenario, schema)
    if any(statuses.get(slot) == "conflicting" for slot in request.confirmed_context):
        raise ValueError("conflicting value leaked into confirmed question context")
    if not scenario.get("question_context_confirmation_event"):
        raise ValueError("missing question context confirmation provenance")
    if not validate_question(QuestionOutput(question=scenario["ideal_question"]), request):
        raise ValueError("question contract failed")
    if scenario["reviewer_status"] != REVIEW_STATUS:
        raise ValueError("synthetic generator cannot claim human acceptance")


def sft_examples(scenario: dict, schema=None):
    schema = schema or load_schema()
    common = {key: scenario[key] for key in ("scenario_id", "source_fact_id", "split", "category", "domain", "origin", "language", "template_family", "reviewer_status")}
    for turn in scenario["turns"]:
        yield {"messages": [
            {"role": "system", "content": build_v5_extractor_instructions()},
            {"role": "user", "content": build_v5_extractor_input(turn["message"], turn_state(turn, schema), schema)},
            {"role": "assistant", "content": dumps(turn["gold_extraction"])}],
            "metadata": {**common, "task": "extract", "turn_id": turn["turn_id"], "parent_id": turn["parent_id"],
                         "source_fact_ids": turn["source_fact_ids"], "variant": turn["variant"]}}
    request = question_request(scenario, schema)
    yield {"messages": [
        {"role": "system", "content": build_slm_question_instructions()},
        {"role": "user", "content": build_slm_question_input(request)},
        {"role": "assistant", "content": dumps({"question": scenario["ideal_question"]})}],
        "metadata": {**common, "task": "ask", "turn_id": "question", "parent_id": scenario["scenario_id"],
                     "selected_slot": request.slot_id}}


def normalized_input(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def validate_collection(scenarios: list[dict], plan: dict, schema=None) -> dict:
    schema = schema or load_schema()
    assignments = {r["scenario_id"]: r for r in plan["assignments"]}
    seen, exact, normalized, utterances = set(), {}, {}, Counter()
    for scenario in scenarios:
        sid = scenario["scenario_id"]
        if sid in seen:
            raise ValueError("duplicate scenario or split leakage")
        seen.add(sid)
        row = assignments.get(sid)
        if row is None or any(row[k] != scenario[k] for k in ("split", "domain", "category", "template_family")):
            raise ValueError("frozen split/domain/template assignment changed")
        if scenario["domain"] == "synthetic_health_ops" and scenario["split"] != "final":
            raise ValueError("heldout domain leaked")
        validate_scenario(scenario, schema)
        for example in sft_examples(scenario, schema):
            text = dumps(example["messages"][:2])
            for table, key in ((exact, digest(text)), (normalized, digest(normalized_input(text)))):
                if key in table:
                    raise ValueError("duplicate exact/normalized model input")
                table[key] = sid
        utterances.update(normalized_input(t["message"]) for t in scenario["turns"])
    return {"exact_model_input_duplicates": 0, "normalized_model_input_duplicates": 0,
            "normalized_utterance_repetitions": sum(n - 1 for n in utterances.values()),
            "utterance_policy": "Repeated short replies are retained only with distinct confirmed context; dedup uses full system+user input.",
            "checked_examples": len(exact)}


def overlap_report(scenarios: list[dict]) -> tuple[dict, list[dict]]:
    """Exhaustive cross-split representative Jaccard, not an ID-leakage proxy.

    One main-turn representative per source keeps this audit tractable. Template
    tokens mask evidence spans (including their clause wrappers), revealing the
    shared assembly family; this intentionally exposes extensive similarity.
    """
    representatives = []
    for s in scenarios:
        t = next(t for t in s["turns"] if t["render_category"] != "prior_confirmation")
        text = t["message"]
        template = text
        for span in sorted(t["evidence_spans"], key=lambda x: x["start"], reverse=True):
            template = template[:span["start"]] + " FACT_" + span["slot_id"] + " " + template[span["end"]:]
        representatives.append((s, set(normalized_input(text).split()), set(normalized_input(template).split())))
    bins, template_bins, flagged = Counter(), Counter(), []
    comparisons = 0
    for i, (left, words, template) in enumerate(representatives):
        for right, other, other_template in representatives[i + 1:]:
            if left["split"] == right["split"]:
                continue
            score = len(words & other) / max(1, len(words | other))
            structural = len(template & other_template) / max(1, len(template | other_template))
            comparisons += 1
            bins[str(int(score * 10) / 10)] += 1
            template_bins[str(int(structural * 10) / 10)] += 1
            if score >= .72 or structural >= .9:
                flagged.append({"left_id": left["scenario_id"], "right_id": right["scenario_id"],
                                "text_jaccard": round(score, 4), "template_jaccard": round(structural, 4),
                                "reviewer_status": "pending_human", "reason": "cross_split_template_or_lexical_similarity"})
    return {"method": "exhaustive token-set Jaccard on first main variant per source; all split pairs",
            "template_method": "replace evidence clauses by slot IDs; measures shared fact assembly, not paraphrase independence",
            "comparisons": comparisons, "text_histogram_floor_tenths": dict(bins),
            "template_histogram_floor_tenths": dict(template_bins), "thresholds": {"text": .72, "template": .9},
            "flagged_pairs": len(flagged), "reviewed_pairs": 0, "pending_pairs": len(flagged),
            "limitation": "Other variants, semantic similarity and external corpora are not exhaustively compared. Shared clauses and prompt schema remain; zero leakage is NOT claimed."}, flagged


def statistics_report(scenarios: list[dict], schema=None) -> dict:
    schema = schema or load_schema()
    hist, categories, split_examples, positives, negatives, lengths, boolean_values = Counter(), Counter(), Counter(), Counter(), Counter(), [], Counter()
    enum_values, prompt_lengths, completion_lengths = defaultdict(Counter), [], []
    by_split = {split: {"positives": Counter(), "negative_opportunities": Counter()} for split in ("train", "validation", "final")}
    for s in scenarios:
        for example in sft_examples(s, schema):
            prompt_lengths.append(sum(len(m["content"]) for m in example["messages"][:2]))
            completion_lengths.append(len(example["messages"][-1]["content"]))
        split_examples[s["split"]] += len(s["turns"]) + 1
        for t in s["turns"]:
            slots = {u["slot_id"] for u in t["gold_extraction"]["updates"]}
            hist[len(slots - {"intent"})] += 1
            categories[s["category"]] += 1
            positives.update(slots)
            by_split[s["split"]]["positives"].update(slots)
            omitted = {d.slot_id for d in schema.slots} - slots
            negatives.update(omitted)
            by_split[s["split"]]["negative_opportunities"].update(omitted)
            lengths.append(len(t["message"]))
            for u in t["gold_extraction"]["updates"]:
                if schema.get(u["slot_id"]).value_type == "boolean":
                    boolean_values[f"{u['slot_id']}:{json.loads(u['candidate_value'])}"] += 1
                elif schema.get(u["slot_id"]).value_type == "enum":
                    enum_values[u["slot_id"]][json.loads(u["candidate_value"])] += 1
    total = sum(hist.values())
    def length_stats(items):
        ordered = sorted(items)
        return {"min": min(items), "max": max(items), "mean": sum(items) / len(items),
                "p50": ordered[len(ordered) // 2], "p95": ordered[math.ceil(len(ordered) * .95) - 1]}
    return {"source_scenarios": len(scenarios), "split_scenarios": dict(Counter(s["split"] for s in scenarios)),
            "domain_scenarios": dict(Counter(s["domain"] for s in scenarios)), "sft_examples": dict(split_examples),
            "total_examples": sum(split_examples.values()), "extraction_examples": total, "question_examples": len(scenarios),
            "nonintent_update_histogram": dict(sorted(hist.items())),
            "three_to_eight_update_fraction": sum(n for k, n in hist.items() if 3 <= k <= 8) / total,
            "nine_plus_update_fraction": sum(n for k, n in hist.items() if k >= 9) / total,
            "extraction_category_counts": dict(categories), "category_denominator": "extraction examples, including prior correction/conflict turns; questions reported separately",
            "per_slot": {s.slot_id: {"positives": positives[s.slot_id], "negative_opportunities": negatives[s.slot_id]} for s in schema.slots},
            "per_split_slot_counts": by_split, "negative_opportunity_definition": "slot absent from expected current-turn updates, NOT a labeled false value",
            "boolean_value_counts": dict(boolean_values), "enum_value_counts": dict(enum_values),
            "message_characters": length_stats(lengths), "sft_prompt_characters": length_stats(prompt_lengths),
            "sft_completion_characters": length_stats(completion_lengths),
            "all_required_slots_positive": all(positives[s.slot_id] for s in schema.slots if s.requiredness.value == "required"),
            "positive_slot_count": sum(bool(positives[s.slot_id]) for s in schema.slots),
            "all_schema_slots_positive": all(positives[s.slot_id] for s in schema.slots),
            "generation_exclusions": {"authentication_reference": "embedded secret URI redacted by pinned prompt; no ungrounded positive labels"}}


def review_decisions(scenarios: list[dict]) -> list[dict]:
    strata = defaultdict(list)
    for s in scenarios:
        if s["split"] == "train":
            strata[(s["domain"], s["category"])].append(s["scenario_id"])
    sample = {min(ids, key=lambda sid: digest(["review-sample", sid])) for ids in strata.values()}
    decisions = []
    for s in scenarios:
        reasons = []
        if s["split"] == "final":
            reasons.append("all_final_cases_require_human_review_before_evaluation")
        if s["category"] in {"correction", "confirmed_conflict", "abstention", "selected_short"}:
            reasons.append("uncertain_or_context_dependent_example")
        if s["scenario_id"] in sample:
            reasons.append("stratified_train_domain_category_sample")
        decisions.append({"scenario_id": s["scenario_id"], "split": s["split"], "reviewer_status": REVIEW_STATUS,
                          "machine_decision": "accepted_for_pending_human_pool",
                          "acceptance_reasons": ["canonical_types_and_enums", "literal_evidence_and_offsets", "atom_label_equivalence", "reducer_invariants", "question_contract", "all_schema_prompt", "frozen_split_and_input_dedup"],
                          "rejection_reasons": [], "human_review_required": bool(reasons), "human_review_reasons": reasons})
    return decisions


def file_record(path: Path) -> dict:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def dependency_record() -> dict:
    commit = subprocess.check_output(["git", "-C", str(FPY_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if commit != FPY_COMMIT:
        raise ValueError("fpy is not at the pinned commit")
    files = ["domain/models.py", "domain/schema.py", "application/state_reducer.py", "application/normalization.py", "application/validation.py"]
    records = {}
    for name in files:
        path = FPY_ROOT / "src/forecasting_assistant" / name
        committed = subprocess.check_output(["git", "-C", str(FPY_ROOT), "show", f"HEAD:src/forecasting_assistant/{name}"])
        if path.read_bytes().replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
            raise ValueError("pinned dependency source has local modifications")
        records[name] = file_record(path)
    return {"commit": commit, "files": records}


def verify_artifacts(output: Path, sealed_output: Path) -> dict:
    """Never parses sealed labels: only streams their bytes for SHA256."""
    output, sealed_output = output.resolve(), sealed_output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    manifest_hash = json.loads((output / "manifest.sha256.json").read_bytes())["manifest.json"]
    if file_record(output / "manifest.json") != manifest_hash:
        raise ValueError("frozen manifest hash/size mismatch")
    if manifest["build_status"] != "complete_pending_human":
        raise ValueError("incomplete corpus")
    if {p.relative_to(sealed_output).as_posix() for p in sealed_output.rglob("*") if p.is_file()} != set(manifest["sealed_files"]):
        raise ValueError("unexpected sealed artifact inventory")
    expected_public = set(manifest["files"]) | {"manifest.json", "manifest.sha256.json"}
    if {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} != expected_public:
        raise ValueError("unexpected public artifact inventory")
    for root, entries in ((output, manifest["files"]), (sealed_output, manifest["sealed_files"])):
        for name, expected in entries.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or file_record(path) != expected:
                raise ValueError("artifact hash/size mismatch")
    plan = json.loads((output / "split-freeze.json").read_bytes())
    scenarios = []
    for split in ("train", "validation"):
        scenarios.extend(json.loads(line) for line in (output / "splits" / f"{split}.jsonl").read_bytes().splitlines())
    dedup = validate_collection(scenarios, plan)
    expected_ids = {r["scenario_id"] for r in plan["assignments"] if r["split"] in {"train", "validation"}}
    if {s["scenario_id"] for s in scenarios} != expected_ids:
        raise ValueError("development records do not cover their frozen IDs")
    for split in ("train", "validation"):
        expected = (dumps(e).encode("utf-8") for s in scenarios if s["split"] == split for e in sft_examples(s))
        actual = (output / "sft" / f"{split}.jsonl").read_bytes().splitlines()
        if list(expected) != actual:
            raise ValueError("development SFT differs from validated component records")
    validation_ids = {s["scenario_id"] for s in scenarios if s["split"] == "validation"}
    smoke = [json.loads(line) for line in (output / "splits/smoke.jsonl").read_bytes().splitlines()]
    if not {s["scenario_id"] for s in smoke} <= validation_ids:
        raise ValueError("smoke is not a validation subset")
    return {"status": "verified_development_and_all_file_hashes", "sealed_labels_parsed": False,
            "development_scenarios": len(scenarios), "dedup": dedup}


README = """# v5 synthetic forecasting corpus

All source facts are fictional simulations, including health operations; no real patient,
person, institution or external API data is used. v1-v3 are excluded pending source,
label and leakage human audit; untrusted v4 is excluded. No old biased examples are mixed in.

## Version and schema contract

See schema.json (all 79 definitions), dependency.json (pinned fpy source hashes),
prompt-manifest.json and split-freeze.json. The latter is written before generation
or training. Candidate values are JSON-encoded text with strict canonical types;
free text keeps exact spelling, punctuation and case. Durations use periods/unit.
Date labels are ISO strings; the reducer normalizes them to datetime internally.
Evidence offsets are Python Unicode character offsets, half-open [start,end), not bytes.
All JSONL files are UTF-8 with LF bytes. Component records use corpus.py/component_eval.py
scenario/turn/gold_extraction shape; source_facts, context and evidence_spans add provenance.
SFT uses messages (system/user/assistant), v5 all-schema extractor prompts and existing
compact question prompts. Each source has three fact-preserving main-turn renderings
and one question; correction/conflict sources additionally contain a prior turn.
Variants are alternative renderings, NOT consecutive user utterances. Their contexts
are explicit fixtures. Prior confirmation is a separately declared user event.
component_eval reconstructs context values but does not restore established state.intent;
v5 SFT restores it. Selected short answers rely on the actual reducer/selector context.
Questions follow the existing question_request contract, which treats supplied context as
confirmed; their provenance is synthetic and still needs human review.

## Splits and procedural seal

All source turns/variants/questions stay in the same frozen split. The entire synthetic
health-operations domain and the reported-speech handover template family are final only.
Other domains cross splits at source level. Shared clauses/fact assembly remain across
splits: this is not a zero-leakage or linguistically independent benchmark. See the
measured representative fuzzy overlap report and pending pair queue.

splits/train.jsonl and splits/validation.jsonl are development component cases.
splits/smoke.jsonl is a deterministic SUBSET of validation, never an independent split.
sft/train.jsonl and sft/validation.jsonl are the only public training-format examples.
Final component cases, labels, facts and SFT are ONLY in the separately ignored sealed
root. Public manifests contain aggregate final statistics and hashes, not final labels.
This is procedural sealing, not encryption, cryptographic access control, independent
human testing, or a license to tune against final results. Do not open final labels for
training/debugging; verification hashes their bytes without parsing them.

## Review gate

Every case is machine_validated_pending_human; none is accepted_human. Read
review-decisions.jsonl for every machine acceptance reason and human-review requirement.
review-queue.jsonl lists ALL final IDs, uncertain/context-dependent cases and at least
one train case per domain/category stratum. Review every variant and prior context,
not only the first utterance: check naturalness, canonical semantics, polarity, exact
quoted wording, evidence offsets, non-invention, corrections, and question relevance.
All flagged cross-split pairs remain pending; reviewer must record accept/reject with
identity, time and substantive reasons in a separately versioned review ledger. A
final reviewer must not communicate final labels or case-level feedback to the trainer.
Do not claim the review gate passed until all required cases and pairs are resolved.
Rejected or revised frozen examples require a NEW corpus revision, never overwrite.

## Measured limitations

Category ratios refer to extraction examples, not question examples. Statistics report
actual update histograms, per-slot positives and absence opportunities, lengths and
counts. Absence opportunities are not factual false labels. No ambiguous/inferred or
dont_care labels are synthesized. authentication_reference has NO positive examples:
the pinned sanitizer redacts embedded secret:// text, so evidence would not survive
in the required prompt. Its schema definition and negative opportunities remain;
this deliberate coverage gap is preferable to ungrounded labels or bypassing redaction.
Complex object values are explicitly written as JSON inside natural clauses.
Some enum diversity is restricted to keep simulated facts
coherent. Deterministic English-only templates are not human paraphrases. Prompt-schema
length, boilerplate similarity and context-conditioned short reply repetition remain.
Fuzzy audit compares one representative per source, not all pairwise variants or old
corpora. No training, live model evaluation, network calls or human review occurs here.
"""


def write_artifacts(output: Path, sealed_output: Path, *, per_domain: int = 140, version: str = VERSION,
                    check_dependency: bool = True) -> dict:
    output, sealed_output = output.resolve(), sealed_output.resolve()
    if output.exists() or sealed_output.exists():
        raise FileExistsError("refusing overwrite: use a new version, including after a failed build")
    if output == sealed_output or output.is_relative_to(sealed_output) or sealed_output.is_relative_to(output):
        raise ValueError("public and sealed roots must be separate")
    if check_dependency:
        permitted = (PROJECT_ROOT / "training-runs/v5-sealed").resolve()
        if not sealed_output.is_relative_to(permitted):
            raise ValueError("official sealed output must be under ignored training-runs/v5-sealed")
        ignored = subprocess.run(["git", "-C", str(PROJECT_ROOT), "check-ignore", "--quiet", str(sealed_output / "splits/final.jsonl")])
        if ignored.returncode:
            raise ValueError("sealed output is not git-ignored")
    dependency = dependency_record() if check_dependency else {"commit": FPY_COMMIT, "miniature_test_fixture": True}
    schema = load_schema()
    plan = freeze_plan(per_domain, version)
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / "split-freeze.json", plan)
        sealed_output.mkdir(parents=True, exist_ok=False)
        write_json(output / "build-status.json", {"status": "building", "version": version})
        scenarios = [build_scenario(row, version, schema) for row in plan["assignments"]]
        dedup = validate_collection(scenarios, plan, schema)
        stats = statistics_report(scenarios, schema)
        overlap, pairs = overlap_report(scenarios)
        decisions = review_decisions(scenarios)
        for split in ("train", "validation", "final"):
            root = sealed_output if split == "final" else output
            subset = [s for s in scenarios if s["split"] == split]
            write_jsonl(root / "splits" / f"{split}.jsonl", subset)
            write_jsonl(root / "sft" / f"{split}.jsonl", (e for s in subset for e in sft_examples(s, schema)))
        validation = [s for s in scenarios if s["split"] == "validation"]
        smoke = sorted(validation, key=lambda s: digest(["smoke", s["scenario_id"]]))[:min(24, len(validation))]
        write_jsonl(output / "splits/smoke.jsonl", smoke)
        write_json(output / "schema.json", schema.model_dump(mode="json"))
        write_json(output / "dependency.json", dependency)
        write_json(output / "prompt-manifest.json", {"version": PROMPT_VERSION,
                   "extractor_instructions_sha256": hashlib.sha256(build_v5_extractor_instructions().encode("utf-8")).hexdigest(),
                   "source_files": {name: file_record(PROJECT_ROOT / name) for name in ("local_slm_lab/v5_prompts.py", "local_slm_lab/slm_prompts.py", "local_slm_lab/corpus_v5.py")}})
        write_json(output / "statistics.json", stats)
        write_json(output / "overlap-summary.json", overlap)
        write_jsonl(output / "overlap-review-queue.jsonl", pairs)
        write_jsonl(output / "review-decisions.jsonl", decisions)
        write_jsonl(output / "review-queue.jsonl", (d for d in decisions if d["human_review_required"]))
        (output / "README.md").write_bytes(README.encode("utf-8"))
        write_json(output / "build-status.json", {"status": "complete_pending_human", "version": version})
        manifest = {"corpus_version": version, "schema_version": schema.version, "prompt_version": PROMPT_VERSION,
                    "fpy_commit": FPY_COMMIT, "build_status": "complete_pending_human", "reviewer_status": REVIEW_STATUS,
                    "statistics": stats, "dedup": dedup, "fuzzy_overlap": overlap,
                    "smoke": {"count": len(smoke), "subset_of": "validation"},
                    "review": {"machine_accepted": len(decisions), "machine_rejected": 0, "accepted_human": 0,
                               "human_required": sum(d["human_review_required"] for d in decisions),
                               "acceptance_reason_counts": dict(Counter(r for d in decisions for r in d["acceptance_reasons"])),
                               "rejection_reason_counts": {}, "pending_overlap_pairs": len(pairs)},
                    "legacy_sources": {"v1": "excluded_pending_label_leakage_human_audit", "v2": "excluded_pending_label_leakage_human_audit", "v3": "excluded_pending_label_leakage_human_audit", "v4": "excluded_untrusted"},
                    "seal": "procedural_only_not_cryptographic_or_human_independent", "files": {}, "sealed_files": {}}
        for root, key in ((output, "files"), (sealed_output, "sealed_files")):
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    manifest[key][path.relative_to(root).as_posix()] = file_record(path)
        write_json(output / "manifest.json", manifest)
        # A separately hashed frozen manifest makes post-build changes detectable.
        write_json(output / "manifest.sha256.json", {"manifest.json": file_record(output / "manifest.json")})
        return manifest
    except Exception:
        # Do not disclose per-case final facts in error messages or debug files.
        write_json(output / "build-status.json", {"status": "failed_do_not_reuse_version", "version": version,
                   "reason": "generation_or_validation_failed; diagnose using miniature fixtures and build a new revision"})
        raise RuntimeError("corpus build failed; version marked failed, do not overwrite") from None
