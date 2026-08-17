"""Build corpus v2 with compact prompts and conversational intent transitions."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
if str(FPY_SRC) not in sys.path:
    sys.path.insert(0, str(FPY_SRC))

from forecasting_assistant.application.clarification import select_next_slot  # noqa: E402
from forecasting_assistant.application.state_reducer import apply_extraction  # noqa: E402
from forecasting_assistant.domain.conditions import is_slot_active  # noqa: E402
from forecasting_assistant.domain.models import (  # noqa: E402
    DialogueState,
    DialogueTurn,
    ExtractorResult,
    QuestionRequest,
    SlotState,
    SlotStatus,
)
from forecasting_assistant.domain.schema import (  # noqa: E402
    ForecastingSchema,
    create_initial_state,
    load_schema,
)
from local_slm_lab.corpus import DOMAINS, build_corpus, split_for_domain  # noqa: E402
from local_slm_lab.slm_prompts import (  # noqa: E402
    build_slm_extractor_input,
    build_slm_extractor_instructions,
    build_slm_question_input,
    build_slm_question_instructions,
)


CORPUS_VERSION = "forecasting-llmrei-320-v2"
SCHEMA_VERSION = "1.0.0"
INTENT_QUESTION = "Do you want the system to create a time-series forecast?"


def _update(slot_id: str, value: Any, message: str) -> dict[str, Any]:
    return {
        "slot_id": slot_id,
        "candidate_value": json.dumps(value, ensure_ascii=False, sort_keys=True),
        "status": "provided",
        "confidence": 1.0,
        "evidence_text": message,
    }


def _gold(message: str, intent: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": intent,
        "intent_confidence": 1.0,
        "updates": [_update(slot_id, value, message) for slot_id, value in values.items()],
        "correction_detected": False,
        "unsupported_claims": [],
    }


def _state_for_turn(
    schema: ForecastingSchema, turn: dict[str, Any]
) -> DialogueState:
    state = create_initial_state(schema)
    for slot_id, value in turn.get("context_slots", {}).items():
        state.slots[slot_id] = SlotState(
            slot_id=slot_id,
            value=value,
            status=SlotStatus.CONFIRMED,
            confidence=1.0,
            evidence_text="synthetic reviewed context",
            confirmed_by_user=True,
        )
    previous = turn.get("last_assistant_question")
    if previous:
        state.turns.append(
            DialogueTurn(
                turn_number=1,
                user_message="[earlier turn]",
                assistant_message=previous,
            )
        )
    return state


def _post_extraction_state(
    schema: ForecastingSchema, turn: dict[str, Any]
) -> DialogueState:
    state = _state_for_turn(schema, turn)
    result = ExtractorResult.model_validate(turn["gold_extraction"])
    updated = apply_extraction(
        state,
        result,
        schema,
        len(state.turns) + 1,
        turn["message"],
    )
    updated.intent = result.intent
    return updated


def _conversation_scenario(
    pattern: str,
    domain_index: int,
    schema: ForecastingSchema,
) -> dict[str, Any]:
    domain = DOMAINS[domain_index]
    split = split_for_domain(domain_index)
    prior_question: str | None = None
    forbidden = ["target_column", "source_reference", "forecast_horizon"]

    if pattern == "yes":
        message = "yes"
        prior_question = INTENT_QUESTION
        values = {"intent": "create_forecast"}
        intent = "create_forecast"
        forbidden.append("problem_statement")
    elif pattern == "explicit_request":
        message = f"I want to forecast {domain.target_description}."
        values = {
            "intent": "create_forecast",
            "problem_statement": f"forecast {domain.target_description}",
            "target_description": domain.target_description,
        }
        intent = "create_forecast"
    elif pattern == "please_predict":
        message = f"Please predict {domain.target_description}."
        values = {
            "intent": "create_forecast",
            "problem_statement": f"forecast {domain.target_description}",
            "target_description": domain.target_description,
        }
        intent = "create_forecast"
    elif pattern == "roman_urdu":
        message = f"Mujhe {domain.target_description} ka forecast banana hai."
        values = {
            "intent": "create_forecast",
            "problem_statement": f"forecast {domain.target_description}",
            "target_description": domain.target_description,
        }
        intent = "create_forecast"
    elif pattern == "roman_yes":
        message = "haan, forecast banana hai"
        prior_question = INTENT_QUESTION
        values = {"intent": "create_forecast"}
        intent = "create_forecast"
        forbidden.append("problem_statement")
    elif pattern == "negative":
        message = "No, I do not want a forecast."
        prior_question = INTENT_QUESTION
        values = {}
        intent = "not_forecasting"
    else:
        raise ValueError(f"unknown conversation pattern: {pattern}")

    turn = {
        "message": message,
        "gold_extraction": _gold(message, intent, values),
    }
    if prior_question:
        turn["last_assistant_question"] = prior_question

    expected_slot = None
    ideal_question = None
    if intent == "create_forecast":
        state = _post_extraction_state(schema, turn)
        candidate = select_next_slot(schema, state)
        if candidate is not None:
            expected_slot = candidate.slot_id
            ideal_question = schema.get(candidate.slot_id).static_question

    return {
        "corpus_version": CORPUS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "scenario_id": f"conversation-{pattern}-{domain.slug}",
        "category": "conversational_intent",
        "domain": domain.slug,
        "split": split,
        "behavior_tags": [
            "conversational_intent",
            "multi_turn_context" if prior_question else "short_request",
            "no_invention",
            "state_progression",
        ],
        "turns": [turn],
        "gold_intent": intent,
        "gold_final_slots": values,
        "expected_question_slot": expected_slot,
        "ideal_question": ideal_question,
        "must_not_infer": forbidden,
    }


def build_corpus_v2(schema: ForecastingSchema | None = None) -> list[dict[str, Any]]:
    schema = schema or load_schema()
    original = copy.deepcopy(build_corpus(schema))
    for scenario in original:
        scenario["corpus_version"] = CORPUS_VERSION
    patterns = (
        "yes",
        "explicit_request",
        "please_predict",
        "roman_urdu",
        "roman_yes",
        "negative",
    )
    conversational = [
        _conversation_scenario(pattern, domain_index, schema)
        for pattern in patterns
        for domain_index in range(len(DOMAINS))
    ]
    return [*original, *conversational]


def _question_request(
    schema: ForecastingSchema,
    state: DialogueState,
    slot_id: str,
) -> QuestionRequest:
    definition = schema.get(slot_id)
    return QuestionRequest(
        slot_id=slot_id,
        reason="missing or unclear requirement",
        slot_description=definition.description,
        current_state=state.slots[slot_id].model_copy(deep=True),
        confirmed_context={
            key: slot.value
            for key, slot in state.slots.items()
            if slot.confirmed_by_user or slot.status == SlotStatus.CONFIRMED
        },
        static_question=definition.static_question,
        allowed_values=definition.allowed_values,
        other_active_slot_ids=tuple(
            item.slot_id
            for item in schema.slots
            if item.slot_id != slot_id and is_slot_active(item, state)
        ),
    )


def build_sft_examples_v2(
    corpus: list[dict[str, Any]], schema: ForecastingSchema | None = None
) -> list[dict[str, Any]]:
    schema = schema or load_schema()
    examples: list[dict[str, Any]] = []
    for scenario in corpus:
        if scenario["split"] == "test":
            continue
        for turn_index, turn in enumerate(scenario["turns"], start=1):
            state = _state_for_turn(schema, turn)
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": build_slm_extractor_instructions()},
                        {
                            "role": "user",
                            "content": build_slm_extractor_input(turn["message"], state, schema),
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                turn["gold_extraction"], ensure_ascii=False, sort_keys=True
                            ),
                        },
                    ],
                    "metadata": {
                        "scenario_id": scenario["scenario_id"],
                        "split": scenario["split"],
                        "task": "extract",
                        "turn_index": turn_index,
                    },
                }
            )

        slot_id = scenario.get("expected_question_slot")
        if slot_id:
            final_turn = scenario["turns"][-1]
            state = _post_extraction_state(schema, final_turn)
            request = _question_request(schema, state, slot_id)
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": build_slm_question_instructions()},
                        {"role": "user", "content": build_slm_question_input(request)},
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {"question": scenario["ideal_question"]}, ensure_ascii=False
                            ),
                        },
                    ],
                    "metadata": {
                        "scenario_id": scenario["scenario_id"],
                        "split": scenario["split"],
                        "task": "ask",
                        "slot_id": slot_id,
                    },
                }
            )
    return examples


def validate_corpus_v2(
    corpus: list[dict[str, Any]], sft: list[dict[str, Any]]
) -> None:
    if len(corpus) != 320:
        raise ValueError(f"expected 320 v2 scenarios, found {len(corpus)}")
    ids = [scenario["scenario_id"] for scenario in corpus]
    if len(ids) != len(set(ids)):
        raise ValueError("v2 scenario IDs must be unique")
    if Counter(item["split"] for item in corpus) != Counter(
        {"train": 224, "validation": 48, "test": 48}
    ):
        raise ValueError("unexpected v2 split counts")
    if sum(item["category"] == "conversational_intent" for item in corpus) != 120:
        raise ValueError("expected 120 conversational intent scenarios")
    if any(item["metadata"]["split"] == "test" for item in sft):
        raise ValueError("test scenarios must never enter SFT examples")
    yes_examples = [
        item
        for item in sft
        if item["metadata"]["task"] == "extract"
        and json.loads(item["messages"][1]["content"])["current_message"].lower() == "yes"
    ]
    if not yes_examples:
        raise ValueError("v2 SFT data must include yes intent confirmations")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifacts_v2(output_root: Path) -> dict[str, Any]:
    schema = load_schema()
    corpus = build_corpus_v2(schema)
    sft = build_sft_examples_v2(corpus, schema)
    validate_corpus_v2(corpus, sft)

    paths: list[Path] = []
    corpus_path = output_root / "corpus-v2.jsonl"
    _write_jsonl(corpus_path, corpus)
    paths.append(corpus_path)
    for split in ("train", "validation", "test"):
        path = output_root / "splits" / f"{split}.jsonl"
        _write_jsonl(path, [item for item in corpus if item["split"] == split])
        paths.append(path)
    for split in ("train", "validation"):
        path = output_root / "sft" / f"{split}.jsonl"
        _write_jsonl(path, [item for item in sft if item["metadata"]["split"] == split])
        paths.append(path)

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "scenario_count": len(corpus),
        "conversational_scenario_count": 120,
        "category_counts": dict(sorted(Counter(item["category"] for item in corpus).items())),
        "split_counts": dict(sorted(Counter(item["split"] for item in corpus).items())),
        "sft_task_counts": dict(sorted(Counter(item["metadata"]["task"] for item in sft).items())),
        "sft_split_counts": dict(sorted(Counter(item["metadata"]["split"] for item in sft).items())),
        "prompt_profile": "compact_slm_v2",
        "test_examples_in_sft": 0,
        "files": {},
    }
    for path in paths:
        manifest["files"][str(path.relative_to(output_root)).replace("\\", "/")] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest

