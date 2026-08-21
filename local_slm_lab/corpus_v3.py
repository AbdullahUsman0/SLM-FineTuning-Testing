"""Build corpus v3 for completion-only forecasting requirement extraction."""

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

from forecasting_assistant.application.clarification import select_next_slot
from forecasting_assistant.domain.models import (
    DialogueTurn,
    ExtractorResult,
    Intent,
    Requiredness,
    SlotState,
    SlotStatus,
)
from forecasting_assistant.domain.schema import (
    ForecastingSchema,
    create_initial_state,
    load_schema,
)

from local_slm_lab.corpus import DOMAINS, full_gold
from local_slm_lab.corpus_v2 import (
    SCHEMA_VERSION,
    _gold,
    _post_extraction_state,
    _state_for_turn,
    build_corpus_v2,
)
from local_slm_lab.slm_prompts import (
    build_slm_extractor_input,
    build_slm_extractor_instructions,
)


CORPUS_VERSION = "forecasting-llmrei-extraction-1049-v3"
NON_ANSWERS = (
    "I don't know yet.",
    "Not sure.",
    "Can we decide that later?",
)


def build_corpus_v3(schema: ForecastingSchema | None = None) -> list[dict[str, Any]]:
    """Retain the reviewed v2 scenarios while giving the new corpus its own identity."""
    corpus = copy.deepcopy(build_corpus_v2(schema))
    for scenario in corpus:
        scenario["corpus_version"] = CORPUS_VERSION
    return corpus


def _prompt_completion(
    message: str,
    state,
    schema: ForecastingSchema,
    gold: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": build_slm_extractor_instructions()},
            {"role": "user", "content": build_slm_extractor_input(message, state, schema)},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps(gold, ensure_ascii=False, sort_keys=True),
            }
        ],
        "metadata": metadata,
    }


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict) and {"periods", "unit"} <= set(value):
        amount = value["periods"]
        if isinstance(amount, float) and amount.is_integer():
            amount = int(amount)
        unit = str(value["unit"])
        if amount != 1 and not unit.endswith("s"):
            unit += "s"
        return f"{amount} {unit}"
    if isinstance(value, list):
        return ", ".join(_render_value(item) for item in value)
    return str(value)


def _short_answer(slot_id: str, value: Any) -> str:
    if slot_id == "intent":
        return "yes"
    if slot_id == "source_mode" and value == "upload":
        return "uploaded"
    return _render_value(value)


def _answer_sentence(slot_id: str, rendered: str) -> str:
    templates = {
        "intent": "Yes, I want to create a time-series forecast.",
        "problem_statement": "The forecasting problem is {value}.",
        "business_goal": "It will support {value}.",
        "success_criteria": "Success means {value}.",
        "target_column": "The target column is {value}.",
        "target_description": "The target represents {value}.",
        "target_unit": "It is measured in {value}.",
        "time_column": "The time column is {value}.",
        "frequency": "The observations arrive every {value}.",
        "forecast_horizon": "Forecast the next {value}.",
        "dataset_type": "The dataset type is {value}.",
        "source_mode": "The data will be provided by {value}.",
        "source_reference": "Use {value}.",
        "file_format": "The file format is {value}.",
        "forecast_type": "Use a {value} forecast.",
        "output_granularity": "Return results every {value}.",
        "primary_metric": "Use {value} as the primary metric.",
        "contains_sensitive_data": "The data contains sensitive information: {value}.",
        "privacy_constraints": "The restrictions are {value}.",
        "series_id_columns": "The series identifiers are {value}.",
        "covariate_availability": "The covariates will be available {value}.",
    }
    return templates.get(slot_id, "The answer is {value}.").format(value=rendered)


def _targeted_state(
    schema: ForecastingSchema, domain, selected_slot: str
):
    """Create a realistic state where exactly one core requirement is next."""
    values = full_gold(domain)
    state = create_initial_state(schema)
    for definition in schema.slots:
        slot_id = definition.slot_id
        if definition.requiredness != Requiredness.REQUIRED:
            continue
        if slot_id == selected_slot:
            continue
        value = values[slot_id]
        state.slots[slot_id] = SlotState(
            slot_id=slot_id,
            value=value,
            status=SlotStatus.CONFIRMED,
            confidence=1.0,
            evidence_text="synthetic reviewed context",
            confirmed_by_user=True,
        )
    state.intent = (
        Intent.AMBIGUOUS
        if selected_slot == "intent"
        else Intent.CREATE_FORECAST
    )
    selected = select_next_slot(schema, state)
    if selected is None or selected.slot_id != selected_slot:
        actual = None if selected is None else selected.slot_id
        raise ValueError(
            f"cannot construct selected-slot state for {selected_slot}: found {actual}"
        )
    state.turns.append(
        DialogueTurn(
            turn_number=1,
            user_message="[earlier reviewed requirements]",
            assistant_message=schema.get(selected_slot).static_question,
        )
    )
    return state, values[selected_slot]


def build_sft_examples_v3(
    corpus: list[dict[str, Any]], schema: ForecastingSchema | None = None
) -> list[dict[str, Any]]:
    """Create extraction-only prompt/completion records with answer-state augmentation."""
    schema = schema or load_schema()
    examples: list[dict[str, Any]] = []
    for scenario in corpus:
        if scenario["split"] == "test":
            continue

        for turn_index, turn in enumerate(scenario["turns"], start=1):
            examples.append(
                _prompt_completion(
                    turn["message"],
                    _state_for_turn(schema, turn),
                    schema,
                    turn["gold_extraction"],
                    {
                        "scenario_id": scenario["scenario_id"],
                        "split": scenario["split"],
                        "task": "extract",
                        "variant": "reviewed_base",
                        "turn_index": turn_index,
                        "domain": scenario["domain"],
                        "category": scenario["category"],
                    },
                )
            )

    core_slots = [
        definition.slot_id
        for definition in schema.slots
        if definition.requiredness == Requiredness.REQUIRED
    ]
    for domain_index, domain in enumerate(DOMAINS):
        split = "train" if domain_index < 14 else "validation" if domain_index < 17 else "test"
        if split == "test":
            continue
        for slot_index, slot_id in enumerate(core_slots):
            state, value = _targeted_state(schema, domain, slot_id)
            rendered = _render_value(value)
            answer_messages = (
                ("selected_short", _short_answer(slot_id, value)),
                ("selected_sentence", _answer_sentence(slot_id, rendered)),
            )
            for variant, message in answer_messages:
                intent = "create_forecast"
                examples.append(
                    _prompt_completion(
                        message,
                        state,
                        schema,
                        _gold(message, intent, {slot_id: value}),
                        {
                            "scenario_id": f"targeted-{slot_id}-{domain.slug}",
                            "split": split,
                            "task": "extract",
                            "variant": variant,
                            "selected_slot": slot_id,
                            "domain": domain.slug,
                            "category": "targeted_selected_slot",
                        },
                    )
                )

            non_answer = NON_ANSWERS[(domain_index + slot_index) % len(NON_ANSWERS)]
            state_intent = state.intent.value
            examples.append(
                _prompt_completion(
                    non_answer,
                    state,
                    schema,
                    _gold(non_answer, state_intent, {}),
                    {
                        "scenario_id": f"targeted-{slot_id}-{domain.slug}",
                        "split": split,
                        "task": "extract",
                        "variant": "selected_non_answer",
                        "selected_slot": slot_id,
                        "domain": domain.slug,
                        "category": "targeted_selected_slot",
                    },
                )
            )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example in examples:
        key = json.dumps(
            [example["prompt"], example["completion"]],
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def validate_corpus_v3(
    corpus: list[dict[str, Any]], sft: list[dict[str, Any]]
) -> None:
    if len(corpus) != 320:
        raise ValueError(f"expected 320 v3 scenarios, found {len(corpus)}")
    if any(item["metadata"]["split"] == "test" for item in sft):
        raise ValueError("test scenarios must never enter SFT examples")
    if any(item["metadata"]["task"] != "extract" for item in sft):
        raise ValueError("v3 must contain extraction examples only")
    if any("messages" in item for item in sft):
        raise ValueError("v3 must use explicit prompt/completion records")
    if len(sft) != 1049:
        raise ValueError(f"expected 1049 focused v3 examples, found {len(sft)}")

    for item in sft:
        if [message["role"] for message in item["prompt"]] != ["system", "user"]:
            raise ValueError("prompt roles must be system then user")
        if [message["role"] for message in item["completion"]] != ["assistant"]:
            raise ValueError("completion must contain one assistant message")
        result = ExtractorResult.model_validate_json(item["completion"][0]["content"])
        payload = json.loads(item["prompt"][1]["content"])
        for update in result.updates:
            if update.evidence_text not in payload["current_message"]:
                raise ValueError(
                    f"ungrounded evidence in {item['metadata']['scenario_id']}"
                )
        expected_slot = item["metadata"].get("selected_slot")
        if expected_slot and payload["selected_slot"] != expected_slot:
            raise ValueError(
                f"answer augmentation selected the wrong slot in "
                f"{item['metadata']['scenario_id']}"
            )

    variants = Counter(item["metadata"]["variant"] for item in sft)
    if variants["selected_short"] == 0 or variants["selected_non_answer"] == 0:
        raise ValueError("v3 is missing selected-slot answer coverage")
    serialized = [
        json.dumps([item["prompt"], item["completion"]], sort_keys=True)
        for item in sft
    ]
    if len(serialized) != len(set(serialized)):
        raise ValueError("v3 must not contain exact duplicate training examples")
    core_slots = {
        definition.slot_id
        for definition in load_schema().slots
        if definition.requiredness == Requiredness.REQUIRED
    }
    covered_slots = {
        item["metadata"].get("selected_slot")
        for item in sft
        if item["metadata"]["variant"] == "selected_short"
    }
    if covered_slots != core_slots:
        raise ValueError("v3 must cover every required slot with direct answers")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifacts_v3(output_root: Path) -> dict[str, Any]:
    schema = load_schema()
    corpus = build_corpus_v3(schema)
    sft = build_sft_examples_v3(corpus, schema)
    validate_corpus_v3(corpus, sft)

    paths: list[Path] = []
    corpus_path = output_root / "corpus-v3.jsonl"
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
        "split_counts": dict(sorted(Counter(item["split"] for item in corpus).items())),
        "sft_example_count": len(sft),
        "sft_split_counts": dict(
            sorted(Counter(item["metadata"]["split"] for item in sft).items())
        ),
        "sft_variant_counts": dict(
            sorted(Counter(item["metadata"]["variant"] for item in sft).items())
        ),
        "training_task": "structured_extraction_only",
        "dataset_format": "conversational_prompt_completion",
        "loss_scope": "assistant_completion_only",
        "question_policy": "deterministic_schema_questions_at_runtime",
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
