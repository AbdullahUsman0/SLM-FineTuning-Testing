"""Build corpus v4: larger, more diverse extraction-only dataset.

v4 keeps the reviewed v3 scenarios as a base and adds:
- 10 new domains (7 train, 3 validation) for vocabulary diversity
- OpenAI-paraphrased variants of base scenario messages
- Multi-slot answer clusters that teach the model to extract several
  requirements from one user turn
- Roman-Urdu and expanded-context answer templates
- More balanced per-slot positive-example counts

The output is extraction-only prompt/completion records aligned with the
production `completion_only_loss=True` training objective.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
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

from local_slm_lab.corpus import DOMAINS as BASE_DOMAINS, DomainSpec, full_gold
from local_slm_lab.corpus_v2 import SCHEMA_VERSION, _gold, _state_for_turn
from local_slm_lab.corpus_v3 import (
    CORPUS_VERSION as V3_VERSION,
    build_corpus_v3,
    _prompt_completion,
    _render_value,
    _targeted_state,
)
from local_slm_lab.slm_prompts import (
    build_slm_extractor_input,
    build_slm_extractor_instructions,
)


CORPUS_VERSION = "forecasting-llmrei-extraction-v4"
PARAPHRASE_MODEL = "gpt-5-mini-2025-08-07"
PARAPHRASES_PER_MESSAGE = 3
NON_ANSWERS = (
    "I don't know yet.",
    "Not sure.",
    "Can we decide that later?",
    "I'll need to check with my team.",
    "Let's skip that for now.",
)

NEW_DOMAINS = (
    DomainSpec(
        "ride_hailing",
        "rides_completed",
        "daily ride-hailing trips",
        "trips",
        "date",
        "daily",
        {"periods": 1, "unit": "day"},
        "14 days",
        {"periods": 14, "unit": "day"},
        "rides_completed.csv",
        "driver supply planning",
        "MAE below 8 percent",
        "daily",
    ),
    DomainSpec(
        "kse_index",
        "kse100_close",
        "Karachi Stock Exchange 100 index close",
        "index points",
        "date",
        "daily",
        {"periods": 1, "unit": "day"},
        "10 days",
        {"periods": 10, "unit": "day"},
        "kse100.csv",
        "portfolio hedging",
        "MAE below 300 points",
        "daily",
    ),
    DomainSpec(
        "restaurant_orders",
        "order_count",
        "restaurant order volume",
        "orders",
        "date",
        "daily",
        {"periods": 1, "unit": "day"},
        "21 days",
        {"periods": 21, "unit": "day"},
        "restaurant_orders.csv",
        "kitchen staffing",
        "MAE below 10 percent",
        "daily",
    ),
    DomainSpec(
        "university_enrollment",
        "enrolled_students",
        "university student enrollment",
        "students",
        "academic_year",
        "yearly",
        {"periods": 1, "unit": "year"},
        "5 years",
        {"periods": 5, "unit": "year"},
        "enrollment.csv",
        "capacity planning",
        "MAE below 5 percent",
        "yearly",
    ),
    DomainSpec(
        "manufacturing_defects",
        "defect_rate",
        "product defect rate",
        "percent",
        "date",
        "daily",
        {"periods": 1, "unit": "day"},
        "30 days",
        {"periods": 30, "unit": "day"},
        "defect_rate.csv",
        "quality control",
        "MAE below 0.5 percent",
        "daily",
    ),
    DomainSpec(
        "streaming_subscriptions",
        "subscribers",
        "streaming service subscribers",
        "subscribers",
        "month",
        "monthly",
        {"periods": 1, "unit": "month"},
        "12 months",
        {"periods": 12, "unit": "month"},
        "streaming_subs.csv",
        "content budgeting",
        "MAE below 6 percent",
        "monthly",
    ),
    DomainSpec(
        "port_traffic",
        "container_count",
        "seaport container throughput",
        "TEU",
        "month",
        "monthly",
        {"periods": 1, "unit": "month"},
        "9 months",
        {"periods": 9, "unit": "month"},
        "port_traffic.csv",
        "logistics planning",
        "MAE below 7 percent",
        "monthly",
    ),
    DomainSpec(
        "mobile_data_usage",
        "data_gb",
        "mobile data consumption",
        "GB",
        "date",
        "daily",
        {"periods": 1, "unit": "day"},
        "14 days",
        {"periods": 14, "unit": "day"},
        "data_usage.csv",
        "network capacity planning",
        "MAE below 8 percent",
        "daily",
    ),
    DomainSpec(
        "vaccine_doses",
        "doses_administered",
        "vaccine doses administered",
        "doses",
        "week_start",
        "weekly",
        {"periods": 1, "unit": "week"},
        "8 weeks",
        {"periods": 8, "unit": "week"},
        "vaccine_doses.csv",
        "supply chain planning",
        "MAE below 10 percent",
        "weekly",
    ),
    DomainSpec(
        "carbon_emissions",
        "co2_tonnes",
        "carbon dioxide emissions",
        "tonnes",
        "month",
        "monthly",
        {"periods": 1, "unit": "month"},
        "6 months",
        {"periods": 6, "unit": "month"},
        "emissions.csv",
        "compliance reporting",
        "MAE below 5 percent",
        "monthly",
    ),
)

ALL_DOMAINS = (*BASE_DOMAINS, *NEW_DOMAINS)


def split_for_domain_v4(index: int) -> str:
    """Domain-disjoint splits: original 20 domains plus 10 new train/val domains."""
    if index < 14:
        return "train"
    if index < 17:
        return "validation"
    if index < 20:
        return "test"
    if index < 27:
        return "train"
    if index < 30:
        return "validation"
    return "test"


def _load_env_key() -> str | None:
    """Read OPENAI_API_KEY from the production fpy .env file."""
    env_path = PROJECT_ROOT.parent / "fpy" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                return value.strip().strip('"').strip("'")
    return os.environ.get("OPENAI_API_KEY")


def _paraphrase_cache_path() -> Path:
    return PROJECT_ROOT / ".cache" / "paraphrases_v4.json"


def _load_paraphrase_cache() -> dict[str, list[str]]:
    path = _paraphrase_cache_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_paraphrase_cache(cache: dict[str, list[str]]) -> None:
    path = _paraphrase_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _openai_paraphrases(message: str, n: int) -> list[str]:
    """Generate `n` task-preserving paraphrases of a user message."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for OpenAI paraphrasing") from exc

    key = _load_env_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in fpy/.env or environment")

    prompt = (
        "Paraphrase the following user message in 3 different natural ways. "
        "Preserve all concrete facts (numbers, column names, file names, units). "
        "Return only the paraphrases, one per line, with no numbering or labels."
    )

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": PARAPHRASE_MODEL,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            "n": 1,
            "max_completion_tokens": 400,
        },
        timeout=60,
    )
    try:
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"OpenAI API error: {response.status_code} {response.text[:500]}"
        ) from exc
    content = response.json()["choices"][0]["message"]["content"]
    paraphrases = [line.strip() for line in content.splitlines() if line.strip()]
    if len(paraphrases) < n:
        paraphrases = (paraphrases + [message] * n)[:n]
    return paraphrases[:n]


def get_paraphrases(message: str, n: int = PARAPHRASES_PER_MESSAGE) -> list[str]:
    """Return cached paraphrases or fetch and cache them from OpenAI."""
    cache = _load_paraphrase_cache()
    if message in cache and len(cache[message]) >= n:
        return cache[message][:n]
    paraphrases = _openai_paraphrases(message, n)
    cache[message] = paraphrases
    _save_paraphrase_cache(cache)
    time.sleep(0.2)
    return paraphrases


def _full_gold_for_domain(domain: DomainSpec) -> dict[str, Any]:
    """Mirror corpus.full_gold for a possibly new domain."""
    return {
        "intent": "create_forecast",
        "problem_statement": f"forecast {domain.target_description}",
        "business_goal": domain.business_goal,
        "success_criteria": domain.success_criteria,
        "target_column": domain.target_column,
        "target_description": domain.target_description,
        "target_unit": domain.unit,
        "time_column": domain.time_column,
        "frequency": domain.frequency_value,
        "forecast_horizon": domain.horizon_value,
        "dataset_type": "single_series",
        "source_mode": "upload",
        "source_reference": domain.source,
        "file_format": "csv",
        "forecast_type": "point",
        "output_granularity": domain.frequency_value,
        "primary_metric": domain.primary_metric,
        "contains_sensitive_data": False,
    }


def build_corpus_v4(schema: ForecastingSchema | None = None) -> list[dict[str, Any]]:
    """Return 420 scenarios: 320 reviewed v3 scenarios plus 100 new-domain scenarios."""
    schema = schema or load_schema()
    corpus = copy.deepcopy(build_corpus_v3(schema))
    for scenario in corpus:
        scenario["corpus_version"] = CORPUS_VERSION

    # Generate v1-style scenarios for the 10 new domains.
    from local_slm_lab.corpus import _scenario as base_scenario

    for domain_index, domain in enumerate(NEW_DOMAINS, start=len(BASE_DOMAINS)):
        for category in (
            "complete",
            "missing_required",
            "ambiguous",
            "correction",
            "conflicting",
            "multi_series",
            "probabilistic_covariate",
            "governance_privacy",
            "robustness",
            "intent_boundary",
        ):
            scenario = base_scenario(category, domain, domain_index, schema)
            scenario["corpus_version"] = CORPUS_VERSION
            scenario["split"] = split_for_domain_v4(domain_index)
            scenario["scenario_id"] = f"{category}-{domain.slug}"
            corpus.append(scenario)
    return corpus


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


def _roman_urdu_answer(slot_id: str, rendered: str) -> str:
    templates = {
        "intent": "Haan, mujhe time-series forecast chahiye.",
        "problem_statement": "Mujhe {value} forecast karna hai.",
        "business_goal": "Yeh {value} ke liye hai.",
        "success_criteria": "Kamyabi ka matlab {value} hai.",
        "target_column": "Target column {value} hai.",
        "target_description": "Yeh {value} represent karta hai.",
        "target_unit": "Iski unit {value} hai.",
        "time_column": "Time column {value} hai.",
        "frequency": "Yeh har {value} observe hota hai.",
        "forecast_horizon": "Agle {value} ke liye forecast karo.",
        "dataset_type": "Dataset type {value} hai.",
        "source_mode": "Data {value} se aayega.",
        "source_reference": "{value} use karo.",
        "file_format": "File format {value} hai.",
        "forecast_type": "{value} forecast chahiye.",
        "output_granularity": "Result har {value} chahiye.",
        "primary_metric": "Primary metric {value} honi chahiye.",
    }
    return templates.get(slot_id, "Jawab {value} hai.").format(value=rendered)


def _multi_slot_answer(slot_id: str, rendered: str, domain: DomainSpec) -> str | None:
    """Return a message that also provides 1-2 related slots, or None."""
    if slot_id == "target_column":
        return f"The target column is {rendered}, which represents {domain.target_description}."
    if slot_id == "target_description":
        return f"We measure {domain.target_description} in {domain.unit}."
    if slot_id == "target_unit":
        return f"{domain.target_column} is measured in {rendered}."
    if slot_id == "time_column":
        return f"Use {rendered} as the timestamp; observations are {domain.frequency}."
    if slot_id == "frequency":
        return f"Data is {rendered} and we need the next {domain.horizon}."
    if slot_id == "forecast_horizon":
        return f"Forecast {rendered} ahead; the data is {domain.frequency}."
    if slot_id == "source_mode":
        return f"We'll {rendered} the file {domain.source}."
    if slot_id == "source_reference":
        return f"Use {rendered}; it is a CSV upload."
    if slot_id == "file_format":
        return f"The file {domain.source} is in {rendered} format."
    if slot_id == "forecast_type":
        return f"Use a {rendered} forecast at {domain.frequency} granularity."
    if slot_id == "output_granularity":
        return f"Return forecasts every {rendered}, matching the input frequency."
    if slot_id == "primary_metric":
        return f"Judge the forecast with {rendered} as the main metric."
    if slot_id == "business_goal":
        return f"The goal is {rendered}; success means {domain.success_criteria}."
    if slot_id == "success_criteria":
        return f"We need {rendered} to support {domain.business_goal}."
    if slot_id == "problem_statement":
        return f"We need to forecast {domain.target_description} from {domain.source}."
    return None


def _targeted_examples_for_domain(
    schema: ForecastingSchema,
    domain: DomainSpec,
    domain_index: int,
) -> list[dict[str, Any]]:
    """Generate targeted single-slot and multi-slot extraction examples."""
    split = split_for_domain_v4(domain_index)
    if split == "test":
        return []

    core_slots = [
        definition.slot_id
        for definition in schema.slots
        if definition.requiredness == Requiredness.REQUIRED
    ]
    examples: list[dict[str, Any]] = []

    for slot_index, slot_id in enumerate(core_slots):
        state, value = _targeted_state(schema, domain, slot_id)
        rendered = _render_value(value)
        base_metadata = {
            "scenario_id": f"targeted-{slot_id}-{domain.slug}",
            "split": split,
            "task": "extract",
            "selected_slot": slot_id,
            "domain": domain.slug,
            "category": "targeted_selected_slot",
        }

        variants: list[tuple[str, str]] = [
            ("selected_short", _short_answer(slot_id, value)),
            ("selected_sentence", _answer_sentence(slot_id, rendered)),
        ]

        roman = _roman_urdu_answer(slot_id, rendered)
        if roman:
            variants.append(("selected_roman_urdu", roman))

        multi = _multi_slot_answer(slot_id, rendered, domain)
        if multi:
            variants.append(("selected_multi_slot", multi))

        for variant, message in variants:
            # Evidence must be a substring of the generated message.
            evidence = message
            gold_values = {slot_id: value}
            if variant == "selected_multi_slot":
                # Include the related slots actually present in the message.
                extra = _extract_extra_slots_from_multi(slot_id, message, domain, value)
                gold_values.update(extra)
            gold = _gold(message, "create_forecast", gold_values)
            examples.append(
                _prompt_completion(
                    message,
                    state,
                    schema,
                    gold,
                    {**base_metadata, "variant": variant},
                )
            )

        non_answer = NON_ANSWERS[(domain_index + slot_index) % len(NON_ANSWERS)]
        examples.append(
            _prompt_completion(
                non_answer,
                state,
                schema,
                _gold(non_answer, state.intent.value, {}),
                {**base_metadata, "variant": "selected_non_answer"},
            )
        )

    return examples


def _extract_extra_slots_from_multi(
    primary_slot: str, message: str, domain: DomainSpec, primary_value: Any
) -> dict[str, Any]:
    """Return the additional slots present in a multi-slot answer message."""
    extras: dict[str, Any] = {}
    full = _full_gold_for_domain(domain)
    related = {
        "target_column": ("target_description",),
        "target_description": ("target_unit",),
        "target_unit": ("target_column",),
        "time_column": ("frequency",),
        "frequency": ("forecast_horizon",),
        "forecast_horizon": ("frequency",),
        "source_mode": ("source_reference",),
        "source_reference": ("file_format",),
        "file_format": ("source_reference",),
        "business_goal": ("success_criteria",),
        "success_criteria": ("business_goal",),
    }
    message_lower = message.lower()
    for slot_id in related.get(primary_slot, ()):
        value = full[slot_id]
        rendered = _render_value(value).lower()
        if rendered and rendered in message_lower:
            extras[slot_id] = value
    return extras


def _cluster_examples_for_domain(
    schema: ForecastingSchema,
    domain: DomainSpec,
    domain_index: int,
) -> list[dict[str, Any]]:
    """Generate examples where one user message fills several related slots."""
    split = split_for_domain_v4(domain_index)
    if split == "test":
        return []

    clusters: list[tuple[str, list[str], str]] = [
        (
            "source",
            ["source_mode", "source_reference", "file_format"],
            "We will {source_mode} the data in {source_reference}, which is a {file_format} file.",
        ),
        (
            "target",
            ["target_column", "target_description", "target_unit"],
            "The target is {target_column}, representing {target_description}, measured in {target_unit}.",
        ),
        (
            "time",
            ["time_column", "frequency", "forecast_horizon"],
            "Use {time_column} as the timestamp. The series is {frequency} and we need a forecast for the next {forecast_horizon}.",
        ),
        (
            "output",
            ["forecast_type", "output_granularity", "primary_metric"],
            "Produce a {forecast_type} forecast at {output_granularity} granularity, judged by {primary_metric}.",
        ),
        (
            "business",
            ["business_goal", "success_criteria"],
            "The forecast supports {business_goal}. Success means {success_criteria}.",
        ),
    ]

    examples: list[dict[str, Any]] = []
    full = _full_gold_for_domain(domain)
    state = create_initial_state(schema)
    # Confirm all slots except the cluster slots.
    for definition in schema.slots:
        slot_id = definition.slot_id
        if slot_id in [s for _, slots, _ in clusters for s in slots]:
            continue
        if slot_id in full:
            state.slots[slot_id] = SlotState(
                slot_id=slot_id,
                value=full[slot_id],
                status=SlotStatus.CONFIRMED,
                confidence=1.0,
                evidence_text="synthetic reviewed context",
                confirmed_by_user=True,
            )
    state.intent = Intent.CREATE_FORECAST
    state.turns.append(
        DialogueTurn(
            turn_number=1,
            user_message="[earlier reviewed requirements]",
            assistant_message="Please share the remaining details.",
        )
    )

    for cluster_name, slot_ids, template in clusters:
        message = template.format(**{s: _render_value(full[s]) for s in slot_ids})
        values = {slot_id: full[slot_id] for slot_id in slot_ids}
        examples.append(
            _prompt_completion(
                message,
                state,
                schema,
                _gold(message, "create_forecast", values),
                {
                    "scenario_id": f"cluster-{cluster_name}-{domain.slug}",
                    "split": split,
                    "task": "extract",
                    "variant": "cluster_answer",
                    "domain": domain.slug,
                    "category": "multi_slot_cluster",
                },
            )
        )
    return examples


def _paraphrased_base_examples(
    corpus: list[dict[str, Any]], schema: ForecastingSchema
) -> list[dict[str, Any]]:
    """Add paraphrased variants of every base scenario turn."""
    examples: list[dict[str, Any]] = []
    for scenario in corpus:
        if scenario["split"] == "test":
            continue
        for turn_index, turn in enumerate(scenario["turns"], start=1):
            original = turn["message"]
            paraphrases = get_paraphrases(original, PARAPHRASES_PER_MESSAGE)
            for para_index, paraphrase in enumerate(paraphrases, start=1):
                # Re-ground evidence in the paraphrased text.
                gold = copy.deepcopy(turn["gold_extraction"])
                for update in gold.get("updates", []):
                    update["evidence_text"] = paraphrase
                examples.append(
                    _prompt_completion(
                        paraphrase,
                        _state_for_turn(schema, turn),
                        schema,
                        gold,
                        {
                            "scenario_id": scenario["scenario_id"],
                            "split": scenario["split"],
                            "task": "extract",
                            "variant": f"paraphrase_{para_index}",
                            "turn_index": turn_index,
                            "domain": scenario["domain"],
                            "category": scenario["category"],
                        },
                    )
                )
    return examples


def build_sft_examples_v4(
    corpus: list[dict[str, Any]], schema: ForecastingSchema | None = None
) -> list[dict[str, Any]]:
    """Create the full v4 extraction-only SFT dataset."""
    schema = schema or load_schema()
    examples: list[dict[str, Any]] = []

    # 1) Reviewed base examples from v3 scenarios and new-domain base scenarios.
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

    # 2) OpenAI paraphrases of base turns.
    examples.extend(_paraphrased_base_examples(corpus, schema))

    # 3) Targeted single-slot and multi-slot answer variants.
    for domain_index, domain in enumerate(ALL_DOMAINS):
        examples.extend(_targeted_examples_for_domain(schema, domain, domain_index))

    # 4) Clustered multi-slot answer examples.
    for domain_index, domain in enumerate(ALL_DOMAINS):
        examples.extend(_cluster_examples_for_domain(schema, domain, domain_index))

    # Deduplicate on (prompt, completion).
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


def validate_corpus_v4(
    corpus: list[dict[str, Any]], sft: list[dict[str, Any]]
) -> None:
    """Fail-fast checks for v4 corpus integrity."""
    if len(corpus) != 420:
        raise ValueError(f"expected 420 v4 scenarios, found {len(corpus)}")
    if any(item["metadata"]["split"] == "test" for item in sft):
        raise ValueError("test scenarios must never enter SFT examples")
    if any(item["metadata"]["task"] != "extract" for item in sft):
        raise ValueError("v4 must contain extraction examples only")
    if any("messages" in item for item in sft):
        raise ValueError("v4 must use explicit prompt/completion records")

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
                    f"ungrounded evidence in {item['metadata']['scenario_id']} "
                    f"variant={item['metadata'].get('variant')}"
                )
        expected_slot = item["metadata"].get("selected_slot")
        if expected_slot and payload["selected_slot"] != expected_slot:
            raise ValueError(
                f"answer augmentation selected the wrong slot in "
                f"{item['metadata']['scenario_id']}"
            )

    serialized = [
        json.dumps([item["prompt"], item["completion"]], sort_keys=True)
        for item in sft
    ]
    if len(serialized) != len(set(serialized)):
        raise ValueError("v4 must not contain exact duplicate training examples")

    core_slots = {
        definition.slot_id
        for definition in load_schema().slots
        if definition.requiredness == Requiredness.REQUIRED
    }
    covered_slots = {
        item["metadata"].get("selected_slot")
        for item in sft
        if item["metadata"].get("variant") == "selected_short"
    }
    if covered_slots != core_slots:
        raise ValueError("v4 must cover every required slot with direct answers")

    # Require a minimum positive-example count for each required slot.
    slot_counts: Counter = Counter()
    for item in sft:
        result = ExtractorResult.model_validate_json(item["completion"][0]["content"])
        for update in result.updates:
            slot_counts[update.slot_id] += 1
    underrepresented = [
        slot_id
        for slot_id in core_slots
        if slot_counts.get(slot_id, 0) < 100
    ]
    if underrepresented:
        raise ValueError(
            f"required slots with fewer than 100 positive examples: {underrepresented}"
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifacts_v4(output_root: Path) -> dict[str, Any]:
    schema = load_schema()
    corpus = build_corpus_v4(schema)
    sft = build_sft_examples_v4(corpus, schema)
    validate_corpus_v4(corpus, sft)

    paths: list[Path] = []
    corpus_path = output_root / "corpus-v4.jsonl"
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
