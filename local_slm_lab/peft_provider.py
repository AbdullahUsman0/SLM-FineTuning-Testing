"""Lazy Hugging Face/PEFT provider for the isolated fine-tuned-model experiment."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if str(FPY_SRC) not in sys.path:
    sys.path.insert(0, str(FPY_SRC))

from forecasting_assistant.domain.models import (  # noqa: E402
    DialogueState,
    ExtractorResult,
    Intent,
    QuestionOutput,
    QuestionRequest,
    SlotStatus,
    SlotState,
    SlotUpdate,
)
from forecasting_assistant.domain.schema import ForecastingSchema  # noqa: E402
from forecasting_assistant.application.clarification import select_next_slot  # noqa: E402
from forecasting_assistant.application.normalization import normalize_value  # noqa: E402
from forecasting_assistant.application.validation import validate_slot  # noqa: E402
from forecasting_assistant.prompts.extractor import safe_provider_value  # noqa: E402
from local_slm_lab.slm_prompts import (  # noqa: E402
    build_slm_extractor_input,
    build_slm_extractor_instructions,
    build_slm_question_input,
    build_slm_question_instructions,
)


RUN_ROOT = PROJECT_ROOT / "training-runs" / "qwen35-08b-lora-v3"
VARIANT_ADAPTERS = {
    "base": None,
    "best": RUN_ROOT / "best-adapter",
    "final": RUN_ROOT / "checkpoint-165",
}

SHORT_POSITIVE_INTENT_REPLIES = {
    "yes", "yeah", "yep", "y", "haan", "han", "jee", "ji", "bilkul"
}
SHORT_NEGATIVE_INTENT_REPLIES = {
    "no", "nope", "n", "nahi", "nahin", "na", "do not", "dont", "not forecasting"
}

SHORT_SLOT_NON_ANSWERS = {
    "",
    "idk",
    "i dont know",
    "i don t know",
    "i do not know",
    "not sure",
    "skip",
    "unknown",
}
SAFE_SHORT_ANSWER_TYPES = {"boolean", "datetime", "duration", "enum", "string", "timezone"}
LOW_INFORMATION_SLOT_VALUES = {
    "problem_statement": {"data", "forecast", "forecasting", "predict", "prediction", "time series"},
    "target_description": {"data", "forecast", "forecasting", "target", "time series", "value"},
    "business_goal": {"business", "forecasting", "planning", "unknown"},
    "success_criteria": {"accuracy", "accurate", "good", "unknown"},
}

_DURATION_UNITS = {
    "second": "second",
    "seconds": "second",
    "secondly": "second",
    "minute": "minute",
    "minutes": "minute",
    "minutely": "minute",
    "hour": "hour",
    "hours": "hour",
    "hourly": "hour",
    "day": "day",
    "days": "day",
    "daily": "day",
    "week": "week",
    "weeks": "week",
    "weekly": "week",
    "month": "month",
    "months": "month",
    "monthly": "month",
    "quarter": "quarter",
    "quarters": "quarter",
    "quarterly": "quarter",
    "year": "year",
    "years": "year",
    "yearly": "year",
}
_DURATION_TEXT_PATTERN = re.compile(
    r"\s*(?:(?:for|over)\s+)?(?:(?:the\s+)?(?:next|following)\s+)?"
    r"(?:(?P<number>\d+(?:\.\d+)?)|(?:a|an|one))?\s*(?P<unit>[a-zA-Z]+)\s*",
    re.IGNORECASE,
)
PROMPT_LEAK_TERMS = (
    "confirmed_slots",
    "unsupported_claims",
    "candidate_value",
    "slot_id",
    "evidence_text",
    "system:",
    "user:",
    "assistant:",
    "json",
)


@dataclass(frozen=True)
class PeftInferenceConfig:
    base_model: str = "Qwen/Qwen3.5-0.8B"
    variant: str = "best"
    adapter_path: Path | None = None
    device: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 1024


def resolve_adapter_path(variant: str, custom_path: str | Path | None = None) -> Path | None:
    if variant not in VARIANT_ADAPTERS and variant != "custom":
        raise ValueError(f"unsupported PEFT variant: {variant}")
    if variant == "custom":
        if custom_path is None:
            raise ValueError("custom variant requires an adapter path")
        return Path(custom_path).expanduser().resolve()
    return VARIANT_ADAPTERS[variant]


class TransformersPeftProvider:
    """Run the same structured contracts through the HF base model or a LoRA adapter."""

    name = "transformers_peft"

    def __init__(self, config: PeftInferenceConfig, schema: ForecastingSchema) -> None:
        self.config = config
        self.schema = schema
        adapter_path = config.adapter_path or resolve_adapter_path(config.variant)
        if adapter_path is not None and not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"LoRA adapter is incomplete or missing: {adapter_path}")
        self.adapter_path = adapter_path
        self._traces: list[dict[str, Any]] = []
        suffix = "base" if adapter_path is None else str(adapter_path)
        self.model = f"{config.base_model}::{suffix}"
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            import transformers
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PEFT inference dependencies are missing. Install "
                "training/requirements-inference.txt in a dedicated environment."
            ) from exc

        # Use AutoModelForCausalLM: SFTTrainer (used during LoRA training) loads with
        # CausalLM semantics; adapter_config confirms base_model_class is
        # Qwen3_5ForConditionalGeneration which AutoModelForCausalLM dispatches to.
        # AutoModelForImageTextToText adds a vision tower and returns pixel_values=None
        # for text-only inputs, silently breaking inference.
        auto_model = AutoModelForCausalLM

        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.config.dtype not in dtype_map:
            raise ValueError(f"unsupported dtype: {self.config.dtype}")
        requested_dtype = dtype_map[self.config.dtype]
        if self.config.device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = self.config.device
        if target_device == "cpu" and requested_dtype == torch.float16:
            raise ValueError("float16 CPU inference is unsupported; use auto, bfloat16, or float32")

        self._torch = torch
        # Use AutoTokenizer instead of AutoProcessor: the VL processor's
        # apply_chat_template raises exceptions not caught by `except TypeError`,
        # preventing generate() from ever being reached (~4ms latency with 0% success).
        # AutoTokenizer handles apply_chat_template and batch_decode correctly for
        # text-only inference and supports enable_thinking=False for Qwen3.5.
        self._processor = AutoTokenizer.from_pretrained(self.config.base_model)
        model_kwargs: dict[str, Any] = {
            "dtype": requested_dtype,
            "low_cpu_mem_usage": True,
        }
        if target_device == "cuda":
            model_kwargs["device_map"] = "auto"
        try:
            self._model = auto_model.from_pretrained(self.config.base_model, **model_kwargs)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1455 or "paging file is too small" in str(exc).lower():
                raise RuntimeError(
                    "Windows does not have enough free RAM/page-file capacity to load Qwen. "
                    "Restart with memory-heavy apps closed and use --dtype bfloat16."
                ) from exc
            raise
        if self.adapter_path is not None:
            self._model = PeftModel.from_pretrained(self._model, str(self.adapter_path))
        if target_device != "cuda":
            self._model.to(target_device)
        self._model.eval()
        self._input_device = next(self._model.parameters()).device

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        candidate = text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1).strip()
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        # Try repairing truncated JSON strings
        for suffix in ("]}", "}]}", '"]}', '"]}}', "}", '"}', '"]'):
            try:
                repaired = json.loads(candidate + suffix)
                if isinstance(repaired, dict) and ("intent" in repaired or "question" in repaired):
                    return repaired
            except json.JSONDecodeError:
                continue

        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                if isinstance(value, dict):
                    candidates.append(value)
            except json.JSONDecodeError:
                for suffix in ("]}", "}]}", "}"):
                    try:
                        repaired = json.loads(candidate[index:] + suffix)
                        if isinstance(repaired, dict):
                            candidates.append(repaired)
                            break
                    except json.JSONDecodeError:
                        continue

        for item in candidates:
            if "intent" in item or "question" in item:
                return item
        if candidates:
            return candidates[0]
        raise ValueError("no valid JSON object found")

    def drain_traces(self) -> list[dict[str, Any]]:
        traces = self._traces
        self._traces = []
        return traces

    @staticmethod
    def _batch_decode(processor: Any, token_ids: Any) -> list[str]:
        """Decode generated ids across processor API variants.

        Some Transformers multimodal processors expose ``batch_decode`` on the
        processor, while Qwen3.5 processor versions expose it only through the
        nested tokenizer.
        """
        decoder = getattr(processor, "batch_decode", None)
        if not callable(decoder):
            decoder = getattr(getattr(processor, "tokenizer", None), "batch_decode", None)
        if not callable(decoder):
            raise RuntimeError("Loaded processor does not provide a batch decoder")
        return decoder(token_ids, skip_special_tokens=True)

    def _structured(
        self, operation: str, system: str, user: str, output_type: type[Any]
    ) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self._processor(text, return_tensors="pt").to(self._input_device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.15,
            )
        output_text = self._batch_decode(self._processor, generated[:, prompt_tokens:])[0].strip()
        try:
            parsed = self._json_object(output_text)
            result = output_type.model_validate(parsed)
            self._traces.append(
                {
                    "operation": operation,
                    "raw_output": safe_provider_value(output_text),
                    "parsed": True,
                    "error": None,
                }
            )
            return result
        except Exception as exc:
            self._traces.append(
                {
                    "operation": operation,
                    "raw_output": safe_provider_value(output_text),
                    "parsed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise RuntimeError(
                f"{self.config.variant} model returned invalid {output_type.__name__} JSON"
            ) from exc

    def _intent_recovery(
        self, message: str, state: DialogueState
    ) -> ExtractorResult | None:
        selected = select_next_slot(self.schema, state)
        if selected is None or selected.slot_id != "intent":
            return None
        normalized = re.sub(r"[^a-z0-9]+", " ", message.lower()).strip()
        if normalized in SHORT_NEGATIVE_INTENT_REPLIES or re.search(
            r"\b(?:do not|dont|don t|not)\s+(?:want\s+)?(?:a\s+)?forecast\b",
            normalized,
        ):
            return ExtractorResult(intent=Intent.NOT_FORECASTING, intent_confidence=1.0)
        forecast_request = bool(
            re.search(r"\b(?:forecast|forecasting|predict|prediction)\b", normalized)
            or re.search(r"\bforecast\s+(?:banana|karna|chahiye)\b", normalized)
        )
        if normalized not in SHORT_POSITIVE_INTENT_REPLIES and not forecast_request:
            return None
        return ExtractorResult(
            intent=Intent.CREATE_FORECAST,
            intent_confidence=1.0,
            updates=[
                SlotUpdate(
                    slot_id="intent",
                    candidate_value=Intent.CREATE_FORECAST.value,
                    status=SlotStatus.PROVIDED,
                    confidence=1.0,
                    evidence_text=message,
                )
            ],
        )

    def _slot_answer_recovery(
        self,
        message: str,
        state: DialogueState,
        result: ExtractorResult,
    ) -> ExtractorResult | None:
        """Recover a short answer to the one slot the orchestrator just requested.

        This is deliberately narrow: it never guesses a slot, never runs before
        forecasting intent is established, and rejects values that fail the
        schema's normal validation.
        """
        if state.intent != Intent.CREATE_FORECAST:
            return None

        stripped = message.strip()
        normalized_text = re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()
        answer_words = normalized_text.split()
        if (
            not stripped
            or len(stripped) > 120
            or len(stripped.split()) > 4
            or "?" in stripped
            or normalized_text in SHORT_SLOT_NON_ANSWERS
            or re.search(r"\b(?:ignore|instruction|prompt|system|assistant)\b", normalized_text)
            or (
                len(answer_words) > 1
                and re.search(
                    r"\b(?:i|we|you|my|our|will|want|need|have|is|are|the|that|this|from)\b",
                    normalized_text,
                )
            )
        ):
            return None

        selected = select_next_slot(self.schema, state)
        if selected is None or selected.slot_id in {"intent", "authentication_reference"}:
            return None
        if normalized_text in LOW_INFORMATION_SLOT_VALUES.get(selected.slot_id, set()):
            return None
        if any(update.slot_id == selected.slot_id for update in result.updates):
            return None
        definition = self.schema.get(selected.slot_id)
        if definition.value_type not in SAFE_SHORT_ANSWER_TYPES:
            return None

        normalized_value = normalize_value(definition, stripped)
        if definition.value_type == "enum" and normalized_value not in definition.allowed_values:
            return None
        candidate_state = SlotState(
            slot_id=selected.slot_id,
            value=normalized_value,
            status=SlotStatus.PROVIDED,
            confidence=1.0,
            evidence_text=stripped,
        )
        if validate_slot(definition, candidate_state):
            return None

        # A short answer follows one explicit question. Preserve only an intent
        # update and discard model updates for unrelated slots, which are more
        # likely copied examples than evidence from this answer.
        existing = [update for update in result.updates if update.slot_id == "intent"]
        return result.model_copy(
            update={
                "intent": Intent.CREATE_FORECAST,
                "intent_confidence": max(result.intent_confidence, 0.99),
                "updates": [
                    *existing,
                    SlotUpdate(
                        slot_id=selected.slot_id,
                        candidate_value=normalized_value,
                        status=SlotStatus.PROVIDED,
                        confidence=1.0,
                        evidence_text=stripped,
                    ),
                ],
            }
        )

    @staticmethod
    def _parse_duration_candidate(raw_val: Any, evidence: str, message: str) -> dict[str, Any] | None:
        if isinstance(raw_val, dict):
            periods = raw_val.get("periods", 1)
            unit = raw_val.get("unit", "month")
            try:
                p_float = float(periods)
                p_clean = int(p_float) if p_float.is_integer() else p_float
            except (ValueError, TypeError):
                p_clean = 1
            unit_clean = _DURATION_UNITS.get(str(unit).lower(), str(unit).lower())
            return {"periods": p_clean, "unit": unit_clean}

        if isinstance(raw_val, str):
            candidate_str = raw_val.strip()
            if candidate_str.startswith("{") and candidate_str.endswith("}"):
                try:
                    parsed_dict = json.loads(candidate_str)
                    if isinstance(parsed_dict, dict):
                        return TransformersPeftProvider._parse_duration_candidate(parsed_dict, evidence, message)
                except json.JSONDecodeError:
                    pass

        text_to_search = f"{evidence} {message}"
        dur_match = _DURATION_TEXT_PATTERN.search(text_to_search)
        if dur_match:
            unit = _DURATION_UNITS.get(dur_match.group("unit").lower())
            if unit:
                try:
                    num = float(raw_val) if str(raw_val).isdigit() or isinstance(raw_val, (int, float)) else float(dur_match.group("number") or 1)
                    p_clean = int(num) if num.is_integer() else num
                    return {"periods": p_clean, "unit": unit}
                except (ValueError, TypeError):
                    pass

        for text in (str(raw_val), evidence, message):
            match = _DURATION_TEXT_PATTERN.fullmatch(str(text).strip())
            if match is not None:
                unit = _DURATION_UNITS.get(match.group("unit").lower())
                if unit is not None:
                    p = float(match.group("number") or 1)
                    p_clean = int(p) if p.is_integer() else p
                    return {"periods": p_clean, "unit": unit}
        return None

    def _clean_and_normalize_updates(
        self, result: ExtractorResult, message: str
    ) -> ExtractorResult:
        cleaned_updates: list[SlotUpdate] = []
        seen_updates: set[tuple[str, str]] = set()
        msg_lower = message.lower().strip()

        for update in result.updates:
            try:
                slot_def = self.schema.get(update.slot_id)
            except KeyError:
                continue

            raw_val = update.candidate_value
            if isinstance(raw_val, str) and raw_val.strip().startswith("{") and raw_val.strip().endswith("}"):
                try:
                    parsed = json.loads(raw_val.strip())
                    if isinstance(parsed, dict):
                        raw_val = parsed
                except json.JSONDecodeError:
                    pass

            evidence = str(update.evidence_text or "").strip()
            has_prompt_leak = any(leak in evidence.lower() for leak in PROMPT_LEAK_TERMS)

            if has_prompt_leak or (len(evidence) > 2 and evidence.lower() not in msg_lower):
                if len(msg_lower) <= 3 and msg_lower not in str(raw_val).lower():
                    continue
                evidence = message.strip()

            if slot_def.value_type == "duration":
                dur = self._parse_duration_candidate(raw_val, evidence, message)
                if dur is not None:
                    raw_val = dur
                elif not isinstance(raw_val, dict):
                    continue

            key = (update.slot_id, json.dumps(raw_val, sort_keys=True, default=str))
            if key in seen_updates:
                continue
            seen_updates.add(key)

            cleaned_updates.append(
                update.model_copy(
                    update={
                        "candidate_value": raw_val,
                        "evidence_text": evidence,
                    }
                )
            )

        return result.model_copy(update={"updates": cleaned_updates})

    @staticmethod
    def _drop_low_information_updates(result: ExtractorResult) -> ExtractorResult:
        retained = []
        for update in result.updates:
            normalized = re.sub(
                r"[^a-z0-9]+", " ", str(update.candidate_value).lower()
            ).strip()
            if normalized in LOW_INFORMATION_SLOT_VALUES.get(update.slot_id, set()):
                continue
            retained.append(update)
        if len(retained) == len(result.updates):
            return result
        return result.model_copy(update={"updates": retained})

    async def extract(self, message: str, state: DialogueState) -> ExtractorResult:
        recovery = self._intent_recovery(message, state)
        normalized = re.sub(r"[^a-z0-9]+", " ", message.lower()).strip()
        if recovery is not None and normalized in (
            SHORT_POSITIVE_INTENT_REPLIES | SHORT_NEGATIVE_INTENT_REPLIES
        ):
            self._traces.append(
                {
                    "operation": "deterministic_intent",
                    "raw_output": None,
                    "parsed": True,
                    "error": None,
                }
            )
            return recovery
        if len(message.split()) <= 4 and not re.search(
            r"\b(?:actually|correction|forecast|instead|predict)\b", normalized
        ):
            direct_answer = self._slot_answer_recovery(
                message,
                state,
                ExtractorResult(
                    intent=state.intent,
                    intent_confidence=state.slots["intent"].confidence or 0.99,
                ),
            )
            if direct_answer is not None:
                self._traces.append(
                    {
                        "operation": "deterministic_slot",
                        "raw_output": None,
                        "parsed": True,
                        "error": None,
                    }
                )
                return direct_answer
        try:
            result = self._structured(
                "extract",
                build_slm_extractor_instructions(),
                build_slm_extractor_input(message, state, self.schema),
                ExtractorResult,
            )
        except RuntimeError:
            if recovery is None:
                raise
            self._traces.append(
                {"operation": "extract_recovery", "raw_output": None, "parsed": True, "error": None}
            )
            return recovery
        result = self._drop_low_information_updates(result)
        result = self._clean_and_normalize_updates(result, message)
        if recovery is not None and result.intent != recovery.intent:
            existing = [update for update in result.updates if update.slot_id != "intent"]
            result = result.model_copy(
                update={
                    "intent": recovery.intent,
                    "intent_confidence": 1.0,
                    "updates": [*recovery.updates, *existing],
                }
            )
            self._traces.append(
                {"operation": "extract_recovery", "raw_output": None, "parsed": True, "error": None}
            )
        slot_recovery = self._slot_answer_recovery(message, state, result)
        if slot_recovery is not None:
            result = slot_recovery
            self._traces.append(
                {
                    "operation": "deterministic_slot",
                    "raw_output": None,
                    "parsed": True,
                    "error": None,
                }
            )
        return result

    async def ask(self, request: QuestionRequest) -> QuestionOutput:
        return self._structured(
            "ask",
            build_slm_question_instructions(),
            build_slm_question_input(request),
            QuestionOutput,
        )
