"""Research-only providers for the v5 structured-output comparison.

These providers do not alter the production elicitation engine.  They share the
same v5 prompts, model, adapter, tokenizer, and greedy generation path whenever
the experimental question requires those controls (arms A-C).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[1]
FPY = ROOT.parent / "fpy"
if str(FPY / "src") not in sys.path:
    sys.path.insert(0, str(FPY / "src"))

from forecasting_assistant.domain.models import (
    DialogueState,
    ExtractorResult,
    Intent,
    QuestionOutput,
    QuestionRequest,
    SlotStatus,
)
from forecasting_assistant.domain.schema import ForecastingSchema
from forecasting_assistant.prompts.extractor import safe_provider_value

from local_slm_lab.peft_provider import PeftInferenceConfig
from local_slm_lab.slm_prompts import build_slm_question_input, build_slm_question_instructions
from local_slm_lab.v5_prompts import build_v5_extractor_input, build_v5_extractor_instructions
from local_slm_lab.v5_provider import V5TransformersProvider, strict_json_object, validate_raw_output


def _wire_result(result: ExtractorResult) -> dict[str, Any]:
    value = result.model_dump(mode="json")
    for update in value["updates"]:
        update["candidate_value"] = json.dumps(
            update["candidate_value"], ensure_ascii=False, separators=(",", ":")
        )
    return value


def openai_compatible_schema(value: Any) -> Any:
    """Remove annotation keywords outside the documented Structured Outputs subset."""
    if isinstance(value, dict):
        return {key: openai_compatible_schema(child) for key, child in value.items()
                if key not in {"$schema", "$comment", "default"}}
    if isinstance(value, list):
        return [openai_compatible_schema(child) for child in value]
    return value


def _error_code(error: BaseException) -> str:
    if isinstance(error, ValidationError):
        parts = []
        for item in error.errors(include_url=False, include_input=False):
            location = ".".join(str(part) for part in item.get("loc", ())) or "root"
            parts.append(f"{location}:{item.get('type', 'validation_error')}")
        return "pydantic:" + ",".join(sorted(parts))[:800]
    message = str(error).lower()
    for needle, code in (
        ("duplicate json key", "duplicate_json_key"),
        ("non-finite", "non_finite_number"),
        ("expected one json object", "not_json_object"),
        ("output keys", "wrong_top_level_keys"),
        ("update keys", "wrong_update_keys"),
        ("duplicate slot", "duplicate_slot_update"),
        ("json-encoded", "candidate_not_json_text"),
    ):
        if needle in message:
            return code
    return type(error).__name__


def _attempt(
    provider: V5TransformersProvider,
    system: str,
    user: str,
    output_type: type,
    *,
    logits_processor: list[Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    started = perf_counter()
    trace: dict[str, Any] = {
        "raw_output": None,
        "raw_json_valid": False,
        "raw_schema_valid": False,
        "parsed": False,
        "error": None,
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    try:
        raw, prompt_tokens, completion_tokens, hit_limit = provider._generate_raw(
            system, user, logits_processor=logits_processor
        )
        trace.update(
            raw_output=safe_provider_value(raw),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            hit_generation_limit=hit_limit,
        )
        parsed = strict_json_object(raw)
        trace["raw_json_valid"] = True
        result = validate_raw_output(parsed, output_type)
        trace["raw_schema_valid"] = True
        trace["parsed"] = True
        return result, trace
    except Exception as error:
        trace["error"] = _error_code(error)
        return None, trace
    finally:
        trace["latency_ms"] = (perf_counter() - started) * 1000


def _combined_trace(operation: str, attempts: list[dict[str, Any]], result: Any | None, **extra: Any) -> dict[str, Any]:
    final = attempts[-1]
    trace = {
        "operation": operation,
        "raw_output": final.get("raw_output"),
        "raw_json_valid": final.get("raw_json_valid"),
        "raw_schema_valid": final.get("raw_schema_valid"),
        "parsed": result is not None,
        "error": None if result is not None else final.get("error"),
        "prompt_tokens": sum(item.get("prompt_tokens") or 0 for item in attempts),
        "completion_tokens": sum(item.get("completion_tokens") or 0 for item in attempts),
        "model_calls": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "retry_success": bool(len(attempts) > 1 and result is not None),
        "first_pass_raw_output": attempts[0].get("raw_output"),
        "first_pass_json_valid": attempts[0].get("raw_json_valid"),
        "first_pass_schema_valid": attempts[0].get("raw_schema_valid"),
        "attempts": attempts,
    }
    trace.update(extra)
    return trace


class V5RetryTransformersProvider(V5TransformersProvider):
    """Arm B: exact Arm-A first attempt followed by at most one repair request."""

    name = "v5_transformers_pydantic_retry_one"

    def _structured(self, operation: str, system: str, user: str, output_type: type) -> Any:
        first, first_trace = _attempt(self, system, user, output_type)
        attempts = [first_trace]
        if first is not None:
            self._traces.append(_combined_trace(operation, attempts, first))
            return first

        repair_user = (
            user
            + "\n\nVALIDATION_RETRY (one attempt only):\n"
            + f"The previous response failed with code {_error_code_from_trace(first_trace)}. "
            + "Return a corrected JSON object only. Preserve facts from current_message and do not add new facts.\n"
            + "previous_response="
            + json.dumps(first_trace.get("raw_output"), ensure_ascii=False)
        )
        second, second_trace = _attempt(self, system, repair_user, output_type)
        attempts.append(second_trace)
        self._traces.append(_combined_trace(operation, attempts, second))
        if second is None:
            raise RuntimeError(f"V5 {operation} failed after one validation retry")
        return second


def _error_code_from_trace(trace: dict[str, Any]) -> str:
    value = trace.get("error")
    return value if isinstance(value, str) and value else "validation_error"


class V5XGrammarTransformersProvider(V5TransformersProvider):
    """Arm C: strict exported JSON Schema enforced by XGrammar 0.2.7."""

    name = "v5_transformers_xgrammar_json_schema"

    def __init__(
        self,
        config: PeftInferenceConfig,
        schema: ForecastingSchema,
        output_schemas: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__(config, schema)
        try:
            import xgrammar as xgr
        except ImportError as exc:
            raise RuntimeError("Arm C requires xgrammar==0.2.7") from exc
        started = perf_counter()
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self._processor, vocab_size=self._model.config.vocab_size
        )
        compiler = xgr.GrammarCompiler(tokenizer_info)
        self._xgr = xgr
        self._compiled = {
            ExtractorResult: compiler.compile_json_schema(json.dumps(output_schemas["extract"])),
            QuestionOutput: compiler.compile_json_schema(json.dumps(output_schemas["ask"])),
        }
        self.schema_compile_ms = (perf_counter() - started) * 1000

    def _structured(self, operation: str, system: str, user: str, output_type: type) -> Any:
        logits = self._xgr.contrib.hf.LogitsProcessor(self._compiled[output_type])
        result, attempt = _attempt(self, system, user, output_type, logits_processor=[logits])
        self._traces.append(_combined_trace(
            operation,
            [attempt],
            result,
            constraint_backend="xgrammar==0.2.7",
            schema_compile_ms=self.schema_compile_ms,
        ))
        if result is None:
            raise RuntimeError(f"V5 constrained {operation} failed")
        return result


class ControlProbe(BaseModel):
    intent: Intent
    intent_confidence: float = Field(ge=0, le=1)
    correction_detected: bool
    unsupported_claims: list[str]


class SlotProbe(BaseModel):
    mentioned: bool
    candidate_value: str
    status: SlotStatus
    confidence: float = Field(ge=0, le=1)
    evidence_text: str

    @model_validator(mode="after")
    def consistent_absence(self) -> "SlotProbe":
        if not self.mentioned and (
            self.status != SlotStatus.UNMENTIONED
            or self.evidence_text
            or self.candidate_value != "null"
        ):
            raise ValueError("unmentioned probe must use null, unmentioned, and empty evidence")
        if self.mentioned:
            json.loads(self.candidate_value)
            if not self.evidence_text:
                raise ValueError("mentioned probe requires evidence")
        return self


class V5SlotWiseTransformersProvider(V5TransformersProvider):
    """Arm D: one control call plus one binary extraction probe per non-intent slot."""

    name = "v5_transformers_slot_wise_qa"

    CONTROL_SYSTEM = (
        "Classify only the forecasting intent and conversation controls. Return one JSON object with exactly "
        "intent, intent_confidence, correction_detected, unsupported_claims. Use only the supplied current message."
    )
    SLOT_SYSTEM = (
        "Decide whether the current user message explicitly supplies the one requested forecasting slot. "
        "Return one JSON object with exactly mentioned, candidate_value, status, confidence, evidence_text. "
        "candidate_value must be JSON-encoded text. If absent return false, \"null\", unmentioned, 0, and empty evidence. "
        "For a mentioned value, evidence_text must be a verbatim nonempty substring of current_message."
    )

    def _probe(self, system: str, user: str, output_type: type) -> tuple[Any | None, dict[str, Any]]:
        return _attempt(self, system, user, output_type)

    async def extract(self, message: str, state: DialogueState) -> ExtractorResult:
        shared = build_v5_extractor_input(message, state, self.schema)
        attempts: list[dict[str, Any]] = []
        control, trace = self._probe(self.CONTROL_SYSTEM, shared, ControlProbe)
        attempts.append(trace)
        if control is None:
            self._traces.append(_combined_trace("extract", attempts, None, strategy="slot_wise"))
            raise RuntimeError("slot-wise control probe failed")

        updates = []
        for definition in self.schema.slots:
            if definition.slot_id == "intent":
                continue
            request = json.dumps({
                "slot": {
                    "slot_id": definition.slot_id,
                    "description": definition.description,
                    "allowed_values": list(definition.allowed_values),
                },
                "v5_context": json.loads(shared),
            }, ensure_ascii=False, sort_keys=True)
            probe, trace = self._probe(self.SLOT_SYSTEM, request, SlotProbe)
            trace["slot_id"] = definition.slot_id
            attempts.append(trace)
            if probe is None:
                self._traces.append(_combined_trace("extract", attempts, None, strategy="slot_wise"))
                raise RuntimeError(f"slot-wise probe failed for {definition.slot_id}")
            if probe.mentioned:
                updates.append({
                    "slot_id": definition.slot_id,
                    "candidate_value": probe.candidate_value,
                    "status": probe.status,
                    "confidence": probe.confidence,
                    "evidence_text": probe.evidence_text,
                })

        result = ExtractorResult(
            intent=control.intent,
            intent_confidence=control.intent_confidence,
            updates=updates,
            correction_detected=control.correction_detected,
            unsupported_claims=control.unsupported_claims,
        )
        combined = _combined_trace("extract", attempts, result, strategy="slot_wise")
        combined.update(
            raw_output=json.dumps(_wire_result(result), ensure_ascii=False),
            raw_json_valid=True,
            raw_schema_valid=True,
        )
        self._traces.append(combined)
        return result

    async def ask(self, request: QuestionRequest) -> QuestionOutput:
        result, attempt = self._probe(
            build_slm_question_instructions(), build_slm_question_input(request), QuestionOutput
        )
        self._traces.append(_combined_trace("ask", [attempt], result, strategy="slot_wise"))
        if result is None:
            raise RuntimeError("slot-wise question generation failed")
        return result


class V5RuleProvider:
    """Arm E: deliberately conservative rule-only partial-coverage baseline."""

    name = "v5_deterministic_rules"
    model = "deterministic-rules-v1"

    DURATION_UNITS = {
        "hour": "hour", "hours": "hour", "day": "day", "days": "day",
        "week": "week", "weeks": "week", "month": "month", "months": "month",
        "quarter": "quarter", "quarters": "quarter", "year": "year", "years": "year",
    }
    FREQUENCIES = {
        "hourly": {"periods": 1, "unit": "hour"},
        "daily": {"periods": 1, "unit": "day"},
        "weekly": {"periods": 1, "unit": "week"},
        "monthly": {"periods": 1, "unit": "month"},
        "quarterly": {"periods": 1, "unit": "quarter"},
        "yearly": {"periods": 1, "unit": "year"},
    }

    def __init__(self, schema: ForecastingSchema) -> None:
        self.schema = schema
        self._traces: list[dict[str, Any]] = []

    def drain_traces(self) -> list[dict[str, Any]]:
        traces, self._traces = self._traces, []
        return traces

    @staticmethod
    def _update(slot_id: str, value: Any, evidence: str, confidence: float = 1.0) -> dict[str, Any]:
        return {"slot_id": slot_id, "candidate_value": value, "status": "provided",
                "confidence": confidence, "evidence_text": evidence}

    async def extract(self, message: str, state: DialogueState) -> ExtractorResult:
        lower = message.lower()
        updates: list[dict[str, Any]] = []
        covered = set()

        for word, value in self.FREQUENCIES.items():
            match = re.search(rf"\b{word}\b", lower)
            if match:
                updates.append(self._update("frequency", value, message[match.start():match.end()]))
                covered.add("frequency")
                break

        horizon = re.search(r"\b(?:next|for)\s+(\d+)\s+(hours?|days?|weeks?|months?|quarters?|years?)\b", lower)
        if horizon:
            updates.append(self._update("forecast_horizon", {
                "periods": int(horizon.group(1)), "unit": self.DURATION_UNITS[horizon.group(2)]
            }, message[horizon.start():horizon.end()]))
            covered.add("forecast_horizon")

        source_patterns = ((r"\bupload(?:ing)?\b|\b(?:csv|xlsx?|excel)\s+file\b", "upload"),
                           (r"\b(?:rest\s+)?api\b", "api"),
                           (r"\b(?:sql\s+)?database\b", "database"),
                           (r"\bdata\s+catalog\b", "catalog"))
        for pattern, value in source_patterns:
            match = re.search(pattern, lower)
            if match:
                updates.append(self._update("source_mode", value, message[match.start():match.end()]))
                covered.add("source_mode")
                break

        filename = re.search(r"\b[^\s/\\]+\.(?:csv|xlsx?|parquet|json)\b", message, re.IGNORECASE)
        if filename:
            updates.append(self._update("source_reference", filename.group(0), filename.group(0)))
            covered.add("source_reference")

        for token, value in (("json", "json"), ("csv", "csv"), ("excel", "xlsx"), ("xlsx", "xlsx")):
            match = re.search(rf"\b(?:output|return|deliver)(?:ed)?\s+(?:as|in)\s+{token}\b", lower)
            if match:
                updates.append(self._update("output_format", value, message[match.start():match.end()]))
                covered.add("output_format")
                break

        for token in ("mae", "rmse", "mase", "smape"):
            match = re.search(rf"\b{token}\b", lower)
            if match:
                updates.append(self._update("primary_metric", token, message[match.start():match.end()]))
                covered.add("primary_metric")
                break

        folds = re.search(r"\b(\d+)\s+(?:backtest|validation)\s+(?:folds|windows)\b", lower)
        if folds:
            updates.append(self._update("backtest_folds", int(folds.group(1)), message[folds.start():folds.end()]))
            covered.add("backtest_folds")

        forecast_type = None
        match = re.search(r"\bprediction intervals?\b|\bprobabilistic forecast", lower)
        if match:
            forecast_type = "probabilistic"
        else:
            match = re.search(r"\bpoint forecast", lower)
            if match:
                forecast_type = "point"
        if forecast_type and match:
            updates.append(self._update("forecast_type", forecast_type, message[match.start():match.end()]))
            covered.add("forecast_type")

        approval = re.search(r"\b(?:require|needs?|with)\s+human approval\b", lower)
        if approval:
            updates.append(self._update("human_approval_required", True, message[approval.start():approval.end()]))
            covered.add("human_approval_required")

        forecast_signal = re.search(r"\b(?:forecast|predict|projection)\b", lower)
        intent = Intent.CREATE_FORECAST if forecast_signal else state.intent
        correction = bool(re.search(r"\b(?:actually|instead|correction|change that)\b", lower))
        result = ExtractorResult(intent=intent, intent_confidence=1.0 if forecast_signal else .5,
                                 updates=updates, correction_detected=correction)
        raw = json.dumps(_wire_result(result), ensure_ascii=False)
        self._traces.append({
            "operation": "extract", "raw_output": raw, "raw_json_valid": True,
            "raw_schema_valid": True, "parsed": True, "error": None,
            "prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0,
            "rule_covered_slots": sorted(covered), "strategy": "deterministic_rules_v1",
        })
        return result

    async def ask(self, request: QuestionRequest) -> QuestionOutput:
        result = QuestionOutput(question=request.static_question)
        raw = result.model_dump_json()
        self._traces.append({"operation": "ask", "raw_output": raw, "raw_json_valid": True,
                             "raw_schema_valid": True, "parsed": True, "error": None,
                             "prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0,
                             "strategy": "static_schema_question"})
        return result


class V5OpenAIStructuredProvider:
    """Arm F: OpenAI Responses Structured Outputs with the exact v5 prompts."""

    name = "v5_openai_structured_outputs"

    def __init__(
        self,
        api_key: str,
        model: str,
        schema: ForecastingSchema,
        output_schemas: dict[str, dict[str, Any]],
        *,
        client: Any | None = None,
        max_output_tokens: int = 1024,
    ) -> None:
        if not model.strip():
            raise ValueError("OpenAI model snapshot must be explicit")
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)
        self._client = client
        self.model = model
        self.schema = schema
        self.output_schemas = output_schemas
        self.max_output_tokens = max_output_tokens
        self._traces: list[dict[str, Any]] = []

    def drain_traces(self) -> list[dict[str, Any]]:
        traces, self._traces = self._traces, []
        return traces

    async def _structured(self, operation: str, system: str, user: str, output_type: type) -> Any:
        task = "extract" if output_type is ExtractorResult else "ask"
        trace: dict[str, Any] = {"operation": operation, "raw_output": None, "parsed": False,
                                "raw_json_valid": False, "raw_schema_valid": False, "error": None,
                                "model_calls": 1, "constraint_backend": "openai_structured_outputs"}
        try:
            response = await self._client.responses.create(
                model=self.model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                text={"format": {"type": "json_schema", "name": f"v5_{task}", "strict": True,
                                 "schema": openai_compatible_schema(self.output_schemas[task])}},
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
            raw = response.output_text
            trace["raw_output"] = safe_provider_value(raw)
            parsed = strict_json_object(raw)
            trace["raw_json_valid"] = True
            result = validate_raw_output(parsed, output_type)
            trace["raw_schema_valid"] = True
            trace["parsed"] = True
            usage = getattr(response, "usage", None)
            trace["prompt_tokens"] = getattr(usage, "input_tokens", None)
            trace["completion_tokens"] = getattr(usage, "output_tokens", None)
            return result
        except Exception as error:
            trace["error"] = _error_code(error)
            raise RuntimeError(f"OpenAI v5 {operation} failed ({type(error).__name__})") from None
        finally:
            self._traces.append(trace)

    async def extract(self, message: str, state: DialogueState) -> ExtractorResult:
        return await self._structured("extract", build_v5_extractor_instructions(),
                                      build_v5_extractor_input(message, state, self.schema), ExtractorResult)

    async def ask(self, request: QuestionRequest) -> QuestionOutput:
        return await self._structured("ask", build_slm_question_instructions(),
                                      build_slm_question_input(request), QuestionOutput)
