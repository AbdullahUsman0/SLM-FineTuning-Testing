from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from local_slm_lab.peft_provider import PeftInferenceConfig, TransformersPeftProvider
from local_slm_lab.slm_prompts import build_slm_question_input, build_slm_question_instructions
from local_slm_lab.v5_prompts import build_v5_extractor_input, build_v5_extractor_instructions
from forecasting_assistant.domain.models import DialogueState, ExtractorResult, QuestionOutput, QuestionRequest
from forecasting_assistant.domain.schema import ForecastingSchema
from forecasting_assistant.prompts.extractor import safe_provider_value


def strict_json_object(text: str) -> dict[str, Any]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Non-finite JSON number")

    value = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


def validate_raw_output(parsed: dict, output_type: type) -> Any:
    fields = set(output_type.model_fields)
    if set(parsed) != fields:
        raise ValueError("Output keys do not match the contract")
    if output_type is ExtractorResult:
        seen = set()
        for update in parsed.get("updates", []):
            if set(update) != {"slot_id", "candidate_value", "status", "confidence", "evidence_text"}:
                raise ValueError("Update keys do not match the contract")
            if update["slot_id"] in seen:
                raise ValueError("Duplicate slot update")
            seen.add(update["slot_id"])
            if not isinstance(update["candidate_value"], str):
                raise ValueError("candidate_value must be JSON-encoded text")
            json.loads(update["candidate_value"])
    return output_type.model_validate(parsed)


class V5TransformersProvider(TransformersPeftProvider):
    name = "v5_transformers_raw"

    def __init__(self, config: PeftInferenceConfig, schema: ForecastingSchema) -> None:
        if not config.revision or not re.fullmatch(r"[0-9a-f]{40}", config.revision):
            raise ValueError("V5 requires a pinned 40-character base-model revision")
        super().__init__(config, schema)
        if config.device == "cuda" and any(p.device.type != "cuda" for p in self._model.parameters()):
            raise RuntimeError("Matched CUDA evaluation must not silently offload to CPU")
        self.chat_template_sha256 = hashlib.sha256(
            self._processor.get_chat_template().encode("utf-8")
        ).hexdigest()

    def _structured(self, operation: str, system: str, user: str, output_type: type) -> Any:
        trace = {
            "operation": operation,
            "raw_output": None,
            "raw_json_valid": None,
            "raw_schema_valid": None,
            "parsed": False,
            "error": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "recovery_applied": False,
        }
        try:
            text = self._processor.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._processor(text, return_tensors="pt", add_special_tokens=False).to(self._input_device)
            prompt_tokens = inputs["input_ids"].shape[-1]
            trace["prompt_tokens"] = prompt_tokens
            self._torch.manual_seed(42)
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    repetition_penalty=1.0,
                )
            trace["completion_tokens"] = generated.shape[-1] - prompt_tokens
            trace["hit_generation_limit"] = trace["completion_tokens"] == self.config.max_new_tokens
            raw = self._batch_decode(self._processor, generated[:, prompt_tokens:])[0].strip()
            trace["raw_output"] = safe_provider_value(raw)
            trace["raw_json_valid"] = False
            trace["raw_schema_valid"] = False
            parsed = strict_json_object(raw)
            trace["raw_json_valid"] = True
            result = validate_raw_output(parsed, output_type)
            trace["raw_schema_valid"] = True
            trace["parsed"] = True
            return result
        except Exception as error:
            trace["error"] = type(error).__name__
            raise RuntimeError(f"V5 {operation} failed ({type(error).__name__})") from None
        finally:
            self._traces.append(trace)

    async def extract(self, message: str, state: DialogueState) -> ExtractorResult:
        return self._structured(
            "extract",
            build_v5_extractor_instructions(),
            build_v5_extractor_input(message, state, self.schema),
            ExtractorResult,
        )

    async def ask(self, request: QuestionRequest) -> QuestionOutput:
        return self._structured(
            "ask", build_slm_question_instructions(), build_slm_question_input(request), QuestionOutput
        )
