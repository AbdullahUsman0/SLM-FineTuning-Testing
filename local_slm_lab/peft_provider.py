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
    SlotUpdate,
)
from forecasting_assistant.domain.schema import ForecastingSchema  # noqa: E402
from forecasting_assistant.application.clarification import select_next_slot  # noqa: E402
from forecasting_assistant.prompts.extractor import safe_provider_value  # noqa: E402
from local_slm_lab.slm_prompts import (  # noqa: E402
    build_slm_extractor_input,
    build_slm_extractor_instructions,
    build_slm_question_input,
    build_slm_question_instructions,
)


RUN_ROOT = PROJECT_ROOT / "training-runs" / "qwen35-08b-lora-v1"
VARIANT_ADAPTERS = {
    "base": None,
    "best": RUN_ROOT / "best-adapter",
    "final": RUN_ROOT / "checkpoint-68",
}

SHORT_POSITIVE_INTENT_REPLIES = {
    "yes", "yeah", "yep", "y", "haan", "han", "jee", "ji", "bilkul"
}
SHORT_NEGATIVE_INTENT_REPLIES = {
    "no", "nope", "n", "nahi", "nahin", "na", "do not", "dont", "not forecasting"
}


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
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "PEFT inference dependencies are missing. Install "
                "training/requirements-inference.txt in a dedicated environment."
            ) from exc

        auto_model = getattr(transformers, "AutoModelForImageTextToText", None)
        if auto_model is None:
            auto_model = getattr(transformers, "AutoModelForVision2Seq", None)
        if auto_model is None:
            raise RuntimeError("Installed Transformers cannot load Qwen3.5 image-text models")

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
        self._processor = AutoProcessor.from_pretrained(self.config.base_model)
        model_kwargs: dict[str, Any] = {"dtype": requested_dtype}
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

        decoder = json.JSONDecoder()
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("no valid JSON object found")

    def drain_traces(self) -> list[dict[str, Any]]:
        traces = self._traces
        self._traces = []
        return traces

    def _structured(
        self, operation: str, system: str, user: str, output_type: type[Any]
    ) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            inputs = self._processor.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            inputs = self._processor.apply_chat_template(messages, **template_kwargs)
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        inputs = {key: value.to(self._input_device) for key, value in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        text = self._processor.batch_decode(
            generated[:, prompt_tokens:], skip_special_tokens=True
        )[0].strip()
        try:
            parsed = self._json_object(text)
            result = output_type.model_validate(parsed)
            self._traces.append(
                {
                    "operation": operation,
                    "raw_output": safe_provider_value(text),
                    "parsed": True,
                    "error": None,
                }
            )
            return result
        except Exception as exc:
            self._traces.append(
                {
                    "operation": operation,
                    "raw_output": safe_provider_value(text),
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
        return result

    async def ask(self, request: QuestionRequest) -> QuestionOutput:
        return self._structured(
            "ask",
            build_slm_question_instructions(),
            build_slm_question_input(request),
            QuestionOutput,
        )
