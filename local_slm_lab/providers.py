"""Structured local and production OpenAI providers for the common benchmark."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
if str(FPY_SRC) not in sys.path:
    sys.path.insert(0, str(FPY_SRC))

from forecasting_assistant.domain.models import (
    DialogueState,
    ExtractorResult,
    QuestionOutput,
    QuestionRequest,
)
from forecasting_assistant.domain.schema import ForecastingSchema
from forecasting_assistant.prompts.extractor import (
    build_extractor_input,
    build_extractor_instructions,
    safe_provider_value,
)
from forecasting_assistant.prompts.llmrei_long import build_question_input, build_question_instructions

from .client import LocalModelConfig, chat_completion
from .peft_provider import (
    SHORT_NEGATIVE_INTENT_REPLIES,
    SHORT_POSITIVE_INTENT_REPLIES,
    TransformersPeftProvider,
)
from .slm_prompts import (
    build_slm_extractor_input,
    build_slm_extractor_instructions,
    build_slm_question_input,
    build_slm_question_instructions,
)


class LocalStructuredProvider:
    name = "local_llama_cpp"

    def __init__(self, config: LocalModelConfig, schema: ForecastingSchema) -> None:
        self.config = config
        self.schema = schema
        self.model = config.model

    def _complete(self, system: str, user: str, output_type: type[Any]) -> Any:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_type.__name__,
                    "strict": True,
                    "schema": output_type.model_json_schema(),
                },
            },
        }
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        http_request = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return output_type.model_validate_json(content)
        except (error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
            raise RuntimeError(f"local structured request failed: {exc}") from exc

    async def extract(self, message: str, state: DialogueState) -> ExtractorResult:
        return await asyncio.to_thread(
            self._complete,
            build_extractor_instructions(),
            build_extractor_input(message, state, self.schema),
            ExtractorResult,
        )

    async def ask(self, request_value: QuestionRequest) -> QuestionOutput:
        return await asyncio.to_thread(
            self._complete,
            build_question_instructions(),
            build_question_input(request_value),
            QuestionOutput,
        )


class LocalFineTunedProvider(LocalStructuredProvider):
    """Run a fine-tuned GGUF LoRA through the prompts used during SFT.

    The HTTP transport remains llama.cpp's OpenAI-compatible endpoint, while
    extraction, question generation, and deterministic intent recovery match
    ``TransformersPeftProvider``. This keeps GGUF and native-PEFT evaluation
    behavior comparable without loading the full Transformers model on Windows.
    """

    name = "local_llama_cpp_sft"

    def __init__(self, config: LocalModelConfig, schema: ForecastingSchema) -> None:
        super().__init__(config, schema)
        self._traces: list[dict[str, Any]] = []

    _intent_recovery = TransformersPeftProvider._intent_recovery
    _slot_answer_recovery = TransformersPeftProvider._slot_answer_recovery
    _drop_low_information_updates = staticmethod(
        TransformersPeftProvider._drop_low_information_updates
    )

    def drain_traces(self) -> list[dict[str, Any]]:
        traces = self._traces
        self._traces = []
        return traces

    def _structured(
        self, operation: str, system: str, user: str, output_type: type[Any]
    ) -> Any:
        try:
            text = self._raw_completion(system, user)
            parsed = TransformersPeftProvider._json_object(text)
            result = output_type.model_validate(parsed)
        except Exception as exc:
            self._traces.append(
                {
                    "operation": operation,
                    "raw_output": safe_provider_value(locals().get("text")),
                    "parsed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise RuntimeError(
                f"fine-tuned local model returned invalid {output_type.__name__} JSON"
            ) from exc
        self._traces.append(
            {
                "operation": operation,
                "raw_output": safe_provider_value(text),
                "parsed": True,
                "error": None,
            }
        )
        return result

    def _raw_completion(self, system: str, user: str) -> str:
        return chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            self.config,
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
            result = await asyncio.to_thread(
                self._structured,
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

    async def ask(self, request_value: QuestionRequest) -> QuestionOutput:
        return await asyncio.to_thread(
            self._structured,
            "ask",
            build_slm_question_instructions(),
            build_slm_question_input(request_value),
            QuestionOutput,
        )


class OpenAIProductionProvider:
    name = "openai_production_client"

    def __init__(self, api_key: str, model: str, schema: ForecastingSchema) -> None:
        from forecasting_assistant.infrastructure.llm.openai_responses import (
            OpenAIResponsesClient,
        )

        self.model = model
        self._client = OpenAIResponsesClient(api_key=api_key, model=model, schema=schema)

    async def extract(self, message: str, state: DialogueState) -> ExtractorResult:
        try:
            return await self._client.extract(message, state)
        except Exception as exc:
            cause = exc.__cause__
            detail = _safe_openai_error(cause or exc)
            raise RuntimeError(f"OpenAI extraction failed: {detail}") from exc

    async def ask(self, request_value: QuestionRequest) -> QuestionOutput:
        try:
            return await self._client.ask(request_value)
        except Exception as exc:
            cause = exc.__cause__
            detail = _safe_openai_error(cause or exc)
            raise RuntimeError(f"OpenAI question generation failed: {detail}") from exc


def _safe_openai_error(exc: BaseException) -> str:
    """Keep diagnostic codes while excluding request bodies and credential fragments."""
    parts = [type(exc).__name__]
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(f"status={status_code}")
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error_code = body.get("code")
        if error_code:
            parts.append(f"code={error_code}")
    return " ".join(parts)


def read_openai_credentials(env_path: Path) -> tuple[str, str]:
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip().strip('"').strip("'")
    api_key = values.get("openai_api_key")
    model = values.get("openai_model")
    if not api_key or not model:
        raise ValueError(f"OPENAI_API_KEY and OPENAI_MODEL are required in {env_path}")
    return api_key, model
