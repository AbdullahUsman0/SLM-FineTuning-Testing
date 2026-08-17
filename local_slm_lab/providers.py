"""Structured local and production OpenAI providers for the common benchmark."""

from __future__ import annotations

import asyncio
import json
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
from forecasting_assistant.prompts.extractor import build_extractor_input, build_extractor_instructions
from forecasting_assistant.prompts.llmrei_long import build_question_input, build_question_instructions

from .client import LocalModelConfig


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
