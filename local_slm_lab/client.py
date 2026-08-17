"""Minimal OpenAI-compatible client using only the Python standard library."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class LocalModelConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "local-qwen"
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: int = 120

    @classmethod
    def from_file(cls, path: str | Path) -> "LocalModelConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**values)


class LocalModelError(RuntimeError):
    """Raised when the local inference server cannot return a valid response."""


def chat_completion(
    messages: list[dict[str, str]],
    config: LocalModelConfig,
) -> str:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }
    endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
    http_request = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=config.timeout_seconds) as response:
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LocalModelError(
            f"Could not get a valid response from the local server at {endpoint}: {exc}"
        ) from exc

    try:
        content = str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LocalModelError(f"Unexpected local-server response: {body!r}") from exc

    if not content.strip():
        raise LocalModelError(
            "The local model returned empty content. Start llama-server with "
            "'--reasoning off' for this experiment."
        )
    return content
