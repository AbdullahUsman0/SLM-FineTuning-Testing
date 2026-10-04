"""V5/v6 extraction prompts and strict parsing for local quantized manual tests."""

from __future__ import annotations

import json
from urllib import request
from urllib.parse import urlparse

from local_slm_lab.v5_provider import V5TransformersProvider


class V5LocalServerProvider(V5TransformersProvider):
    name = "v5_local_quantized_manual"

    def __init__(self, config, schema, *, base_url: str, model: str) -> None:
        # No Transformers model is loaded: llama.cpp owns the quantized model.
        parsed = urlparse(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path.rstrip("/") != "/v1"):
            raise ValueError("Manual server URL must be a loopback HTTP /v1 endpoint")
        self.config = config
        self.schema = schema
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._traces = []

    def _generate_raw(self, system, user, *, logits_processor=None):
        if logits_processor is not None:
            raise ValueError("Manual llama.cpp transport does not implement HF logits processors")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "top_k": 1,
            "repeat_penalty": 1.0,
            "seed": 42,
            "max_tokens": self.config.max_new_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        http_request = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=900) as response:
            body = json.load(response)
        choice = body["choices"][0]
        raw = choice["message"]["content"]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Local server returned no extraction text")
        usage = body.get("usage", {})
        return (raw.strip(), usage.get("prompt_tokens"), usage.get("completion_tokens"),
                choice.get("finish_reason") == "length")
