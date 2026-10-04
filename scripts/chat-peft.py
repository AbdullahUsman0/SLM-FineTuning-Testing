"""Interactive questioning/cross-questioning session for the isolated PEFT model."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
PEFT_DEPS = PROJECT_ROOT / ".peft-deps"
for path in (PROJECT_ROOT, FPY_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting_assistant.domain.schema import load_schema  # noqa: E402
from local_slm_lab.memory_repository import MemoryRepository  # noqa: E402
from local_slm_lab.peft_provider import (  # noqa: E402
    PeftInferenceConfig,
    TransformersPeftProvider,
    resolve_adapter_path,
)
from local_slm_lab.slm_engine import LocalSLMElicitationEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "best", "final", "custom"), default="best")
    parser.add_argument("--adapter", help="Adapter directory for --variant custom")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", help="Pinned base-model revision (required for v5/v6 prompts)")
    parser.add_argument("--prompt-version", choices=("legacy", "v5"), default="legacy",
                        help="Use v5 for adapters trained on the v5 or v6 corpus")
    parser.add_argument("--adapter-manifest", help="Completion JSON containing the expected adapter hashes")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--transcript", help="Output JSON path; defaults under results/")
    parser.add_argument("--debug", action="store_true", help="Print raw structured-model traces")
    args = parser.parse_args()
    if args.prompt_version == "v5" and not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        parser.error("--prompt-version v5 requires --revision with the pinned 40-character model commit")

    # Add the optional inference runtime only after application dependencies
    # are imported; this prevents it from shadowing the main pipeline stack.
    if PEFT_DEPS.is_dir() and str(PEFT_DEPS) not in sys.path:
        sys.path.append(str(PEFT_DEPS))

    adapter_path = resolve_adapter_path(args.variant, args.adapter)
    if args.adapter_manifest:
        if adapter_path is None:
            parser.error("--adapter-manifest requires an adapter")
        expected = json.loads(Path(args.adapter_manifest).read_text(encoding="utf-8"))["adapter"]
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            path = adapter_path / name
            if not path.is_file():
                parser.error(f"Missing adapter file: {path}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected[name]["sha256"] or path.stat().st_size != expected[name]["bytes"]:
                parser.error(f"Adapter checksum or size mismatch: {path}")
        print("Adapter weights and config match the completion record.")
    provider_type = TransformersPeftProvider
    if args.prompt_version == "v5":
        from local_slm_lab.v5_provider import V5TransformersProvider
        provider_type = V5TransformersProvider
    schema = load_schema()
    print(f"Loading {args.variant} variant; this can take several minutes on CPU...")
    provider = provider_type(
        PeftInferenceConfig(
            base_model=args.base_model,
            variant=args.variant,
            adapter_path=adapter_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            revision=args.revision,
        ),
        schema,
    )
    repository = MemoryRepository()
    engine = LocalSLMElicitationEngine(schema, provider, repository)
    state = engine.start_dialogue()
    transcript: list[dict[str, Any]] = []

    print("Commands: /state shows collected slots; /reset starts a new case; /confirm confirms when ready; /quit exits.")
    print("Clarification questions use the schema's fixed wording; --debug shows the model's extraction.")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/reset":
            state = engine.start_dialogue()
            transcript.append({"event": "reset", "dialogue_id": str(state.dialogue_id)})
            print("Started a new dialogue with no previous test context.")
            continue
        if message == "/state":
            current = engine.get_state(state.dialogue_id)
            populated = {
                key: value.model_dump(mode="json")
                for key, value in current.slots.items()
                if value.value is not None
            }
            print(json.dumps(populated, indent=2, ensure_ascii=False))
            continue
        if message == "/confirm":
            try:
                specification = engine.confirm_specification(state.dialogue_id, confirm=True)
                print(json.dumps(specification.model_dump(mode="json"), indent=2))
            except ValueError as exc:
                print(f"not ready: {exc}")
            continue

        result = asyncio.run(engine.handle_user_message(state.dialogue_id, message))
        print(f"model> {result.assistant_message}")
        traces = provider.drain_traces()
        if args.debug:
            print(json.dumps({"debug_traces": traces}, indent=2, ensure_ascii=False))
        transcript.append(
            {
                "user": message,
                "assistant": result.assistant_message,
                "ready": result.readiness.ready,
                "unresolved_slots": result.readiness.unresolved_slots,
                "traces": traces,
            }
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.transcript).resolve()
        if args.transcript
        else PROJECT_ROOT / "results" / f"peft-chat-{args.variant}-{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "variant": args.variant,
                "base_model": args.base_model,
                "revision": args.revision,
                "prompt_version": args.prompt_version,
                "adapter": None if adapter_path is None else str(adapter_path),
                "turns": transcript,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Transcript saved to {output}")


if __name__ == "__main__":
    main()
