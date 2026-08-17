"""Interactive questioning/cross-questioning session for the isolated PEFT model."""

from __future__ import annotations

import argparse
import asyncio
import json
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--transcript", help="Output JSON path; defaults under results/")
    parser.add_argument("--debug", action="store_true", help="Print raw structured-model traces")
    args = parser.parse_args()

    # Add the optional inference runtime only after application dependencies
    # are imported; this prevents it from shadowing the main pipeline stack.
    if PEFT_DEPS.is_dir() and str(PEFT_DEPS) not in sys.path:
        sys.path.append(str(PEFT_DEPS))

    adapter_path = resolve_adapter_path(args.variant, args.adapter)
    schema = load_schema()
    print(f"Loading {args.variant} variant; this can take several minutes on CPU...")
    provider = TransformersPeftProvider(
        PeftInferenceConfig(
            base_model=args.base_model,
            variant=args.variant,
            adapter_path=adapter_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        ),
        schema,
    )
    repository = MemoryRepository()
    engine = LocalSLMElicitationEngine(schema, provider, repository)
    state = engine.start_dialogue()
    transcript: list[dict[str, Any]] = []

    print("Commands: /state shows collected slots; /confirm confirms when ready; /quit exits.")
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
