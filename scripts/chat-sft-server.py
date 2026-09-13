"""Interactive FYP elicitation chat through a fine-tuned llama.cpp server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
for path in (PROJECT_ROOT, FPY_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting_assistant.domain.schema import load_schema  # noqa: E402
from local_slm_lab.client import LocalModelConfig  # noqa: E402
from local_slm_lab.memory_repository import MemoryRepository  # noqa: E402
from local_slm_lab.providers import LocalFineTunedProvider  # noqa: E402
from local_slm_lab.slm_engine import LocalSLMElicitationEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.v2-evaluation.json")
    parser.add_argument("--transcript")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    schema = load_schema()
    config = LocalModelConfig.from_file(PROJECT_ROOT / args.config)
    provider = LocalFineTunedProvider(config, schema)
    engine = LocalSLMElicitationEngine(schema, provider, MemoryRepository())
    state = engine.start_dialogue()
    transcript: list[dict[str, object]] = []

    print("Commands: /state shows collected slots; /confirm confirms; /quit exits.")
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
        else PROJECT_ROOT / "results" / f"sft-server-chat-{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "model": config.model,
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
