"""Run end-to-end SLM conversation metrics without touching the fpy database."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
PEFT_DEPS = PROJECT_ROOT / ".peft-deps"
for path in (PROJECT_ROOT, FPY_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting_assistant.domain.schema import load_schema  # noqa: E402
from local_slm_lab.conversation_eval import (  # noqa: E402
    evaluate_conversations,
    load_cases,
    write_report,
)
from local_slm_lab.peft_provider import (  # noqa: E402
    PeftInferenceConfig,
    TransformersPeftProvider,
    resolve_adapter_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "best", "final", "custom"), default="best")
    parser.add_argument("--adapter", help="Adapter directory for --variant custom")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--cases", default="evaluation/conversation-cases-v2.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    if PEFT_DEPS.is_dir() and str(PEFT_DEPS) not in sys.path:
        sys.path.append(str(PEFT_DEPS))

    schema = load_schema()
    adapter_path = resolve_adapter_path(args.variant, args.adapter)
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
    cases = load_cases((PROJECT_ROOT / args.cases).resolve())
    report = asyncio.run(evaluate_conversations(provider, cases, schema))
    output = (PROJECT_ROOT / args.output).resolve()
    write_report(report, output)
    print(f"Report: {output}")
    print(report["metrics"])


if __name__ == "__main__":
    main()
