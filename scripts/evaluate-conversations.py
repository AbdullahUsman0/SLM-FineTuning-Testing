"""Run end-to-end SLM conversation metrics without touching the fpy database."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
PEFT_DEPS = PROJECT_ROOT / ".peft-deps"
OPENAI_DEPS = PROJECT_ROOT / ".openai-deps"
for path in (PROJECT_ROOT, FPY_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting_assistant.domain.schema import load_schema  # noqa: E402
from forecasting_assistant.application.orchestrator import ElicitationEngine  # noqa: E402
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
from local_slm_lab.client import LocalModelConfig  # noqa: E402
from local_slm_lab.providers import LocalFineTunedProvider  # noqa: E402
from local_slm_lab.providers import OpenAIProductionProvider, read_openai_credentials  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("transformers", "server", "openai"), default="transformers"
    )
    parser.add_argument("--config", default="config.v2-evaluation.json")
    parser.add_argument("--env-file", default="../fpy/.env")
    parser.add_argument("--variant", choices=("base", "best", "final", "custom"), default="best")
    parser.add_argument("--adapter", help="Adapter directory for --variant custom")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--cases", default="evaluation/conversation-cases-v3.json")
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
    engine_class = None
    if args.backend == "server":
        provider = LocalFineTunedProvider(
            LocalModelConfig.from_file(PROJECT_ROOT / args.config), schema
        )
    elif args.backend == "openai":
        if str(OPENAI_DEPS) not in sys.path:
            sys.path.insert(0, str(OPENAI_DEPS))
        api_key, model = read_openai_credentials((PROJECT_ROOT / args.env_file).resolve())
        provider = OpenAIProductionProvider(api_key, model, schema)
        engine_class = ElicitationEngine
    else:
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
    evaluation_kwargs = {"engine_class": engine_class} if engine_class is not None else {}
    report = asyncio.run(evaluate_conversations(provider, cases, schema, **evaluation_kwargs))
    output = (PROJECT_ROOT / args.output).resolve()
    write_report(report, output)
    print(f"Report: {output}")
    print(report["metrics"])


if __name__ == "__main__":
    main()
