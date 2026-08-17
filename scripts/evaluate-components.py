"""Evaluate the held-out corpus against one structured LLM provider."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
VENDOR = PROJECT_ROOT / ".vendor"
for path in (PROJECT_ROOT, FPY_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting_assistant.domain.schema import load_schema  # noqa: E402
from local_slm_lab.client import LocalModelConfig  # noqa: E402
from local_slm_lab.component_eval import evaluate, load_jsonl, write_report  # noqa: E402
from local_slm_lab.providers import (  # noqa: E402
    LocalStructuredProvider,
    OpenAIProductionProvider,
    read_openai_credentials,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("local", "openai"), required=True)
    parser.add_argument("--cases", default="corpus/splits/test.jsonl")
    parser.add_argument("--config", default="config.evaluation.json")
    parser.add_argument("--env-file", default="../fpy/.env")
    parser.add_argument("--limit", type=int, help="Smoke-test only; omit for final reports")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    schema = load_schema()
    scenarios = load_jsonl(PROJECT_ROOT / args.cases)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    if args.provider == "local":
        provider = LocalStructuredProvider(
            LocalModelConfig.from_file(PROJECT_ROOT / args.config), schema
        )
    else:
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        api_key, model = read_openai_credentials((PROJECT_ROOT / args.env_file).resolve())
        provider = OpenAIProductionProvider(api_key, model, schema)

    report = asyncio.run(evaluate(provider, scenarios, schema=schema))
    output = PROJECT_ROOT / args.output
    write_report(report, output)
    metrics = report["metrics"]
    print(f"Saved {report['scenario_count']} scenarios to {output}")
    print(f"Provider: {report['provider']} ({report['model']})")
    print(f"Intent accuracy: {metrics['intent_accuracy']:.4f}")
    print(f"Slot micro F1: {metrics['slot_micro_f1']:.4f}")
    print(f"Question contract: {metrics['question_contract_accuracy']:.4f}")


if __name__ == "__main__":
    main()
