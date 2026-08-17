"""Evaluate the restored HF base or LoRA adapter without touching the main pipeline."""

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
from local_slm_lab.component_eval import evaluate, load_jsonl, write_report  # noqa: E402
from local_slm_lab.peft_provider import (  # noqa: E402
    PeftInferenceConfig,
    TransformersPeftProvider,
    resolve_adapter_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "best", "final", "custom"), required=True)
    parser.add_argument("--adapter", help="Adapter directory for --variant custom")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--cases", default="corpus-v2/splits/test.jsonl")
    parser.add_argument("--output", help="Required unless --check-artifacts is used")
    parser.add_argument("--limit", type=int, help="Smoke-test only; omit for final reports")
    parser.add_argument("--offset", type=int, default=0, help="Skip scenarios for targeted smoke tests")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--check-artifacts",
        action="store_true",
        help="Validate inputs and adapter files without loading model weights.",
    )
    args = parser.parse_args()

    adapter_path = resolve_adapter_path(args.variant, args.adapter)
    cases_path = (PROJECT_ROOT / args.cases).resolve()
    if not cases_path.is_file():
        raise SystemExit(f"Evaluation cases are missing: {cases_path}")
    if adapter_path is not None and not (adapter_path / "adapter_model.safetensors").is_file():
        raise SystemExit(f"Adapter weights are missing: {adapter_path}")
    if args.check_artifacts:
        print(f"Cases: {cases_path}")
        print(f"Variant: {args.variant}")
        print(f"Adapter: {adapter_path or 'none (base model)'}")
        return
    if not args.output:
        raise SystemExit("--output is required for an evaluation run")

    # Keep the isolated inference environment behind the application's own
    # dependencies so it cannot shadow Pydantic or other pipeline packages.
    if PEFT_DEPS.is_dir() and str(PEFT_DEPS) not in sys.path:
        sys.path.append(str(PEFT_DEPS))

    schema = load_schema()
    scenarios = load_jsonl(cases_path)
    scenarios = scenarios[args.offset :]
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
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
    report = asyncio.run(evaluate(provider, scenarios, schema=schema))
    output = (PROJECT_ROOT / args.output).resolve()
    write_report(report, output)
    print(f"Saved {report['scenario_count']} scenarios to {output}")
    print(f"Variant: {args.variant}")
    print(f"Intent accuracy: {report['metrics']['intent_accuracy']:.4f}")
    print(f"Slot micro F1: {report['metrics']['slot_micro_f1']:.4f}")
    print(f"Question contract: {report['metrics']['question_contract_accuracy']:.4f}")


if __name__ == "__main__":
    main()
