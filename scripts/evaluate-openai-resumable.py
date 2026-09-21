"""Run a resumable, scenario-parallel OpenAI component benchmark.

Credentials come from process environment. No credential is saved to the report.
Each scenario is evaluated with the same component evaluator as other providers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT.parent / "fpy" / "src", PROJECT_ROOT / ".openai-deps"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting_assistant.domain.schema import load_schema  # noqa: E402
from local_slm_lab.component_eval import (  # noqa: E402
    CallRecord,
    evaluate,
    load_jsonl,
    score_records,
    write_report,
)
from local_slm_lab.providers import OpenAIProductionProvider, read_openai_credentials  # noqa: E402


def assemble(
    scenarios: list[dict], completed: dict[str, dict], model: str, cases_sha256: str
) -> dict:
    records = [
        CallRecord(**record)
        for scenario in scenarios
        if scenario["scenario_id"] in completed
        for record in completed[scenario["scenario_id"]]
    ]
    categories = Counter(record.category for record in records)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "provider": OpenAIProductionProvider.name,
        "model": model,
        "scenario_count": len(completed),
        "expected_scenario_count": len(scenarios),
        "cases_sha256": cases_sha256,
        "category_call_counts": dict(sorted(categories.items())),
        "category_metrics": {
            category: score_records([record for record in records if record.category == category])
            for category in sorted(categories)
        },
        "metrics": score_records(records),
        "records": [asdict(record) for record in records],
    }


async def run(args: argparse.Namespace) -> None:
    cases = (PROJECT_ROOT / args.cases).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    cases_sha256 = hashlib.sha256(cases.read_bytes()).hexdigest()
    scenarios = load_jsonl(cases)
    ids = [scenario["scenario_id"] for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Scenario IDs must be unique for safe resume")

    api_key, model = read_openai_credentials((PROJECT_ROOT / args.env_file).resolve())
    schema = load_schema()
    provider = OpenAIProductionProvider(api_key, model, schema)
    completed: dict[str, list[dict]] = {}
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("model") != model or previous.get("cases_sha256") != cases_sha256:
            raise ValueError("Existing report has different model or cases; choose another output")
        for record in previous["records"]:
            completed.setdefault(record["scenario_id"], []).append(record)
        expected_counts = {
            scenario["scenario_id"]: len(scenario["turns"])
            + int(bool(scenario.get("expected_question_slot")))
            for scenario in scenarios
        }
        completed = {
            scenario_id: records
            for scenario_id, records in completed.items()
            if len(records) == expected_counts.get(scenario_id)
        }
        print(f"Resuming after {len(completed)} completed scenarios", flush=True)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(scenario: dict) -> tuple[str, list[dict]]:
        async with semaphore:
            report = await evaluate(provider, [scenario], schema=schema)
            return scenario["scenario_id"], report["records"]

    pending = [asyncio.create_task(one(scenario)) for scenario in scenarios if scenario["scenario_id"] not in completed]
    for task in asyncio.as_completed(pending):
        scenario_id, records = await task
        completed[scenario_id] = records
        report = assemble(scenarios, completed, model, cases_sha256)
        write_report(report, output)
        errors = sum(record["error"] is not None for record in records)
        print(f"Completed {len(completed)}/{len(scenarios)}: {scenario_id}; errors={errors}", flush=True)

    report = assemble(scenarios, completed, model, cases_sha256)
    write_report(report, output)
    print(f"Saved {report['scenario_count']} scenarios to {output}")
    print(f"Slot micro F1: {report['metrics']['slot_micro_f1']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="corpus-v3/splits/validation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", default="../fpy/.env")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
