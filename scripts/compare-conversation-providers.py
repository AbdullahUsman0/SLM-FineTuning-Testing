"""Run local v2 and the configured OpenAI pipeline concurrently on identical cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPY_SRC = PROJECT_ROOT.parent / "fpy" / "src"
OPENAI_DEPS = PROJECT_ROOT / ".openai-deps"
for path in (OPENAI_DEPS, PROJECT_ROOT, FPY_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting_assistant.application.orchestrator import ElicitationEngine  # noqa: E402
from forecasting_assistant.domain.schema import create_initial_state, load_schema  # noqa: E402
from local_slm_lab.client import LocalModelConfig  # noqa: E402
from local_slm_lab.conversation_eval import (  # noqa: E402
    evaluate_conversations,
    load_cases,
    score_conversations,
    write_report,
)
from local_slm_lab.providers import (  # noqa: E402
    LocalFineTunedProvider,
    OpenAIProductionProvider,
    read_openai_credentials,
)


HIGHER_IS_BETTER = (
    "case_success_rate",
    "intent_accuracy",
    "exact_final_state_accuracy",
    "slot_micro_precision",
    "slot_micro_recall",
    "slot_micro_f1",
    "slot_id_micro_f1",
    "assistant_safety_rate",
    "state_progression_rate",
)
LOWER_IS_BETTER = (
    "unexpected_slot_inference_rate",
    "stalled_turn_rate",
    "consecutive_question_repeat_rate",
)


async def preflight(provider, schema, label: str) -> None:
    try:
        await provider.extract("yes", create_initial_state(schema))
    except Exception as exc:
        raise RuntimeError(f"{label} preflight failed: {exc}") from None


def comparison_rows(local_metrics: dict, openai_metrics: dict) -> list[dict]:
    rows = []
    for metric in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER):
        local_value = local_metrics.get(metric)
        openai_value = openai_metrics.get(metric)
        delta = None
        winner = "unavailable"
        if isinstance(local_value, (int, float)) and isinstance(openai_value, (int, float)):
            delta = round(local_value - openai_value, 4)
            if local_value == openai_value:
                winner = "tie"
            elif metric in HIGHER_IS_BETTER:
                winner = "local_v2" if local_value > openai_value else "openai"
            else:
                winner = "local_v2" if local_value < openai_value else "openai"
        rows.append(
            {
                "metric": metric,
                "local_v2": local_value,
                "openai": openai_value,
                "local_minus_openai": delta,
                "winner": winner,
            }
        )
    return rows


def report_for(provider, records: list[dict]) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "provider": provider.name,
        "model": provider.model,
        "metrics": score_conversations(records),
        "records": records,
    }


def load_completed_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("records", [])
    return records if isinstance(records, list) else []


def write_outputs(
    output_dir: Path,
    args,
    cases: list[dict],
    local_provider,
    openai_provider,
    local_records: list[dict],
    openai_records: list[dict],
) -> None:
    local_report = report_for(local_provider, local_records)
    openai_report = report_for(openai_provider, openai_records)
    write_report(local_report, output_dir / "local-v2.json")
    write_report(openai_report, output_dir / "openai.json")
    comparison = {
        "created_at": datetime.now(UTC).isoformat(),
        "cases": args.cases,
        "requested_case_count": len(cases),
        "completed_case_count": len(local_records),
        "complete": len(local_records) == len(cases),
        "local_model": local_provider.model,
        "openai_model": openai_provider.model,
        "rows": comparison_rows(local_report["metrics"], openai_report["metrics"]),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )


async def run(args) -> None:
    schema = load_schema()
    cases = load_cases((PROJECT_ROOT / args.cases).resolve())
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    local_provider = LocalFineTunedProvider(
        LocalModelConfig.from_file((PROJECT_ROOT / args.local_config).resolve()), schema
    )
    api_key, model = read_openai_credentials((PROJECT_ROOT / args.env_file).resolve())
    openai_provider = OpenAIProductionProvider(api_key, model, schema)

    await asyncio.gather(
        preflight(local_provider, schema, "local v2"),
        preflight(openai_provider, schema, "OpenAI"),
    )

    for provider in (local_provider, openai_provider):
        if hasattr(provider, "drain_traces"):
            provider.drain_traces()

    local_records = load_completed_records(output_dir / "local-v2.json") if args.resume else []
    openai_records = load_completed_records(output_dir / "openai.json") if args.resume else []
    common_completed = {
        record["case_id"] for record in local_records
    } & {record["case_id"] for record in openai_records}
    local_records = [record for record in local_records if record["case_id"] in common_completed]
    openai_records = [record for record in openai_records if record["case_id"] in common_completed]
    if common_completed:
        print(f"Resuming after {len(common_completed)} completed case(s).", flush=True)

    for index, case in enumerate(cases, start=1):
        if case["case_id"] in common_completed:
            continue
        print(f"[{index}/{len(cases)}] Starting {case['case_id']}...", flush=True)
        local_case, openai_case = await asyncio.gather(
            evaluate_conversations(local_provider, [case], schema),
            evaluate_conversations(
                openai_provider,
                [case],
                schema,
                engine_class=ElicitationEngine,
            ),
        )
        local_records.extend(local_case["records"])
        openai_records.extend(openai_case["records"])
        write_outputs(
            output_dir,
            args,
            cases,
            local_provider,
            openai_provider,
            local_records,
            openai_records,
        )
        print(
            f"[{index}/{len(cases)}] Saved {case['case_id']}: "
            f"local={local_case['metrics']['case_success_rate']}, "
            f"OpenAI={openai_case['metrics']['case_success_rate']}",
            flush=True,
        )

    write_outputs(
        output_dir,
        args,
        cases,
        local_provider,
        openai_provider,
        local_records,
        openai_records,
    )
    local_report = report_for(local_provider, local_records)
    openai_report = report_for(openai_provider, openai_records)
    comparison = {
        "rows": comparison_rows(local_report["metrics"], openai_report["metrics"])
    }

    print(f"Reports: {output_dir}")
    print(f"{'metric':38} {'local v2':>10} {'OpenAI':>10} {'winner':>12}")
    for row in comparison["rows"]:
        print(
            f"{row['metric']:38} {str(row['local_v2']):>10} "
            f"{str(row['openai']):>10} {row['winner']:>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/conversation-cases-v3.json")
    parser.add_argument("--local-config", default="config.v2-evaluation.json")
    parser.add_argument("--env-file", default="../fpy/.env")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=f"results/paired-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
