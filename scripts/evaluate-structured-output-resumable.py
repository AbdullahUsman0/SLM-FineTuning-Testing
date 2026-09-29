#!/usr/bin/env python3
"""Durable scenario-by-scenario validation runner for structured-output arms.

Completed scenarios are immutable. An interrupted scenario is preserved and
replayed, limiting lost work without combining reports from different runtime
configurations. The sealed final split is unsupported.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.v5_eval import (
    PROTOCOL, SCORER_VERSION, build_call_plan, digest, evaluate, load_cases, load_schema,
    now, provenance, runtime_metadata, safe_error, score_records, sha256, strict_json,
)


def load_base_runner():
    path = ROOT / "scripts/evaluate-structured-output.py"
    spec = importlib.util.spec_from_file_location("structured_output_base_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_runner()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arm", required=True, choices=BASE.ARMS)
    result.add_argument("--cases", type=Path, default=ROOT / "corpus-v5/v5-20260919-r1/splits/validation.jsonl")
    result.add_argument("--study-manifest", type=Path,
                        default=ROOT / "evaluation/v5-structured-output/prepared-validation-v2/study-manifest.json")
    result.add_argument("--schemas", type=Path, default=ROOT / "evaluation/v5-structured-output/schemas")
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--model")
    result.add_argument("--revision")
    result.add_argument("--adapter", type=Path)
    result.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    result.add_argument("--device", default="cuda")
    result.add_argument("--max-new-tokens", type=int, default=1024)
    result.add_argument("--warmup-cases", type=Path,
                        default=ROOT / "corpus-v5/v5-20260919-r1/splits/smoke.jsonl")
    result.add_argument("--allow-high-call-count", action="store_true")
    result.add_argument("--openai-cost-approved", action="store_true")
    return result


def write_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_name(index: int, scenario_id: str) -> str:
    return f"{index:04d}-{digest(scenario_id)[:12]}.json"


def runtime_signature(runtime: dict) -> dict:
    return {key: runtime.get(key) for key in (
        "actual_device", "actual_dtype", "resolved_revision", "gpu_names", "device_map",
        "chat_template_sha256", "generation_config", "cuda_version",
    )}


def validate_completed(path: Path, scenario_id: str, fingerprint: str) -> dict:
    report = strict_json(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("scenario_ids") != [scenario_id]:
        raise ValueError(f"invalid completed scenario report: {path.name}")
    if report.get("metadata", {}).get("run_fingerprint") != fingerprint:
        raise ValueError(f"scenario run fingerprint differs: {path.name}")
    return report


def quarantine_partial(partial: Path, interrupted: Path) -> None:
    if not partial.exists():
        return
    interrupted.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    os.replace(partial, interrupted / f"{partial.stem}-{stamp}.json")


def combine(run_dir: Path, manifest: dict, scenarios: list[dict], schema) -> dict:
    reports = []
    scenario_hashes = {}
    for index, scenario in enumerate(scenarios):
        sid = scenario.get("scenario_id", scenario.get("id"))
        path = run_dir / "scenarios" / safe_name(index, sid)
        report = validate_completed(path, sid, manifest["run_fingerprint"])
        reports.append(report)
        scenario_hashes[path.name] = sha256(path)
    records = [record for report in reports for record in report["records"]]
    slot_ids = [slot.slot_id for slot in schema.slots]
    categories = sorted({record["category"] for record in records})
    return {
        "report_version": SCORER_VERSION,
        "status": "complete",
        "started_at": manifest["created_at_utc"],
        "finished_at": now(),
        "protocol": PROTOCOL,
        "metadata": manifest["metadata"],
        "scenario_count": len(scenarios),
        "scenario_ids": [scenario.get("scenario_id", scenario.get("id")) for scenario in scenarios],
        "metrics": score_records(records, slot_ids),
        "category_metrics": {category: score_records([r for r in records if r["category"] == category], slot_ids)
                             for category in categories},
        "records": records,
        "scenario_reports": {"directory": "scenarios", "sha256": scenario_hashes},
        "run_fingerprint": manifest["run_fingerprint"],
    }


def run(args: argparse.Namespace) -> dict:
    run_dir = BASE.absolute(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir = run_dir / "scenarios"
    interrupted_dir = run_dir / "interrupted"
    sessions_dir = run_dir / "sessions"
    scenario_dir.mkdir(exist_ok=True)
    sessions_dir.mkdir(exist_ok=True)

    cases, input_metadata = load_cases(BASE.absolute(args.cases), split="validation")
    schema = load_schema()
    plan = build_call_plan(cases, schema)
    study = BASE.verify_frozen_study(args.study_manifest, input_metadata, plan)
    output_schemas, schema_metadata = BASE.load_output_schemas(args.schemas)
    audit = provenance("base")
    declared = BASE.settings(args, audit, schema_metadata)
    identity = {
        "input_sha256": input_metadata["sha256"], "study": study,
        "settings": declared, "schema_manifest_sha256": schema_metadata["manifest_sha256"],
        "scorer_version": SCORER_VERSION, "scorer_sha256": audit["scorer_sha256"],
        "provider_sha256": sha256(ROOT / "local_slm_lab/structured_output_providers.py"),
        "runner_sha256": sha256(Path(__file__)),
    }
    fingerprint = digest(identity)
    manifest_path = run_dir / "run-manifest.json"
    existing = strict_json(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if existing and existing.get("run_fingerprint") != fingerprint:
        raise ValueError("run directory belongs to different inputs, code, adapter, or settings")
    combined_path = run_dir / "combined-report.json"
    if existing and combined_path.exists():
        combined = strict_json(combined_path.read_text(encoding="utf-8"))
        if combined.get("status") != "complete" or combined.get("run_fingerprint") != fingerprint:
            raise ValueError("existing combined report is incompatible")
        return combined

    provider = BASE.create_provider(args, schema, output_schemas)
    runtime = runtime_metadata(provider)
    if existing and existing["runtime_signature"] != runtime_signature(runtime):
        raise ValueError("runtime differs from the frozen resumable run")
    metadata = {"input": input_metadata, "settings": declared, "settings_sha256": digest(declared),
                "study": study, "schema": schema_metadata, "provenance": audit,
                "runtime": runtime, "run_fingerprint": fingerprint,
                "research_provider_sha256": identity["provider_sha256"]}
    if existing is None:
        existing = {"manifest_version": "v5-structured-resumable-1", "created_at_utc": now(),
                    "run_fingerprint": fingerprint, "identity": identity,
                    "runtime_signature": runtime_signature(runtime), "metadata": metadata,
                    "scenario_count": len(cases), "completion": "scenario_files_then_combined_report"}
        write_exclusive(manifest_path, existing)

    session_started = now()
    warmup = asyncio.run(BASE.warmup(provider, schema, args.warmup_cases))
    BASE.reset_memory(provider)
    by_scenario = defaultdict(list)
    for call in plan:
        by_scenario[call["scenario_id"]].append(call)

    for index, scenario in enumerate(cases):
        sid = scenario.get("scenario_id", scenario.get("id"))
        completed = scenario_dir / safe_name(index, sid)
        if completed.exists():
            validate_completed(completed, sid, fingerprint)
            print(f"SKIP complete {index + 1}/{len(cases)} {sid}")
            continue
        partial = scenario_dir / ("." + safe_name(index, sid) + ".partial")
        quarantine_partial(partial, interrupted_dir)
        progress = {"status": "partial", "scenario_id": sid, "run_fingerprint": fingerprint,
                    "started_at": now(), "records": []}
        atomic_write(partial, progress)
        def recorded(record):
            progress["records"].append(record)
            atomic_write(partial, progress)
        report = asyncio.run(evaluate(provider, [scenario], schema=schema, metadata=metadata,
                                      on_record=recorded, plan=by_scenario[sid]))
        report["metadata"]["run_fingerprint"] = fingerprint
        atomic_write(partial, report)
        os.replace(partial, completed)
        print(f"DONE {index + 1}/{len(cases)} {sid}")

    combined = combine(run_dir, existing, cases, schema)
    if combined_path.exists():
        observed = strict_json(combined_path.read_text(encoding="utf-8"))
        if observed.get("run_fingerprint") != fingerprint or observed.get("status") != "complete":
            raise ValueError("existing combined report is incompatible")
        combined = observed
    else:
        write_exclusive(combined_path, combined)

    session = {"session_started_at": session_started, "session_finished_at": now(),
               "warmup": warmup, "memory": BASE.memory_metadata(provider),
               "combined_report_sha256": sha256(combined_path)}
    session_path = sessions_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    write_exclusive(session_path, session)
    return combined


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        report = run(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "run_dir": str(BASE.absolute(args.run_dir)),
                      "scenarios": report["scenario_count"],
                      "nonintent_f1": report["metrics"]["nonintent"]["f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
