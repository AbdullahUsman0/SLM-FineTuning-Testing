#!/usr/bin/env python3
"""Freeze the validation call plan shared by structured-output study arms.

This command performs no model inference and never opens the sealed final split.
It creates a new, immutable preparation directory containing an auditable call
plan plus the hashes and protocol declarations needed for paired comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.v5_eval import (
    SCORER_VERSION,
    build_call_plan,
    digest,
    load_cases,
    load_schema,
    now,
    provenance,
    sha256,
)

STUDY_VERSION = "v5-structured-output-1"
ARMS = {
    "A": {
        "name": "prompt_json_pydantic_no_retry",
        "generation": "greedy prompt-only JSON",
        "validation": "strict JSON parser then Pydantic",
        "retries": 0,
        "implementation_status": "available_in_evaluate_v5",
    },
    "B": {
        "name": "prompt_json_pydantic_one_retry",
        "generation": "same first attempt as A",
        "validation": "strict JSON parser then Pydantic; compact validation feedback",
        "retries": 1,
        "implementation_status": "planned_step_2",
    },
    "C": {
        "name": "json_schema_constrained_pydantic",
        "generation": "same model, adapter, prompt, greedy decoding and token limit as A with grammar enforcement",
        "validation": "exported strict JSON Schema during decoding, then local Pydantic",
        "retries": 0,
        "implementation_status": "planned_step_3_backend_benchmark_required",
    },
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--cases", type=Path,
        default=ROOT / "corpus-v5/v5-20260919-r1/splits/validation.jsonl",
    )
    result.add_argument(
        "--schemas", type=Path,
        default=ROOT / "evaluation/v5-structured-output/schemas",
    )
    result.add_argument("--output", type=Path, required=True, help="A new directory; it must not exist")
    return result


def absolute(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verified_schemas(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("schema manifest has no files")
    observed = {}
    for name, metadata in sorted(declared.items()):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"missing schema file: {name}")
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != metadata:
            raise ValueError(f"schema manifest mismatch: {name}")
        observed[name] = actual
    return {
        "directory": str(directory),
        "manifest_sha256": sha256(manifest_path),
        "json_schema_dialect": manifest.get("json_schema_dialect"),
        "pydantic_version_at_export": manifest.get("pydantic_version"),
        "slot_count": manifest.get("slot_count"),
        "files": observed,
    }


def frozen_record(index: int, call: dict[str, Any]) -> dict[str, Any]:
    record = {
        "ordinal": index,
        "key": call["key"],
        "scenario_id": call["scenario_id"],
        "cluster_id": call["cluster_id"],
        "category": call["category"],
        "task": call["task"],
        "context": call["context"],
        "expected": call["expected"],
        "forbidden_slots": call["forbidden_slots"],
    }
    if call["task"] == "extract":
        record["gold_after"] = call["gold_after"]
    record["context_sha256"] = digest(record["context"])
    record["gold_sha256"] = digest({
        "expected": record["expected"],
        "forbidden_slots": record["forbidden_slots"],
        "gold_after": record.get("gold_after"),
    })
    return record


def repository_state() -> dict[str, Any]:
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
    return {"revision": revision, "dirty": bool(dirty.strip())}


def prepare(cases_path: Path, schemas_path: Path, output: Path) -> dict[str, Any]:
    cases_path, schemas_path, output = map(absolute, (cases_path, schemas_path, output))
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    scenarios, input_metadata = load_cases(cases_path, split="validation")
    schema = load_schema()
    calls = build_call_plan(scenarios, schema)
    frozen = [frozen_record(index, call) for index, call in enumerate(calls)]
    if len({tuple(row["key"]) for row in frozen}) != len(frozen):
        raise ValueError("duplicate call key")

    provider_provenance = provenance("base")
    schema_metadata = verified_schemas(schemas_path)
    task_counts = Counter(row["task"] for row in frozen)
    category_counts = Counter(row["category"] for row in frozen)
    cluster_count = len({row["cluster_id"] for row in frozen})

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        call_plan_path = temporary / "call-plan.jsonl"
        with call_plan_path.open("x", encoding="utf-8") as handle:
            for row in frozen:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        manifest = {
            "study_version": STUDY_VERSION,
            "status": "prepared_validation_only",
            "created_at_utc": now(),
            "input": input_metadata,
            "schema": schema_metadata,
            "forecasting_schema_sha256": digest(schema.model_dump(mode="json")),
            "call_plan": {
                "file": call_plan_path.name,
                "sha256": sha256(call_plan_path),
                "scenario_count": len(scenarios),
                "cluster_count": cluster_count,
                "call_count": len(frozen),
                "task_counts": dict(sorted(task_counts.items())),
                "category_counts": dict(sorted(category_counts.items())),
                "ordered_keys_sha256": digest([row["key"] for row in frozen]),
                "ordered_context_gold_sha256": digest([
                    [row["key"], row["context_sha256"], row["gold_sha256"]] for row in frozen
                ]),
            },
            "arms": ARMS,
            "comparison_contract": {
                "primary_metric": "non-intent exact slot micro F1",
                "paired_cluster_bootstrap": {"samples": 10000, "seed": 42, "unit": "source scenario cluster"},
                "same_first_attempt_for_A_and_B": True,
                "raw_text_retained_before_parsing": True,
                "semantic_validation_after_generation_in_all_arms": True,
                "sealed_final_used": False,
            },
            "software": {
                "repository": repository_state(),
                "preparer_sha256": sha256(Path(__file__)),
                "scorer_version": SCORER_VERSION,
                "scorer_sha256": provider_provenance["scorer_sha256"],
                "prompt_sha256": provider_provenance["prompt_sha256"],
                "dependency_sha256": provider_provenance["dependency_sha256"],
                "fpy_revision": provider_provenance["fpy_revision"],
                "versions": provider_provenance["versions"],
            },
        }
        write_json(temporary / "study-manifest.json", manifest)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest = prepare(args.cases, args.schemas, args.output)
    print(json.dumps({
        "output": str(absolute(args.output)),
        "status": manifest["status"],
        "scenario_count": manifest["call_plan"]["scenario_count"],
        "call_count": manifest["call_plan"]["call_count"],
        "call_plan_sha256": manifest["call_plan"]["sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
