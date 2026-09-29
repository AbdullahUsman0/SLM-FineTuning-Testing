#!/usr/bin/env python3
"""Run one prepared v5 structured-output research arm on smoke or validation.

This runner never accepts the sealed final split. Validation runs must match the
committed frozen study cohort exactly. Outputs and call journals are exclusive.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import sys
from time import perf_counter
from pathlib import Path

try:
    import resource
except ImportError:  # Windows verification; Colab/Linux reports native process peak RSS.
    resource = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.peft_provider import PeftInferenceConfig
from local_slm_lab.structured_output_providers import (
    V5OpenAIStructuredProvider,
    V5RetryTransformersProvider,
    V5RuleProvider,
    V5SlotWiseTransformersProvider,
    V5XGrammarTransformersProvider,
)
from local_slm_lab.v5_eval import (
    PROTOCOL, SCORER_VERSION, build_call_plan, digest, evaluate, load_cases, load_schema,
    now, provenance, runtime_metadata, safe_error, sha256, strict_json,
)
from local_slm_lab.v5_provider import V5TransformersProvider

ARMS = ("A", "B", "C", "D", "E", "F")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arm", required=True, choices=ARMS)
    result.add_argument("--cases", type=Path, default=ROOT / "corpus-v5/v5-20260919-r1/splits/validation.jsonl")
    result.add_argument("--split", choices=("smoke", "validation"), default="validation")
    result.add_argument("--study-manifest", type=Path,
                        default=ROOT / "evaluation/v5-structured-output/prepared-validation-v2/study-manifest.json")
    result.add_argument("--schemas", type=Path, default=ROOT / "evaluation/v5-structured-output/schemas")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model", help="Explicit local model ID/path or OpenAI model snapshot")
    result.add_argument("--revision", help="Pinned 40-character Hugging Face revision for A-D")
    result.add_argument("--adapter", type=Path, help="Chosen frozen LoRA adapter for A-D")
    result.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    result.add_argument("--device", default="cuda")
    result.add_argument("--max-new-tokens", type=int, default=1024)
    result.add_argument("--warmup-cases", type=Path,
                        default=ROOT / "corpus-v5/v5-20260919-r1/splits/smoke.jsonl")
    result.add_argument("--skip-warmup", action="store_true", help="Diagnostics only; forbidden for validation")
    result.add_argument("--allow-high-call-count", action="store_true",
                        help="Required for slot-wise Arm D (approximately 79 model calls per extraction)")
    result.add_argument("--openai-cost-approved", action="store_true",
                        help="Required with V5_OPENAI_COST_APPROVED=yes for paid Arm F")
    return result


def absolute(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_output_schemas(directory: Path) -> tuple[dict[str, dict], dict]:
    directory = absolute(directory)
    manifest_path = directory / "manifest.json"
    manifest = strict_json(manifest_path.read_text(encoding="utf-8"))
    mapping = {"extract": "extractor-result.schema.json", "ask": "question-output.schema.json"}
    schemas = {}
    for task, name in mapping.items():
        path = directory / name
        declared = manifest["files"][name]
        observed = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if observed != declared:
            raise ValueError(f"output schema hash mismatch: {name}")
        schemas[task] = strict_json(path.read_text(encoding="utf-8"))
    return schemas, {"path": str(directory), "manifest_sha256": sha256(manifest_path),
                     "files": manifest["files"], "pydantic_version_at_export": manifest["pydantic_version"]}


def frozen_rows(plan: list[dict]) -> list[dict]:
    rows = []
    for index, call in enumerate(plan):
        rows.append({
            "ordinal": index, "key": call["key"], "scenario_id": call["scenario_id"],
            "cluster_id": call["cluster_id"], "category": call["category"], "task": call["task"],
            "context_sha256": digest(call["context"]),
            "gold_sha256": digest({"expected": call["expected"],
                                   "forbidden_slots": call["forbidden_slots"],
                                   "gold_after": call.get("gold_after")}),
        })
    return rows


def verify_frozen_study(manifest_path: Path, input_metadata: dict, plan: list[dict]) -> dict:
    manifest_path = absolute(manifest_path)
    manifest = strict_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_validation_only":
        raise ValueError("study manifest is not prepared")
    if input_metadata["split"] != "validation" or input_metadata["sha256"] != manifest["input"]["sha256"]:
        raise ValueError("validation input differs from frozen study")
    rows = frozen_rows(plan)
    call_plan = manifest["call_plan"]
    if len(rows) != call_plan["call_count"]:
        raise ValueError("call count differs from frozen study")
    if digest([row["key"] for row in rows]) != call_plan["ordered_keys_sha256"]:
        raise ValueError("ordered call keys differ from frozen study")
    context_gold = digest([[row["key"], row["context_sha256"], row["gold_sha256"]] for row in rows])
    if context_gold != call_plan["ordered_context_gold_sha256"]:
        raise ValueError("context/gold hashes differ from frozen study")
    return {"path": str(manifest_path), "sha256": sha256(manifest_path),
            "call_plan_sha256": call_plan["sha256"], "ordered_context_gold_sha256": context_gold}


def local_config(args: argparse.Namespace) -> PeftInferenceConfig:
    if not args.model or not args.revision or len(args.revision) != 40:
        raise ValueError("Arms A-D require explicit model and pinned 40-character revision")
    if args.adapter is None:
        raise ValueError("Arms A-D require the chosen adapter")
    adapter = absolute(args.adapter)
    return PeftInferenceConfig(base_model=args.model, revision=args.revision, variant="custom",
                               adapter_path=adapter, dtype=args.dtype, device=args.device,
                               max_new_tokens=args.max_new_tokens)


def create_provider(args: argparse.Namespace, schema, output_schemas):
    if args.arm in "ABCD":
        config = local_config(args)
        if args.arm == "A":
            return V5TransformersProvider(config, schema)
        if args.arm == "B":
            return V5RetryTransformersProvider(config, schema)
        if args.arm == "C":
            return V5XGrammarTransformersProvider(config, schema, output_schemas)
        if not args.allow_high_call_count:
            raise PermissionError("Arm D requires --allow-high-call-count after reviewing the estimated workload")
        return V5SlotWiseTransformersProvider(config, schema)
    if args.arm == "E":
        if args.adapter is not None:
            raise ValueError("Arm E is rule-only and does not accept an adapter")
        return V5RuleProvider(schema)
    if not args.model:
        raise ValueError("Arm F requires an explicit OpenAI model snapshot")
    if not args.openai_cost_approved or os.environ.get("V5_OPENAI_COST_APPROVED") != "yes":
        raise PermissionError("Arm F requires --openai-cost-approved and V5_OPENAI_COST_APPROVED=yes")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is required in the process environment")
    return V5OpenAIStructuredProvider(key, args.model, schema, output_schemas,
                                      max_output_tokens=args.max_new_tokens)


def settings(args: argparse.Namespace, audit: dict, schema_metadata: dict) -> dict:
    adapter = absolute(args.adapter) if args.adapter else None
    adapter_hashes = None
    if adapter:
        adapter_hashes = {name: sha256(adapter / name) for name in
                          ("adapter_model.safetensors", "adapter_config.json")}
    provider = "openai" if args.arm == "F" else ("rules" if args.arm == "E" else "lora")
    return {
        "provider": provider, "study_arm": args.arm, "model": args.model or "deterministic-rules-v1",
        "revision": args.revision, "adapter": str(adapter) if adapter else None,
        "adapter_sha256": adapter_hashes, "dtype": args.dtype if args.arm in "ABCD" else None,
        "device": args.device if args.arm in "ABCD" else None,
        "max_new_tokens": args.max_new_tokens if args.arm in "ABCDF" else None,
        "decoding": {"mode": "greedy", "do_sample": False, "repetition_penalty": 1.0,
                     "enable_thinking": False, "retries": 1 if args.arm == "B" else 0,
                     "constraint": "xgrammar_json_schema" if args.arm == "C" else
                                   ("openai_json_schema" if args.arm == "F" else None)},
        "prompt_sha256": audit["prompt_sha256"], "scorer_sha256": audit["scorer_sha256"],
        "dependency_sha256": audit["dependency_sha256"],
        "output_schema_manifest_sha256": schema_metadata["manifest_sha256"],
    }


async def warmup(provider, schema, path: Path) -> dict:
    # smoke.jsonl is a deterministic sample of the validation cohort and keeps
    # each scenario's original `split: validation` provenance label.
    cases, metadata = load_cases(absolute(path), split="validation")
    plan = build_call_plan(cases[:1], schema)
    selected = []
    extraction = next(call for call in plan if call["task"] == "extract")
    selected.append(extraction)
    question = next((call for call in plan if call["task"] == "ask"), None)
    if question is not None:
        selected.append(question)
    started = perf_counter()
    for call in selected:
        if call["task"] == "extract":
            await provider.extract(call["context"]["message"], call["state"].model_copy(deep=True))
        else:
            await provider.ask(call["request"].model_copy(deep=True))
    traces = provider.drain_traces() if hasattr(provider, "drain_traces") else []
    return {"cases_sha256": metadata["sha256"], "calls": len(selected),
            "reported_model_calls": sum(trace.get("model_calls", 1) for trace in traces),
            "elapsed_ms": (perf_counter() - started) * 1000, "completed": True}


def reset_memory(provider) -> None:
    torch = getattr(provider, "_torch", None)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def memory_metadata(provider) -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF) if resource is not None else None
    result = {"ru_maxrss": usage.ru_maxrss if usage is not None else None,
              "ru_maxrss_unit": "kilobytes_linux_or_bytes_macos" if usage is not None else None,
              "platform": platform.system()}
    torch = getattr(provider, "_torch", None)
    if torch is not None and torch.cuda.is_available():
        result.update(cuda_max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_max_memory_reserved_bytes=torch.cuda.max_memory_reserved())
    else:
        result.update(cuda_max_memory_allocated_bytes=None, cuda_max_memory_reserved_bytes=None)
    return result


def run(args: argparse.Namespace) -> dict:
    output = absolute(args.output)
    journal = output.with_name(output.name + ".calls.jsonl")
    if output.exists() or journal.exists():
        raise FileExistsError("output or call journal already exists")
    if not output.parent.is_dir():
        raise FileNotFoundError("output parent must already exist")
    if args.split == "validation" and args.skip_warmup:
        raise ValueError("validation runs require the declared warmup")
    with output.open("x", encoding="utf-8") as report_file:
        partial = {"report_version": SCORER_VERSION, "status": "partial", "started_at": now(),
                   "protocol": PROTOCOL, "records": [], "resumable": False}
        def save(value):
            report_file.seek(0)
            json.dump(value, report_file, ensure_ascii=False, indent=2, allow_nan=False)
            report_file.write("\n")
            report_file.truncate()
            report_file.flush()
            os.fsync(report_file.fileno())
        save(partial)
        try:
            cases, input_metadata = load_cases(absolute(args.cases), split=args.split)
            schema = load_schema()
            plan = build_call_plan(cases, schema)
            study = verify_frozen_study(args.study_manifest, input_metadata, plan) if args.split == "validation" else None
            output_schemas, schema_metadata = load_output_schemas(args.schemas)
            # Every arm uses the local v5 prompts, including the external structured-output ceiling.
            audit = provenance("base")
            declared = settings(args, audit, schema_metadata)
            metadata = {"input": input_metadata, "settings": declared, "settings_sha256": digest(declared),
                        "study": study, "schema": schema_metadata, "provenance": audit,
                        "runner_sha256": sha256(Path(__file__)),
                        "research_provider_sha256": sha256(ROOT / "local_slm_lab/structured_output_providers.py")}
            partial["metadata"] = metadata
            save(partial)
            with journal.open("x", encoding="utf-8") as calls_file:
                def recorded(record):
                    partial["records"].append(record)
                    calls_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    calls_file.flush()
                    os.fsync(calls_file.fileno())
                provider = create_provider(args, schema, output_schemas)
                metadata["runtime"] = runtime_metadata(provider)
                metadata["versions"] = {name: _version(name) for name in
                                        ("torch", "transformers", "peft", "xgrammar", "openai", "pydantic")}
                metadata["schema_compile_ms"] = getattr(provider, "schema_compile_ms", None)
                metadata["warmup"] = ({"completed": False, "reason": "diagnostic_skip"} if args.skip_warmup
                                      else asyncio.run(warmup(provider, schema, args.warmup_cases)))
                reset_memory(provider)
                save(partial)
                report = asyncio.run(evaluate(provider, cases, schema=schema, metadata=metadata,
                                              on_record=recorded, plan=plan))
                report["metrics"]["memory"] = memory_metadata(provider)
                report["call_journal"] = str(journal)
                report["call_journal_sha256"] = sha256(journal)
                save(report)
                return report
        except BaseException as error:
            partial["error"] = safe_error(error)
            partial["finished_at"] = now()
            save(partial)
            raise


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = run(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"output": str(absolute(args.output)), "arm": args.arm,
                      "nonintent_f1": report["metrics"]["nonintent"]["f1"],
                      "raw_schema_valid": report["metrics"]["raw_schema_valid"]["rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
