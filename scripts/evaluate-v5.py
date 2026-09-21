"""Run raw v5 component evaluation. No env files, implicit models, or recovery.

Final requires a manually reviewed v5-selection-1 JSON manifest. See
local_slm_lab.v5_eval.authorize_final for the strict declaration contract.
Partial reports and call journals are retained; a claimed final cannot retry.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.v5_eval import (
    SCORER_VERSION, PROTOCOL, build_call_plan, claim_final_run, digest, evaluate,
    instrument_openai, load_cases, load_schema, now, provenance, runtime_metadata,
    safe_error, sha256,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--provider", required=True, choices=("base", "lora", "openai"))
    result.add_argument("--cases", type=Path, default=ROOT / "corpus-v5/v5-20260919-r1/splits/smoke.jsonl")
    result.add_argument("--output", type=Path, required=True, help="New report path; parent directory must exist")
    result.add_argument("--model", required=True, help="Explicit model ID/path; never inferred from credentials")
    result.add_argument("--revision", help="Immutable 40-character HF revision required by the local v5 provider")
    result.add_argument("--adapter", type=Path)
    result.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    result.add_argument("--device", default="auto")
    result.add_argument("--max-new-tokens", type=int, default=1024)
    result.add_argument("--split", choices=("dev", "train", "validation", "smoke", "final"), default="dev")
    result.add_argument("--selection-manifest", type=Path)
    result.add_argument("--candidate-id")
    result.add_argument("--reviewed-final", action="store_true")
    return result


def absolute(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def settings_for(args, audit: dict) -> dict:
    adapter = absolute(args.adapter) if args.adapter else None
    if args.provider == "lora" and adapter is None:
        raise ValueError("LoRA requires an explicit adapter")
    if args.provider != "lora" and adapter is not None:
        raise ValueError("only LoRA accepts an adapter")
    if args.max_new_tokens < 1 or not args.model.strip():
        raise ValueError("explicit model and positive token limit required")
    adapter_files = None
    if adapter:
        files = [adapter / "adapter_model.safetensors", adapter / "adapter_config.json"]
        adapter_files = {p.name: sha256(p) for p in files}
    hosted = args.provider == "openai"
    if hosted and args.revision:
        raise ValueError("production OpenAI does not accept an HF revision")
    if args.split == "final" and not hosted and not (args.revision and len(args.revision) == 40 and all(c in "0123456789abcdef" for c in args.revision)):
        raise ValueError("final local model requires an immutable 40-character revision")
    return {"provider": args.provider, "model": args.model, "revision": args.revision,
            "adapter": str(adapter) if adapter else None, "adapter_sha256": adapter_files,
            "dtype": None if hosted else args.dtype, "device": None if hosted else args.device,
            "max_new_tokens": None if hosted else args.max_new_tokens,
            "decoding": {"mode": "production_responses_defaults_not_greedy", "store": False} if hosted else {
                "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0, "enable_thinking": False,
                "max_new_tokens": args.max_new_tokens, "recovery": False, "guardrails": False},
            "prompt_sha256": audit["prompt_sha256"], "scorer_sha256": audit["scorer_sha256"],
            "dependency_sha256": audit["dependency_sha256"]}


def create_provider(args, schema):
    if args.provider == "openai":
        from local_slm_lab.providers import OpenAIProductionProvider
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY required in process environment")
        provider = OpenAIProductionProvider(api_key=key, model=args.model, schema=schema)
        instrumented = instrument_openai(provider)
        return provider, {"raw_response_usage_instrumented": instrumented,
                          "raw_unavailable_on_sdk_parse_error": True, "monetary_cost": "unavailable",
                          "max_new_tokens": "not supported by pinned production client; CLI value not applied",
                          "prompt": "pinned fpy production prompts, separate from local v5"}
    from local_slm_lab.peft_provider import PeftInferenceConfig
    from local_slm_lab.v5_provider import V5TransformersProvider
    provider = V5TransformersProvider(PeftInferenceConfig(
        base_model=args.model, revision=args.revision, variant="custom" if args.provider == "lora" else "base",
        adapter_path=absolute(args.adapter) if args.adapter else None,
        dtype=args.dtype, device=args.device, max_new_tokens=args.max_new_tokens,
    ), schema)
    return provider, {"raw_response_usage_instrumented": True, "monetary_cost": "not_applicable"}


def model_artifacts(args, provider) -> dict:
    """Hash the actual local/HF snapshot if identifiable, never invent weight hashes."""
    if args.provider == "openai":
        return {"model_id": args.model, "weight_hashes": None, "reason": "hosted weights not exposed"}
    model_path = Path(args.model).expanduser()
    resolved = runtime_metadata(provider)["resolved_revision"]
    if not model_path.is_dir():
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(args.model, "config.json", revision=resolved or args.revision)
            model_path = Path(cached).parent if isinstance(cached, str) else model_path
        except (ImportError, ValueError, OSError):
            pass
    hashes = None
    if model_path.is_dir():
        files = sorted(p for p in model_path.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".tiktoken"})
        hashes = {p.name: sha256(p) for p in files}
    return {"model_id": args.model, "requested_revision": args.revision, "resolved_revision": resolved,
            "snapshot_files": hashes, "snapshot_sha256": digest(hashes) if hashes else None,
            "hash_availability": "observed_snapshot" if hashes else "unavailable"}


def run(args) -> dict:
    output = absolute(args.output)
    journal = output.with_name(output.name + ".calls.jsonl")
    # Exclusive reservations precede all cases access and all model construction.
    if journal.exists():
        raise FileExistsError("call journal already exists")
    with output.open("x", encoding="utf-8") as report_file:
        partial = {"report_version": SCORER_VERSION, "status": "partial", "started_at": now(),
                   "protocol": PROTOCOL, "records": [], "resumable": False,
                   "retry_policy": "no automatic final retry; retain marker and request explicit study adjudication"}
        def save(value):
            report_file.seek(0)
            json.dump(value, report_file, ensure_ascii=False, indent=2, allow_nan=False)
            report_file.write("\n")
            report_file.truncate()
            report_file.flush()
            os.fsync(report_file.fileno())
        save(partial)
        try:
            audit = provenance(args.provider)
            settings = settings_for(args, audit)
            cases, input_metadata = load_cases(
                absolute(args.cases), split=args.split,
                manifest_path=absolute(args.selection_manifest) if args.selection_manifest else None,
                reviewed_final=args.reviewed_final, candidate=args.candidate_id, settings=settings,
            )
            schema = load_schema()
            plan = build_call_plan(cases, schema)
            metadata = {"input": input_metadata, "settings": settings, "settings_sha256": digest(settings),
                        "provenance": audit, "runner_sha256": sha256(Path(__file__)),
                        "schema_sha256": digest(schema.model_dump(mode="json"))}
            partial["metadata"] = metadata
            if args.split == "final":
                marker = claim_final_run(input_metadata["sha256"], args.candidate_id, output,
                                         settings_sha256=digest(settings))
                metadata["final_run_marker"] = str(marker)
                save(partial)
            with journal.open("x", encoding="utf-8") as calls_file:
                def recorded(record):
                    partial["records"].append(record)
                    calls_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    calls_file.flush()
                    os.fsync(calls_file.fileno())
                provider, instrumentation = create_provider(args, schema)
                metadata["instrumentation"] = instrumentation
                metadata["runtime"] = runtime_metadata(provider)
                metadata["model_artifacts"] = model_artifacts(args, provider)
                save(partial)
                report = asyncio.run(evaluate(provider, cases, schema=schema, metadata=metadata, on_record=recorded, plan=plan))
                report["call_journal"] = str(journal)
                report["call_journal_sha256"] = sha256(journal)
                save(report)
                return report
        except BaseException as exc:
            partial["error"] = safe_error(exc)
            partial["finished_at"] = now()
            save(partial)
            raise


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": safe_error(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({"output": str(absolute(args.output)), "status": report["status"],
                      "nonintent_f1": report["metrics"]["nonintent"]["f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
