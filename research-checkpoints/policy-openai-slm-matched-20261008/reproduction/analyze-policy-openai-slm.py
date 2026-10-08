"""Summarize the fresh matched study and prepare human review, without inference."""
from __future__ import annotations
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime/fpy-v6-pinned/src"))
sys.path.append(str(ROOT / ".openai-deps"))
from forecasting_assistant.domain.schema import load_schema
from local_slm_lab.comparison_policy import ComparisonPolicy
from local_slm_lab.policy_providers import matched_primary_records
from local_slm_lab.v5_eval import digest, now, score_records, sha256, tuples

OUT = ROOT / "research-checkpoints/policy-openai-slm-matched-20261008"
REMOTE = ROOT / "runtime/returned-policy-gpu-20261008/training-runs"
RUNS = {"OpenAI_F": ROOT / "training-runs/policy-openai-F-20261007",
        "stock_2B": REMOTE / "policy-stock-bf16-20261007",
        "v5": REMOTE / "policy-v5-bf16-20261007",
        "v6": REMOTE / "policy-v6-bf16-20261007"}


def json_write(path, value):
    if path.exists():
        raise FileExistsError("Use a new output; preserve existing analysis")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def without_status(records):
    tp = fp = fn = 0
    for record in records:
        gold = Counter((slot, value) for slot, value, _ in tuples(record["expected"]) if slot != "intent")
        predicted = Counter((slot, value) for slot, value, _ in tuples(record.get("emitted")) if slot != "intent")
        overlap = sum((gold & predicted).values()) if record["prediction_valid"] else 0
        tp += overlap; fp += sum(predicted.values()) - overlap; fn += sum(gold.values()) - overlap
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "profile": "Supplemental exact slot/value without status, same structural gate; not the primary endpoint"}


def main():
    policy, schema = ComparisonPolicy(), load_schema()
    reports, sources, summary, verified_calls = {}, {}, [], 0
    for model, directory in RUNS.items():
        for track in ("component", "rollout"):
            path = directory / (track + "-report.json")
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["status"] != "complete":
                raise ValueError("Incomplete report")
            if score_records(report["records"], [s.slot_id for s in schema.slots]) != report["metrics"]:
                raise ValueError("Recorded primary metrics do not reproduce")
            for record in report["records"]:
                if model == "OpenAI_F":
                    attempt = record["policy_traces"][-1]["attempts"][-1]
                    admission = policy.parse(record["raw_output"], "extract", response_status=attempt["response_status"],
                                             refused=attempt["refused"], hit_generation_limit=attempt["hit_generation_limit"])
                else:
                    trace = record["policy_traces"][-1]
                    admission = policy.parse(record["raw_output"], "extract", hit_generation_limit=trace.get("hit_generation_limit", False))
                if admission.admitted != record["parser_admitted"] or admission.predicted != record["predicted"]:
                    raise ValueError("Recorded admission does not reproduce")
            verified_calls += len(report["records"])
            reports[model, track] = report
            sources[str(path.relative_to(ROOT))] = sha256(path)
            m = report["metrics"]
            value = dict(model=model, track=track, calls=len(report["records"]),
                f1=m["nonintent"]["f1"], precision=m["nonintent"]["precision"], recall=m["nonintent"]["recall"],
                JSON_valid=sum(r["raw_json_valid"] is True for r in report["records"]),
                declared_schema_valid=sum(r["raw_schema_valid"] is True for r in report["records"]),
                parser_admitted=sum(r["parser_admitted"] for r in report["records"]),
                scenario_exact=m["scenario_exact_nonintent"]["numerator"],
                forbidden_unknown_turns=m["forbidden_call_rate"]["numerator"],
                unknown_tests=m["forbidden_call_rate"]["denominator"],
                median_latency_seconds=m["latency_ms"]["p50"] / 1000,
                exact_without_status=without_status(report["records"]))
            summary.append(value)
    for track in ("component", "rollout"):
        hosted = reports["OpenAI_F", track]
        for model in ("stock_2B", "v5", "v6"):
            remote = reports[model, track]
            if hosted["protocol_fingerprint"] != remote["protocol_fingerprint"]:
                raise ValueError("Protocols differ")
            matched_primary_records(hosted["records"], remote["records"], policy=policy, schema=schema, track=track)
    # Supplemental human-review queue: one case per conversation, retaining every
    # prediction and existing gold. No gold updates or semantic repairs are made.
    cases = [json.loads(line) for line in (policy.directory / "primary-natural.jsonl").read_text(encoding="utf-8").splitlines()]
    queue = []
    for case in cases:
        item = {"scenario_id": case["scenario_id"], "domain": case["domain"], "review_status": "pending_human",
                "labels_changed": False, "turns": []}
        priority = 0
        for turn_index, turn in enumerate(case["turns"], 1):
            row = {"turn": turn_index, "message": turn["message"], "gold_wire": turn["gold_extraction"], "models": {}}
            for track in ("component", "rollout"):
                for model in ("OpenAI_F", "v5", "v6"):
                    record = next(r for r in reports[model, track]["records"] if r["scenario_id"] == case["scenario_id"] and r["key"][2] == turn_index)
                    gold = Counter(t for t in tuples(record["expected"]) if t[0] != "intent")
                    emitted = Counter(t for t in tuples(record.get("emitted")) if t[0] != "intent")
                    common = gold & emitted if record["prediction_valid"] else Counter()
                    missing = list((gold - common).elements())
                    extra = list((emitted - common).elements())
                    structural_failure = not record["prediction_valid"]
                    priority += 4 * structural_failure + len(missing) + len(extra)
                    row["models"][model + "/" + track] = dict(prediction_valid=record["prediction_valid"],
                        predicted=record["predicted"], emitted=record.get("emitted"),
                        raw_output=record["raw_output"], missing_exact_tuples=missing, extra_exact_tuples=extra,
                        transition_error=record.get("transition_error"),
                        context_sha256=record["context_sha256"], reference_report=str((RUNS[model] / (track + "-report.json")).relative_to(ROOT)))
            item["turns"].append(row)
        item["review_priority_score"] = priority
        item["review_questions"] = [
            "Do the free-text noun-phrase gold boundaries match the user's intended problem and target?",
            "Is any exact-match disagreement semantically acceptable? Record rationale without changing the frozen primary score.",
            "Does a horizon replacement warrant provided rather than confirmed under the fixed annotation rubric?",
            "Are unknown-turn updates genuinely invented values, explicit unknown markers, or other protocol violations?",
            "Which initial missing/wrong requirements explain incomplete accumulated state after correction?",
            "Is every emitted evidence span grounded in the current user turn?",
        ]
        queue.append(item)
    queue.sort(key=lambda item: (-item["review_priority_score"], item["scenario_id"]))
    json_write(OUT / "summary.json", {"created_at_utc": now(), "policy_sha256": policy.fingerprint,
        "summaries": summary, "source_report_sha256": sources, "verified_scored_calls": verified_calls,
        "all_primary_metrics_reproduced": True, "all_admissions_reproduced": True, "all_pairing_guards_passed": True,
        "human_review_queue_cases": len(queue), "human_review_completed": False,
        "new_inferences": 0, "paid_API_calls": 0, "sealed_final_accessed": False,
        "script_sha256": sha256(Path(__file__))})
    with (OUT / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in summary[0] if k != "exact_without_status"])
        writer.writeheader(); writer.writerows({k: v for k, v in row.items() if k != "exact_without_status"} for row in summary)
    with (OUT / "human-review-queue.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for item in queue:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"verified_scored_calls": verified_calls, "primary_metrics_reproduced": True,
                      "admissions_reproduced": True, "pairing_guards_passed": True,
                      "human_review_pending_cases": len(queue), "new_API_calls": 0}))


if __name__ == "__main__":
    main()
