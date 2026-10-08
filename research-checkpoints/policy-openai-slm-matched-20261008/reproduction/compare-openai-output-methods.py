"""Compare completed OpenAI study jobs, with explicit matched cohort coverage."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime/fpy-v6-pinned/src"))
from forecasting_assistant.domain.schema import load_schema
from local_slm_lab.openai_output_methods import atomic_json
from local_slm_lab.v5_eval import confusion, now, score_records, sha256


def matched(left, right, track):
    """Pair entire completed scenarios; do not select successful calls."""
    def index(records):
        values = {tuple(r["key"]): r for r in records}
        if len(values) != len(records):
            raise ValueError("Duplicate call key")
        return values
    li, ri = index(left), index(right)
    if li.keys() != ri.keys():
        raise ValueError("Completed matched scenarios have different call keys")
    pairs = []
    for key in sorted(li, key=str):
        a, b = li[key], ri[key]
        for field in ("scenario_id", "cluster_id", "task", "expected", "gold_sha256", "forbidden_slots"):
            if a.get(field) != b.get(field):
                raise ValueError(f"Paired gold differs: {field}")
        if track == "component" and (a["context"] != b["context"] or a["context_sha256"] != b["context_sha256"]):
            raise ValueError("Component contexts differ")
        # Rollout deliberately uses each arm's own predicted previous state.
        pairs.append((a, b))
    return pairs


def bootstrap(pairs, samples=10000, seed=42):
    import numpy as np
    if samples < 10000:
        raise ValueError("Use at least 10000 resamples")
    clusters = sorted({a["cluster_id"] for a, _ in pairs})
    if not clusters:
        raise ValueError("No matched scenarios")
    index = {cluster: i for i, cluster in enumerate(clusters)}
    counts = np.zeros((len(clusters), 2, 3), dtype=np.int64)
    for a, b in pairs:
        if a["task"] == "extract":
            for arm, record in enumerate((a, b)):
                counts[index[a["cluster_id"]], arm] += confusion(record, nonintent=True)
    def f1(value):
        tp, fp, fn = value[..., 0], value[..., 1], value[..., 2]
        den = 2 * tp + fp + fn
        return np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)
    point = f1(counts.sum(axis=0))
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, samples, 256):
        selected = rng.integers(0, len(clusters), size=(min(256, samples - start), len(clusters)))
        scores = f1(counts[selected].sum(axis=1))
        deltas.extend(scores[:, 1] - scores[:, 0])
    return {"left_f1": float(point[0]), "right_f1": float(point[1]),
            "delta_right_minus_left": float(point[1] - point[0]),
            "ci95": [float(v) for v in np.quantile(deltas, [.025, .975])],
            "samples": samples, "seed": seed, "cluster_count": len(clusters),
            "interval": "paired source-scenario percentile 95%; recomputed exact micro F1",
            "inferential_warning": "Tiny convenience cohorts, particularly the one-scenario D pilot, do not support general superiority claims"}


def efficiency(records):
    eligible = [r["latency_ms"] for r in records if r.get("timing_eligible")]
    return {"logical_calls": len(records), "model_calls": sum(r.get("model_calls") or 0 for r in records),
            "usage_cost_usd": sum(r.get("usage_cost_usd") or 0 for r in records),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in records),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in records),
            "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in records),
            "cached_input_tokens": sum(r.get("cached_input_tokens") or 0 for r in records),
            "latency_mean_ms_excluding_replays": sum(eligible) / len(eligible) if eligible else None,
            "timing_eligible_calls": len(eligible),
            "latency_note": "Wall time includes API/network and configured request pacing; D uses concurrent probes"}


def compare_run(directory, baseline="A"):
    manifest = json.loads((directory / "run-manifest.json").read_text(encoding="utf-8"))
    groups = defaultdict(dict)
    hashes = {}
    for path in sorted((directory / "jobs").glob("*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["status"] != "complete" or job["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Partial or incompatible job file")
        group = groups[(job["track"], job["arm"])]
        if job["scenario_id"] in group:
            raise ValueError("Duplicate scenario report")
        group[job["scenario_id"]] = job["records"]
        hashes[path.name] = sha256(path)
    schema_ids = [slot.slot_id for slot in load_schema().slots]
    comparisons = {}
    for (track, arm), jobs in sorted(groups.items()):
        if arm == baseline or (track, baseline) not in groups:
            continue
        base = groups[(track, baseline)]
        shared = sorted(jobs.keys() & base.keys())
        if not shared:
            continue
        left = [r for sid in shared for r in base[sid]]
        right = [r for sid in shared for r in jobs[sid]]
        result = bootstrap(matched(left, right, track))
        result.update(track=track, left_arm=baseline, right_arm=arm,
                      matched_scenario_ids=shared, matched_scenario_count=len(shared),
                      left_completed_scenarios=len(base), right_completed_scenarios=len(jobs),
                      left_unmatched=sorted(base.keys() - jobs.keys()),
                      right_unmatched=sorted(jobs.keys() - base.keys()),
                      left_metrics=score_records(left, schema_ids), right_metrics=score_records(right, schema_ids),
                      left_efficiency=efficiency(left), right_efficiency=efficiency(right))
        comparisons[f"{track}/{baseline}-vs-{arm}"] = result
    return {"created_at": now(), "fingerprint": manifest["fingerprint"],
            "manifest_sha256": sha256(directory / "run-manifest.json"), "job_sha256": hashes,
            "comparisons": comparisons, "baseline": baseline,
            "coverage_rule": "Intersection of entire completed scenarios, retaining failed model outputs; unmatched coverage is disclosed",
            "C_reference": "F is the hosted schema counterpart; XGrammar was not run",
            "rollout_rule": "Gold, call keys and source clusters match; predicted previous contexts may differ",
            "multiple_comparisons": "Exploratory intervals, no multiplicity correction; no confirmatory superiority claim",
            "comparison_script_sha256": sha256(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="A")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new comparison output filename")
    result = compare_run(args.run_dir.resolve(), args.baseline)
    atomic_json(args.output, result)
    print(json.dumps({"comparisons": len(result["comparisons"]), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
