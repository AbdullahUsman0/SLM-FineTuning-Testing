"""Compare the three v3 validation reports with scenario-paired uncertainty."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fpy" / "src"))
from local_slm_lab.component_eval import _updates  # noqa: E402


def counts(records: list[dict]) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for record in records:
        if record["task"] != "extract":
            continue
        expected = {item for item in _updates(record["expected"]) if item[0] != "intent"}
        predicted = (
            {item for item in _updates(record["predicted"]) if item[0] != "intent"}
            if record["predicted"] is not None
            else set()
        )
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    return tp, fp, fn


def f1(values: tuple[int, int, int]) -> float:
    tp, fp, fn = values
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def grouped(records: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        output[record["scenario_id"]].append(record)
    return output


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reports-dir", default="research-checkpoints/2026-09-20-v3-validation"
    )
    parser.add_argument("--output", default="results/v3-validation-three-way-summary.json")
    args = parser.parse_args()

    names = {
        "stock_2b": "qwen35-2b-base-v3-validation.json",
        "v3_2b_lora": "qwen35-2b-lora-v3-validation.json",
        "openai_gpt5mini": "openai-gpt5mini-v3-validation.json",
    }
    reports_dir = ROOT / args.reports_dir
    reports = {
        key: json.loads((reports_dir / name).read_text(encoding="utf-8"))
        for key, name in names.items()
    }
    expected = [
        (r["scenario_id"], r["task"], r["turn_index"], r["expected"])
        for r in reports["stock_2b"]["records"]
    ]
    for key, report in reports.items():
        observed = [(r["scenario_id"], r["task"], r["turn_index"], r["expected"]) for r in report["records"]]
        if observed != expected:
            raise ValueError(f"Case/order/gold mismatch for {key}")

    groups = {key: grouped(report["records"]) for key, report in reports.items()}
    scenario_ids = list(groups["stock_2b"])
    variants = {}
    for key, report in reports.items():
        tp, fp, fn = counts(report["records"])
        variants[key] = {
            "model": report["model"],
            "non_intent_true_positive": tp,
            "non_intent_false_positive": fp,
            "non_intent_false_negative": fn,
            "non_intent_gold_count": tp + fn,
            "non_intent_precision": tp / (tp + fp) if tp + fp else 0.0,
            "non_intent_recall": tp / (tp + fn) if tp + fn else 0.0,
            "non_intent_f1": f1((tp, fp, fn)),
            "slot_micro_f1_including_intent": report["metrics"]["slot_micro_f1"],
            "intent_accuracy": report["metrics"]["intent_accuracy"],
            "provider_success_rate": report["metrics"]["provider_success_rate"],
            "forbidden_slot_inference_rate": report["metrics"]["forbidden_slot_inference_rate"],
            "question_contract_accuracy": report["metrics"]["question_contract_accuracy"],
            "call_errors": sum(bool(record["error"]) for record in report["records"]),
        }

    rng = random.Random(args.seed)
    comparisons = [("openai_gpt5mini", "v3_2b_lora"), ("openai_gpt5mini", "stock_2b")]
    samples = {f"{left}_minus_{right}": [] for left, right in comparisons}
    per_scenario_counts = {
        key: {scenario_id: counts(records) for scenario_id, records in group.items()}
        for key, group in groups.items()
    }
    for _ in range(args.resamples):
        drawn = rng.choices(scenario_ids, k=len(scenario_ids))
        scores = {}
        for key in reports:
            triple = tuple(
                sum(per_scenario_counts[key][scenario_id][index] for scenario_id in drawn)
                for index in range(3)
            )
            scores[key] = f1(triple)
        for left, right in comparisons:
            samples[f"{left}_minus_{right}"].append(scores[left] - scores[right])

    differences = {}
    for left, right in comparisons:
        label = f"{left}_minus_{right}"
        differences[label] = {
            "non_intent_f1_delta": variants[left]["non_intent_f1"] - variants[right]["non_intent_f1"],
            "ci_95_percentile": [percentile(samples[label], 0.025), percentile(samples[label], 0.975)],
        }
    summary = {
        "scenario_count": len(scenario_ids),
        "same_order_and_gold_verified": True,
        "bootstrap_resamples": args.resamples,
        "bootstrap_seed": args.seed,
        "variants": variants,
        "differences": differences,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
