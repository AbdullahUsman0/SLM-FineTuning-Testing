#!/usr/bin/env python3
"""Compare two completed structured-output arms on the frozen paired cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.v5_eval import paired_bootstrap, safe_error, sha256, strict_json, write_report

LOCAL_CONTROLLED = {"A", "B", "C"}


def rate_value(report: dict, name: str):
    value = report["metrics"].get(name)
    return value.get("rate") if isinstance(value, dict) else value


def delta(left, right):
    return None if left is None or right is None else right - left


def assert_study_pair(left: dict, right: dict) -> tuple[str, str]:
    for report in (left, right):
        if report.get("status") != "complete":
            raise ValueError("partial report cannot be compared")
        if report["metadata"]["input"]["split"] != "validation":
            raise ValueError("structured-output comparison requires validation reports")
        if not report["metadata"].get("study"):
            raise ValueError("report is not bound to the frozen study")
    ls, rs = left["metadata"]["study"], right["metadata"]["study"]
    if ls["call_plan_sha256"] != rs["call_plan_sha256"] or ls["sha256"] != rs["sha256"]:
        raise ValueError("study manifests or call plans differ")
    left_arm = left["metadata"]["settings"]["study_arm"]
    right_arm = right["metadata"]["settings"]["study_arm"]
    if left_arm == right_arm:
        raise ValueError("comparison requires two different arms")
    if {left_arm, right_arm} <= LOCAL_CONTROLLED:
        left_settings, right_settings = left["metadata"]["settings"], right["metadata"]["settings"]
        for field in ("model", "revision", "adapter_sha256", "dtype", "device", "max_new_tokens", "prompt_sha256"):
            if left_settings.get(field) != right_settings.get(field):
                raise ValueError(f"controlled A-C setting differs: {field}")
        for field in ("actual_device", "actual_dtype", "resolved_revision", "chat_template_sha256", "cuda_version"):
            if left["metadata"]["runtime"].get(field) != right["metadata"]["runtime"].get(field):
                raise ValueError(f"controlled A-C runtime differs: {field}")
    return left_arm, right_arm


def compare(left: dict, right: dict) -> dict:
    left_arm, right_arm = assert_study_pair(left, right)
    result = paired_bootstrap(left, right, samples=10000, seed=42, allow_runtime_difference=True)
    result.update({
        "comparison_type": "structured_output_research",
        "left_arm": left_arm,
        "right_arm": right_arm,
        "controlled_local_ablation": {left_arm, right_arm} <= LOCAL_CONTROLLED,
        "structural": {},
        "efficiency": {},
    })
    for metric in (
        "raw_json_valid", "raw_schema_valid", "first_pass_json_valid", "first_pass_schema_valid",
        "wrong_valid_schema_rate", "retry_rate", "retry_success_rate", "unknown_slot_call_rate",
        "forbidden_call_rate", "hallucinated_slot_call_rate",
    ):
        left_value, right_value = rate_value(left, metric), rate_value(right, metric)
        result["structural"][metric] = {"left": left_value, "right": right_value,
                                         "delta_right_minus_left": delta(left_value, right_value)}
    for metric in ("p50", "p95", "mean"):
        lv, rv = left["metrics"]["latency_ms"].get(metric), right["metrics"]["latency_ms"].get(metric)
        result["efficiency"][f"latency_ms_{metric}"] = {"left": lv, "right": rv,
                                                          "delta_right_minus_left": delta(lv, rv)}
    for metric in ("prompt_tokens", "completion_tokens"):
        lv = left["metrics"]["tokens"][metric]["total_observed"]
        rv = right["metrics"]["tokens"][metric]["total_observed"]
        result["efficiency"][metric] = {"left": lv, "right": rv, "delta_right_minus_left": rv - lv}
    result["efficiency"]["model_calls"] = {
        "left": left["metrics"]["model_calls"]["total"],
        "right": right["metrics"]["model_calls"]["total"],
        "delta_right_minus_left": right["metrics"]["model_calls"]["total"] - left["metrics"]["model_calls"]["total"],
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if output.exists() or not output.parent.is_dir():
            raise FileExistsError("comparison output must be new and its parent must exist")
        left_path, right_path = args.left.resolve(), args.right.resolve()
        left, right = strict_json(left_path.read_text(encoding="utf-8")), strict_json(right_path.read_text(encoding="utf-8"))
        result = compare(left, right)
        result["reports"] = {"left": {"path": str(left_path), "sha256": sha256(left_path)},
                             "right": {"path": str(right_path), "sha256": sha256(right_path)}}
        result["comparison_script_sha256"] = sha256(Path(__file__))
        write_report(result, output)
        print(json.dumps({"output": str(output), "nonintent": result["nonintent"],
                          "arms": [result["left_arm"], result["right_arm"]]}))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
