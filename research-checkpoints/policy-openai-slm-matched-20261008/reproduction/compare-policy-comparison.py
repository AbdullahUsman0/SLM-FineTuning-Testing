"""Pair ONLY fresh completed policy-cohort runs; keep representation explicit."""
import argparse
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
from local_slm_lab.v5_eval import sha256


def compare(left, right):
    policy = ComparisonPolicy()
    spec = importlib.util.spec_from_file_location("policy_paired_bootstrap", ROOT / "scripts/compare-openai-output-methods.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    schema = load_schema()
    results = {}
    sources = {}
    for track in ("component", "rollout"):
        paths = [folder / f"{track}-report.json" for folder in (left, right)]
        a, b = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if any(report["status"] != "complete" for report in (a, b)):
            raise ValueError("Both tracks must be complete before comparison")
        if a["protocol_fingerprint"] != b["protocol_fingerprint"]:
            raise ValueError("Prospective run protocols differ")
        pairs = matched_primary_records(a["records"], b["records"], policy=policy, schema=schema, track=track)
        results[track] = dict(helper.bootstrap(pairs), left_arm=a["arm"], right_arm=b["arm"],
            left_representation=a["metadata"]["identity"]["representation"],
            right_representation=b["metadata"]["identity"]["representation"],
            matched_calls=len(pairs), left_metrics=a["metrics"], right_metrics=b["metrics"])
        sources.update({str(path.resolve()): sha256(path) for path in paths})
    return {"policy_sha256": policy.fingerprint, "comparisons": results, "source_sha256": sources,
            "bootstrap_script_sha256": sha256(ROOT / "scripts/compare-openai-output-methods.py"),
            "api_calls_made": 0, "limitation": "Agent-reviewed posthoc development cohort; no independent held-out superiority claim"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior comparisons; use a new output")
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"completed": True, "output": str(args.output)}))
