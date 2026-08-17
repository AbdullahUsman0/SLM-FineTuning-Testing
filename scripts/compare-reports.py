"""Create a compact metric delta table from two common benchmark reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "provider_success_rate",
    "intent_accuracy",
    "slot_micro_precision",
    "slot_micro_recall",
    "slot_micro_f1",
    "joint_extraction_accuracy",
    "correction_detection_accuracy",
    "forbidden_slot_inference_rate",
    "question_contract_accuracy",
    "one_question_compliance",
    "question_slot_relevance",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True)
    parser.add_argument("--openai", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    local = json.loads(Path(args.local).read_text(encoding="utf-8"))
    openai = json.loads(Path(args.openai).read_text(encoding="utf-8"))
    rows = []
    for metric in METRICS:
        local_value = local["metrics"][metric]
        openai_value = openai["metrics"][metric]
        rows.append(
            {
                "metric": metric,
                "local": local_value,
                "openai": openai_value,
                "local_minus_openai": round(local_value - openai_value, 4),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()
