"""Run a small repeatable baseline evaluation against the local server."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from .chat import SYSTEM_PROMPT
from .client import LocalModelConfig, chat_completion


def load_cases(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_answer(case: dict[str, object], answer: str) -> dict[str, bool]:
    normalized = answer.lower()
    required = [str(value).lower() for value in case.get("must_contain", [])]
    forbidden = [str(value).lower() for value in case.get("must_not_contain", [])]
    checks = {
        "contains_required_text": all(value in normalized for value in required),
        "avoids_forbidden_text": all(value not in normalized for value in forbidden),
    }
    if "max_question_marks" in case:
        checks["question_limit"] = answer.count("?") <= int(case["max_question_marks"])
    if "max_numbered_items" in case:
        numbered_items = len(re.findall(r"^\s*\d+[.)]", answer, flags=re.MULTILINE))
        checks["numbered_item_limit"] = numbered_items <= int(case["max_numbered_items"])
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--cases", default="evaluation/cases.jsonl")
    parser.add_argument("--output", default="results/latest.json")
    args = parser.parse_args()

    config = LocalModelConfig.from_file(args.config)
    results = []

    for case in load_cases(Path(args.cases)):
        started = time.perf_counter()
        answer = chat_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(case["prompt"])},
            ],
            config,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        checks = check_answer(case, answer)
        results.append(
            {
                "id": case["id"],
                "passed": all(checks.values()),
                "checks": checks,
                "latency_ms": elapsed_ms,
                "response": answer,
            }
        )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": config.model,
        "case_count": len(results),
        "passes": sum(item["passed"] for item in results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} results to {output}")


if __name__ == "__main__":
    main()
