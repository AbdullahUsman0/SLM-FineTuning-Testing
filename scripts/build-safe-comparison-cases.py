"""Create an external-provider-safe view of the v3 behavioral cases."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "evaluation" / "conversation-cases-v3.json"
OUTPUT = PROJECT_ROOT / "evaluation" / "conversation-cases-v3-safe.json"
EXCLUDED_CASE_IDS = {"sensitive-data"}


def main() -> None:
    cases = json.loads(SOURCE.read_text(encoding="utf-8"))
    safe_cases = [case for case in cases if case["case_id"] not in EXCLUDED_CASE_IDS]
    OUTPUT.write_text(json.dumps(safe_cases, indent=2), encoding="utf-8")
    print(f"Wrote {len(safe_cases)} safe cases to {OUTPUT}")


if __name__ == "__main__":
    main()
