"""Generate held-out guided cases without adding domain values to runtime code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "evaluation" / "guided-generalization-cases.json"


@dataclass(frozen=True)
class Domain:
    case_id: str
    target_column: str
    target_description: str
    target_unit: str
    time_column: str
    frequency_answer: str
    frequency: dict[str, float | str]
    horizon_answer: str
    horizon: dict[str, float | str]
    problem_statement: str
    business_goal: str
    source_reference: str
    output_granularity: str
    success_criteria: str
    timezone: str | None = None


DOMAINS = (
    Domain(
        "guided-wind-turbine",
        "power_kw",
        "wind turbine power output",
        "kW",
        "timestamp_utc",
        "hourly",
        {"periods": 1.0, "unit": "hour"},
        "48 hours",
        {"periods": 48.0, "unit": "hour"},
        "Wind turbine output planning",
        "grid dispatch",
        "wind_turbine_readings.csv",
        "hourly",
        "MAE below 30 kW",
        "UTC",
    ),
    Domain(
        "guided-pharmacy-prescriptions",
        "prescription_count",
        "daily pharmacy prescription count",
        "prescriptions",
        "service_date",
        "daily",
        {"periods": 1.0, "unit": "day"},
        "14 days",
        {"periods": 14.0, "unit": "day"},
        "Pharmacy prescription demand",
        "staff scheduling",
        "pharmacy_prescriptions.csv",
        "daily",
        "MAE below 12 prescriptions",
    ),
    Domain(
        "guided-port-containers",
        "container_arrivals",
        "port container arrivals",
        "containers",
        "week_start",
        "weekly",
        {"periods": 1.0, "unit": "week"},
        "8 weeks",
        {"periods": 8.0, "unit": "week"},
        "Port container arrival planning",
        "yard capacity planning",
        "port_container_arrivals.csv",
        "weekly",
        "MAE below 40 containers",
    ),
    Domain(
        "guided-university-enrollment",
        "new_enrollments",
        "monthly university enrollment",
        "students",
        "month_start",
        "monthly",
        {"periods": 1.0, "unit": "month"},
        "6 months",
        {"periods": 6.0, "unit": "month"},
        "University enrollment planning",
        "course capacity planning",
        "university_enrollment.csv",
        "monthly",
        "MAE below 25 students",
    ),
)


def build_case(domain: Domain) -> dict:
    return {
        "case_id": domain.case_id,
        "messages": [
            "yes",
            domain.target_column,
            domain.time_column,
            domain.frequency_answer,
            domain.horizon_answer,
            domain.problem_statement,
            "upload",
            domain.target_description,
            "single_series",
            domain.business_goal,
            domain.source_reference,
            domain.target_unit,
            "point",
            domain.output_granularity,
            "mae",
            domain.success_criteria,
            *([domain.timezone] if domain.timezone else []),
            "csv",
            "no",
        ],
        "expected_intent": "create_forecast",
        "expected_final_slots": {
            "intent": "create_forecast",
            "problem_statement": domain.problem_statement,
            "business_goal": domain.business_goal,
            "success_criteria": domain.success_criteria,
            "target_column": domain.target_column,
            "target_description": domain.target_description,
            "target_unit": domain.target_unit,
            "time_column": domain.time_column,
            "frequency": domain.frequency,
            "forecast_horizon": domain.horizon,
            "dataset_type": "single_series",
            "source_mode": "upload",
            "source_reference": domain.source_reference,
            "file_format": "csv",
            "forecast_type": "point",
            "output_granularity": domain.output_granularity,
            "primary_metric": "mae",
            "contains_sensitive_data": False,
            **({"timezone": domain.timezone} if domain.timezone else {}),
        },
    }


def main() -> None:
    cases = [build_case(domain) for domain in DOMAINS]
    OUTPUT.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} held-out guided cases to {OUTPUT}")


if __name__ == "__main__":
    main()
