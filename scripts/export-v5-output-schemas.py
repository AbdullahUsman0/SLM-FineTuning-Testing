#!/usr/bin/env python3
"""Export strict JSON Schemas for v5 constrained-decoding experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from forecasting_assistant.domain.models import ExtractorResult, QuestionOutput
from forecasting_assistant.domain.schema import load_schema


def _strict_objects(value: Any) -> None:
    """Match v5's exact-field wire contract for every declared object."""
    if isinstance(value, dict):
        if value.get("type") == "object" and isinstance(value.get("properties"), dict):
            value["additionalProperties"] = False
            value["required"] = list(value["properties"])
        for child in value.values():
            _strict_objects(child)
    elif isinstance(value, list):
        for child in value:
            _strict_objects(child)


def build_schemas() -> dict[str, dict[str, Any]]:
    extractor = ExtractorResult.model_json_schema(mode="validation")
    question = QuestionOutput.model_json_schema(mode="validation")
    _strict_objects(extractor)
    _strict_objects(question)

    slot_ids = [slot.slot_id for slot in load_schema().slots]
    extractor["$defs"]["SlotUpdate"]["properties"]["slot_id"]["enum"] = slot_ids

    for schema in (extractor, question):
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    extractor["$comment"] = (
        "Structural v5 contract. Duplicate slot IDs, evidence grounding, forbidden slots, "
        "and state-dependent semantics require application validation."
    )
    question["$comment"] = "Structural v5 question-output contract."
    return {"extractor-result": extractor, "question-output": question}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/v5-structured-output/schemas"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    schemas = build_schemas()
    slot_count = len(
        schemas["extractor-result"]["$defs"]["SlotUpdate"]["properties"]["slot_id"]["enum"]
    )
    manifest: dict[str, Any] = {
        "json_schema_dialect": "2020-12",
        "pydantic_version": importlib.metadata.version("pydantic"),
        "models": ["ExtractorResult", "QuestionOutput"],
        "slot_count": slot_count,
        "files": {},
    }
    for name, schema in schemas.items():
        path = args.output / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
