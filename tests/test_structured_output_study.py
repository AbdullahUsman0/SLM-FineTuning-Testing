import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/prepare-structured-output-study.py"
    spec = importlib.util.spec_from_file_location("prepare_structured_output_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StructuredOutputStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = load_module()

    def test_declares_the_three_primary_arms_without_relabeling_pydantic(self):
        arms = self.study.ARMS
        self.assertEqual(set(arms), {"A", "B", "C"})
        self.assertEqual(arms["A"]["retries"], 0)
        self.assertEqual(arms["B"]["retries"], 1)
        self.assertIn("JSON Schema", arms["C"]["validation"])
        self.assertIn("Pydantic", arms["C"]["validation"])

    def test_frozen_record_hashes_context_gold_and_transition(self):
        call = {
            "key": ["scenario", "extract", 1], "scenario_id": "scenario",
            "cluster_id": "source", "category": "weather", "task": "extract",
            "context": {"message": "Forecast rain", "state": {}},
            "expected": {"updates": []}, "forbidden_slots": ["target_unit"],
            "gold_after": {"intent": "create_forecast", "slots": {}, "turns": []},
        }
        first = self.study.frozen_record(0, call)
        second = self.study.frozen_record(0, call)
        self.assertEqual(first, second)
        self.assertEqual(len(first["context_sha256"]), 64)
        self.assertEqual(len(first["gold_sha256"]), 64)
        self.assertNotIn("context", first)
        self.assertNotIn("expected", first)
        changed = dict(call, gold_after={"intent": "unsupported", "slots": {}, "turns": []})
        self.assertNotEqual(first["gold_sha256"], self.study.frozen_record(0, changed)["gold_sha256"])

    def test_schema_manifest_verification_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schema = root / "one.json"
            schema.write_text("{}\n", encoding="utf-8")
            metadata = {"bytes": schema.stat().st_size, "sha256": self.study.sha256(schema)}
            (root / "manifest.json").write_text(json.dumps({"files": {"one.json": metadata}}), encoding="utf-8")
            self.assertEqual(self.study.verified_schemas(root)["files"]["one.json"], metadata)
            schema.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema manifest mismatch"):
                self.study.verified_schemas(root)

    def test_existing_output_is_rejected_before_cases_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "already-there"
            output.mkdir()
            with patch.object(self.study, "repository_state", side_effect=AssertionError("must not inspect")), \
                 patch.object(self.study, "load_cases", side_effect=AssertionError("must not load")):
                with self.assertRaises(FileExistsError):
                    self.study.prepare(Path("cases"), Path("schemas"), output)


if __name__ == "__main__":
    unittest.main()
