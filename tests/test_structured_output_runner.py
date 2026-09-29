import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StructuredOutputRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load("test_structured_runner", "evaluate-structured-output.py")
        cls.resume = load("test_structured_resume", "evaluate-structured-output-resumable.py")
        cls.compare = load("test_structured_compare", "compare-structured-output.py")

    def args(self, arm):
        return SimpleNamespace(
            arm=arm, model="model", revision="a" * 40, adapter=Path("adapter"),
            dtype="float32", device="cpu", max_new_tokens=100,
            allow_high_call_count=False, openai_cost_approved=False,
        )

    def test_slot_wise_arm_requires_explicit_workload_gate_before_model_load(self):
        args = self.args("D")
        with patch.object(self.runner, "V5SlotWiseTransformersProvider",
                          side_effect=AssertionError("must not load model")):
            with self.assertRaisesRegex(PermissionError, "high-call-count"):
                self.runner.create_provider(args, object(), {})

    def test_paid_arm_requires_two_independent_cost_gates(self):
        args = self.args("F")
        args.adapter = None
        with patch.dict("os.environ", {"OPENAI_API_KEY": "unused"}, clear=True):
            with self.assertRaisesRegex(PermissionError, "V5_OPENAI_COST_APPROVED"):
                self.runner.create_provider(args, object(), {})
        args.openai_cost_approved = True
        with patch.dict("os.environ", {"OPENAI_API_KEY": "unused"}, clear=True):
            with self.assertRaises(PermissionError):
                self.runner.create_provider(args, object(), {})

    def test_warmup_accepts_validation_labelled_smoke_sample(self):
        class Provider:
            async def extract(self, message, state):
                return {"updates": [], "correction_detected": False}

            async def ask(self, request):
                return {"question": "Which value should be used?"}

            def drain_traces(self):
                return []

        result = asyncio.run(self.runner.warmup(
            Provider(),
            self.runner.load_schema(),
            ROOT / "corpus-v5/v5-20260919-r1/splits/smoke.jsonl",
        ))
        self.assertTrue(result["completed"])
        self.assertGreaterEqual(result["calls"], 1)

    def test_frozen_study_rejects_changed_context(self):
        call = {"key": ["s", "extract", 1], "scenario_id": "s", "cluster_id": "s",
                "category": "tiny", "task": "extract", "context": {"message": "one"},
                "expected": {"updates": []}, "forbidden_slots": [], "gold_after": {}}
        rows = self.runner.frozen_rows([call])
        manifest = {"status": "prepared_validation_only", "input": {"sha256": "f" * 64},
                    "call_plan": {"call_count": 1,
                                  "ordered_keys_sha256": self.runner.digest([rows[0]["key"]]),
                                  "ordered_context_gold_sha256": self.runner.digest([
                                      [rows[0]["key"], rows[0]["context_sha256"], rows[0]["gold_sha256"]]
                                  ]), "sha256": "a" * 64}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.runner.verify_frozen_study(path, {"split": "validation", "sha256": "f" * 64}, [call])
            call["context"] = {"message": "changed"}
            with self.assertRaisesRegex(ValueError, "context/gold"):
                self.runner.verify_frozen_study(path, {"split": "validation", "sha256": "f" * 64}, [call])

    def test_partial_scenario_is_preserved_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / ".scenario.partial"
            partial.write_text('{"records":[]}', encoding="utf-8")
            interrupted = root / "interrupted"
            self.resume.quarantine_partial(partial, interrupted)
            self.assertFalse(partial.exists())
            preserved = list(interrupted.iterdir())
            self.assertEqual(len(preserved), 1)
            self.assertIn("records", preserved[0].read_text())

    def test_comparison_rejects_controlled_model_mismatch(self):
        def report(arm, model):
            return {"status": "complete", "metadata": {
                "input": {"split": "validation"},
                "study": {"call_plan_sha256": "a", "sha256": "b"},
                "settings": {"study_arm": arm, "model": model, "revision": "r",
                             "adapter_sha256": {}, "dtype": "float16", "device": "cuda",
                             "max_new_tokens": 100, "prompt_sha256": "p"},
                "runtime": {},
            }}
        with self.assertRaisesRegex(ValueError, "model"):
            self.compare.assert_study_pair(report("A", "one"), report("C", "two"))


if __name__ == "__main__":
    unittest.main()
