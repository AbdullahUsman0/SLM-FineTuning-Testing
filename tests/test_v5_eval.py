import asyncio
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_slm_lab import v5_eval as v5
from forecasting_assistant.domain.models import ExtractorResult, QuestionOutput


def update(slot="target_column", value="sales", evidence="sales", status="provided"):
    return {"slot_id": slot, "candidate_value": json.dumps(value), "status": status,
            "confidence": 1.0, "evidence_text": evidence}


def extraction(updates=None, correction=False, intent="create_forecast"):
    return {"intent": intent, "intent_confidence": 1.0, "updates": updates or [],
            "correction_detected": correction, "unsupported_claims": []}


def scenario(sid="one", gold=None):
    return {"scenario_id": sid, "category": "tiny", "split": "validation", "initial_slots": {},
            "turns": [{"message": "sales date", "gold_extraction": gold or extraction([update()])}],
            "gold_final_slots": {"target_column": "sales"}, "question_cases": [],
            "must_not_infer": ["frequency"]}


def settings(provider="base"):
    return {"provider": provider, "model": "explicit-test-model", "revision": "a" * 40,
            "adapter": "test-adapter" if provider == "lora" else None,
            "adapter_sha256": {"adapter_model.safetensors": "b" * 64} if provider == "lora" else None,
            "dtype": "float32", "device": "cpu", "max_new_tokens": 100,
            "decoding": {"do_sample": False, "repetition_penalty": 1.0},
            "prompt_sha256": "c" * 64, "scorer_sha256": "d" * 64, "dependency_sha256": "e" * 64}


class FakeProvider:
    """No model, GPU, API, credential or env-file dependency."""
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.traces = []
        self.states = []
        self.requests = []

    def complete(self, task):
        item = next(self.outputs)
        fail = item is None or (isinstance(item, tuple) and item[0] == "fail")
        raw = item[1] if isinstance(item, tuple) else json.dumps(item) if item is not None else None
        syntax, schema, _ = v5.raw_validity(raw, task)
        self.traces.append({"operation": task, "raw_output": raw, "raw_json_valid": syntax,
                            "raw_schema_valid": schema, "parsed": not fail,
                            "prompt_tokens": 12, "completion_tokens": 4 if raw is not None else None,
                            "error": "SECRET_REQUEST_API_KEY" if fail else None})
        if fail:
            raise RuntimeError("SECRET_REQUEST_API_KEY")
        value = v5.strict_json(raw)
        return (ExtractorResult if task == "extract" else QuestionOutput).model_validate(value)

    async def extract(self, message, state):
        self.states.append(state.model_copy(deep=True))
        return self.complete("extract")

    async def ask(self, request):
        self.requests.append(request)
        return self.complete("ask")

    def drain_traces(self):
        result, self.traces = self.traces, []
        return result


def report(cases, outputs, provider="base"):
    return asyncio.run(v5.evaluate(FakeProvider(outputs), cases, metadata={
        "input": {"sha256": v5.digest(cases), "split": "validation"},
        "settings": settings(provider), "runtime": {"actual_dtype": "float32", "actual_device": "cpu"},
    }))


def manifest(declared=None):
    return {"manifest_version": "v5-selection-1", "selection_split": "validation",
            "selected_at": "2020-01-02T00:00:00+00:00", "final_cases_sha256": "f" * 64,
            "review": {"completed": True, "reviewer": "Named test reviewer",
                       "artifact_sha256": "1" * 64, "completed_at": "2020-01-01T00:00:00+00:00"},
            "candidates": [{"candidate_id": "base", "settings": declared or settings(),
                            "validation_report_sha256": "2" * 64}],
            "all_candidates_declared": True, "selected_candidate_id": "base"}


def runner_module():
    path = Path(v5.ROOT) / "scripts/evaluate-v5.py"
    spec = importlib.util.spec_from_file_location("test_v5_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V5MetricTests(unittest.TestCase):
    def test_oracle_is_perfect_and_zero_forbidden(self):
        gold = extraction([update(), update("time_column", "date", "date")], correction=True)
        case = scenario(gold=gold)
        case["question_cases"] = [{"slot_id": "target_unit", "ideal_question": "What unit is the target measured in?"}]
        result = report([case], [gold, {"question": "What unit is the target measured in?"}])
        metrics = result["metrics"]
        self.assertEqual(metrics["nonintent"]["f1"], 1)
        self.assertEqual(metrics["per_slot"]["target_column"]["support"], 1)
        self.assertEqual(metrics["correction_tuple_accuracy"]["rate"], 1)
        self.assertEqual(metrics["reducer_transition_accuracy"]["rate"], 1)
        self.assertEqual(metrics["forbidden_call_rate"]["rate"], 0)
        self.assertEqual(metrics["question_contract"]["rate"], 1)
        self.assertEqual(metrics["raw_json_valid"]["rate"], 1)
        self.assertEqual(metrics["tokens"]["prompt_tokens"]["total_observed"], 24)

    def test_failures_remain_in_gold_and_forbidden_denominators(self):
        cases = [scenario("one"), scenario("two")]
        wrong = extraction([update("frequency", {"periods": 1, "unit": "day"})])
        result = report(cases, [None, wrong])
        metrics = result["metrics"]
        self.assertEqual(metrics["nonintent"], v5.prf(0, 1, 2))
        self.assertEqual(metrics["forbidden_call_rate"], v5.rate(1, 2))
        self.assertEqual(metrics["forbidden_call_rate_conditional_success"], v5.rate(1, 1))
        self.assertEqual(metrics["provider_success"], v5.rate(1, 2))
        self.assertEqual(metrics["empty_update_collapse"], v5.rate(1, 2))
        self.assertNotIn("SECRET_REQUEST_API_KEY", json.dumps(result))

    def test_duplicate_gold_hard_fails_before_provider(self):
        gold = extraction([update(), update()])
        fake = FakeProvider([])
        with self.assertRaisesRegex(ValueError, "duplicate gold"):
            asyncio.run(v5.evaluate(fake, [scenario(gold=gold)]))
        self.assertFalse(fake.states)

    def test_duplicate_predictions_rejected_not_set_deduplicated(self):
        result = report([scenario()], [extraction([update(), update()])])
        self.assertEqual(result["metrics"]["nonintent"], v5.prf(0, 2, 1))
        self.assertEqual(result["metrics"]["duplicate_prediction_calls"], 1)
        self.assertEqual(result["metrics"]["raw_json_valid"]["rate"], 1)
        self.assertEqual(result["metrics"]["raw_schema_valid"]["rate"], 0)
        self.assertTrue(result["records"][0]["provider_success"])
        self.assertFalse(result["records"][0]["prediction_valid"])

    def test_duplicate_provider_rejection_still_counts_emissions(self):
        raw = json.dumps(extraction([update(), update()]))
        result = report([scenario()], [("fail", raw)])
        self.assertEqual(result["metrics"]["nonintent"], v5.prf(0, 2, 1))
        self.assertEqual(result["records"][0]["raw_output"], raw)

    def test_forbidden_gold_contradiction_is_hard_failure(self):
        case = scenario()
        case["must_not_infer"] = ["target_column"]
        with self.assertRaisesRegex(ValueError, "contradiction"):
            v5.build_call_plan([case])

    def test_empty_gold_scenarios_disclosed_failures_not_success(self):
        cases = [scenario("empty", extraction()), scenario("failed", extraction())]
        result = report(cases, [extraction(), None])
        metrics = result["metrics"]
        self.assertEqual(metrics["empty_gold_nonintent_scenarios"], 2)
        self.assertEqual(metrics["scenario_all_gold_nonintent"], v5.rate(1, 2))
        self.assertIsNone(metrics["scenario_all_gold_nonintent_nonempty"]["rate"])
        self.assertEqual(metrics["empty_update_collapse"]["denominator"], 0)

    def test_inclusive_intent_is_distinct_from_nonintent(self):
        gold = extraction([update("intent", "create_forecast"), update()])
        result = report([scenario(gold=gold)], [extraction([update("intent", "create_forecast")])])
        self.assertEqual(result["metrics"]["inclusive"], v5.prf(1, 0, 1))
        self.assertEqual(result["metrics"]["nonintent"], v5.prf(0, 0, 1))
        self.assertEqual(result["metrics"]["intent_accuracy"]["rate"], 1)

    def test_correction_flag_does_not_imply_tuple_correctness(self):
        gold = extraction([update()], correction=True)
        result = report([scenario(gold=gold)], [extraction([], correction=True)])
        self.assertEqual(result["metrics"]["correction_flag_accuracy"]["rate"], 1)
        self.assertEqual(result["metrics"]["correction_tuple_accuracy"]["rate"], 0)

    def test_reducer_failure_not_invented_success(self):
        bad = extraction([update(evidence="not in message")])
        result = report([scenario()], [bad])
        self.assertEqual(result["metrics"]["nonintent"]["f1"], 1)
        self.assertEqual(result["metrics"]["reducer_transition_accuracy"]["rate"], 0)
        self.assertEqual(result["records"][0]["transition_error"]["type"], "UnsupportedEvidenceError")

    def test_json_looking_string_decoded_once(self):
        gold = extraction([update(value="true")])
        result = report([scenario(gold=gold)], [gold])
        self.assertEqual(result["metrics"]["nonintent"]["f1"], 1)
        self.assertEqual(result["metrics"]["reducer_transition_accuracy"]["rate"], 1)
        self.assertEqual(result["records"][0]["emitted"]["updates"][0]["candidate_value"], "true")

    def test_schema_invalid_failed_call_retains_forbidden_emissions(self):
        raw = extraction([update("frequency", {"periods": 1, "unit": "day"})])
        del raw["updates"][0]["confidence"]
        result = report([scenario()], [("fail", json.dumps(raw))])
        self.assertEqual(result["metrics"]["nonintent"], v5.prf(0, 1, 1))
        self.assertEqual(result["metrics"]["forbidden_call_rate"], v5.rate(1, 1))
        self.assertEqual(result["metrics"]["forbidden_call_rate_conditional_success"], v5.rate(0, 0))

    def test_strict_json_vs_raw_schema(self):
        for raw, syntax, schema in [("{}", True, False), ("[]", True, False),
                                    ('```json\n{}\n```', False, False),
                                    ('{"a":1,"a":2}', False, False), ('{"x":NaN}', False, False)]:
            self.assertEqual(v5.raw_validity(raw, "extract")[:2], (syntax, schema))
        raw = extraction()
        raw["intent"] = []
        self.assertEqual(v5.raw_validity(json.dumps(raw), "extract")[:2], (True, False))
        raw = extraction([update()])
        raw["updates"][0]["confidence"] = "1.0"
        self.assertEqual(v5.raw_validity(json.dumps(raw), "extract")[:2], (True, False))
        raw["updates"][0]["confidence"] = 1
        raw["updates"][0]["candidate_value"] = "unencoded"
        self.assertEqual(v5.raw_validity(json.dumps(raw), "extract")[:2], (True, False))

    def test_status_and_typed_values_are_exact(self):
        gold = extraction([update(value="1")])
        for predicted in (extraction([update(value=1)]), extraction([update(value="1", status="inferred")])):
            result = report([scenario(gold=gold)], [predicted])
            self.assertEqual(result["metrics"]["nonintent"], v5.prf(0, 1, 1))


class V5ContextTests(unittest.TestCase):
    def test_teacher_forcing_and_last_question_survive_failed_turn(self):
        case = scenario()
        case["turns"].append({"message": "date", "last_assistant_question": "Which column contains the timestamps or periods?",
                              "gold_extraction": extraction([update("time_column", "date", "date")])})
        fake = FakeProvider([None, case["turns"][1]["gold_extraction"]])
        asyncio.run(v5.evaluate(fake, [case]))
        self.assertEqual(fake.states[1].slots["target_column"].value, "sales")
        self.assertEqual(fake.states[1].intent.value, "create_forecast")
        self.assertEqual(fake.states[1].turns[-1].assistant_message, case["turns"][1]["last_assistant_question"])

    def test_initial_slots_and_scenario_question(self):
        case = scenario()
        case["initial_slots"] = {"intent": "create_forecast", "source_reference": "data.csv"}
        case["last_assistant_question"] = "Which variable or column should be forecast?"
        plan = v5.build_call_plan([case])
        self.assertEqual(plan[0]["state"].intent.value, "create_forecast")
        self.assertEqual(plan[0]["state"].turns[-1].assistant_message, case["last_assistant_question"])

    def test_paraphrase_variants_share_parent_not_previous_variant(self):
        case = scenario()
        first = case["turns"][0]
        first.update(parent_id="one", turn_id="t1-v0")
        second = copy.deepcopy(first)
        second["turn_id"] = "t1-v1"
        case["turns"].append(second)
        plan = v5.build_call_plan([case])
        self.assertEqual(plan[0]["context"], plan[1]["context"])
        self.assertIsNone(plan[1]["state"].slots["target_column"].value)

    def test_question_is_oracle_conditioned_and_requested_slot_omitted(self):
        case = scenario()
        case["gold_final_slots"]["target_unit"] = "units"
        case["question_cases"] = [{"slot_id": "target_unit", "ideal_question": "What unit is the target measured in?"}]
        fake = FakeProvider([None, {"question": "What unit is the target measured in?"}])
        result = asyncio.run(v5.evaluate(fake, [case]))
        self.assertEqual(fake.requests[0].confirmed_context, {"target_column": "sales"})
        self.assertIn("oracle", result["protocol"]["question_context"])
        self.assertEqual(result["metrics"]["empty_update_collapse"]["denominator"], 1)

    def test_duplicate_scenarios_and_turn_ids_fail(self):
        with self.assertRaises(ValueError):
            v5.build_call_plan([scenario(), scenario()])
        case = scenario()
        case["turns"][0]["turn_id"] = "duplicate"
        case["turns"].append(copy.deepcopy(case["turns"][0]))
        with self.assertRaises(ValueError):
            v5.build_call_plan([case])


class V5BootstrapTests(unittest.TestCase):
    def pair(self):
        one = scenario("one")
        two = scenario("two", extraction([update(), update("time_column", "date", "date"), update("target_description", "sales", "sales")]))
        gold1, gold2 = one["turns"][0]["gold_extraction"], two["turns"][0]["gold_extraction"]
        return report([one, two], [gold1, None]), report([one, two], [gold1, gold2], "lora")

    def test_hand_calculated_two_cluster_f1_and_interval(self):
        left, right = self.pair()
        result = v5.paired_bootstrap(left, right)
        # Left counts 1 TP, 0 FP, 3 FN => F1=.4, not macro F1=.5. Right F1=1.
        # Bootstrap clusters: [one,one] delta 0, mixed .6, [two,two] delta 1.
        self.assertEqual(result["nonintent"]["delta"], .6)
        self.assertEqual(result["nonintent"]["ci95"], [0, 1])
        self.assertEqual(result["inclusive"]["delta"], .6)
        self.assertEqual(result["samples"], 10000)
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["nonintent"], v5.paired_bootstrap(left, right)["nonintent"])

    def test_source_clusters_keep_scenarios_and_variants_together(self):
        left, right = self.pair()
        for r in left["records"] + right["records"]:
            r["cluster_id"] = "shared-source"
        result = v5.paired_bootstrap(left, right)
        self.assertEqual(result["cluster_count"], 1)
        self.assertEqual(result["nonintent"]["ci95"], [.6, .6])

    def test_mismatched_keys_gold_context_scenarios_and_hashes_refused(self):
        for change in (lambda r: r["records"].pop(),
                       lambda r: r["records"][0]["expected"].update(intent="unsupported"),
                       lambda r: r["records"][0].update(context={}),
                       lambda r: r["scenario_ids"].append("missing"),
                       lambda r: r["metadata"]["input"].update(sha256="wrong"),
                       lambda r: r.update(status="partial")):
            left, right = self.pair()
            change(right)
            with self.assertRaises(ValueError):
                v5.paired_bootstrap(left, right)

    def test_runtime_mismatch_requires_labeled_override(self):
        left, right = self.pair()
        right["metadata"]["settings"]["dtype"] = "bfloat16"
        with self.assertRaises(ValueError):
            v5.paired_bootstrap(left, right)
        result = v5.paired_bootstrap(left, right, allow_runtime_difference=True)
        self.assertIn("dtype", result["runtime_differences"])
        self.assertTrue(result["runtime_difference_override"])

    def test_bootstrap_rejects_less_than_10000(self):
        with self.assertRaises(ValueError):
            v5.paired_bootstrap(*self.pair(), samples=9999)

    def test_validation_selection_uses_f1_not_loss(self):
        left, right = self.pair()
        # Ensure correction denominators exist without altering paired gold/keys.
        for r in left["records"] + right["records"]:
            r["expected"]["correction_detected"] = True
            if r["predicted"]:
                r["predicted"]["correction_detected"] = True
        # Failed calls with no raw response make JSON unavailable: fail closed.
        selected = v5.select_validation_candidate(left, {"loss-best": right})
        self.assertIsNone(selected["recommended_candidate"])
        left["records"][1]["raw_json_valid"] = False
        left["records"][1]["raw_schema_valid"] = False
        selected = v5.select_validation_candidate(left, {"f1-best": right})
        self.assertEqual(selected["recommended_candidate"], "f1-best")
        self.assertFalse(selected["review_completed"])
        self.assertFalse(selected["final_authorized"])
        right["metadata"]["input"]["split"] = "final"
        with self.assertRaises(ValueError):
            v5.select_validation_candidate(left, {"bad": right})


class V5SafetyTests(unittest.TestCase):
    def test_sealed_final_blocked_before_any_load(self):
        path = Path("C:/never-read/v5-sealed/v5-20260919-r1/final.jsonl")
        with patch.object(Path, "read_bytes", side_effect=AssertionError("must not read")), \
             patch.object(Path, "read_text", side_effect=AssertionError("must not read")):
            with self.assertRaises(PermissionError):
                v5.load_cases(path)
            with self.assertRaises(PermissionError):
                v5.load_cases(path, split="final")
            with self.assertRaises(PermissionError):
                v5.load_cases(path, split="final", manifest_path=Path("unused.json"), candidate="base", settings=settings())

    def test_manifest_requires_completed_review_and_exact_candidate(self):
        valid = manifest()
        self.assertIs(v5.authorize_final(valid, reviewed_final=True, candidate="base", settings=settings()), valid)
        for change in (lambda m: m["review"].update(completed=False),
                       lambda m: m["review"].update(reviewer="TODO"),
                       lambda m: m.update(all_candidates_declared=False),
                       lambda m: m["candidates"][0]["settings"].update(dtype="auto"),
                       lambda m: m["candidates"][0]["settings"].update(revision="main"),
                       lambda m: m["candidates"][0].update(validation_report_sha256="placeholder")):
            invalid = manifest()
            change(invalid)
            with self.assertRaises(ValueError):
                v5.authorize_final(invalid, reviewed_final=True, candidate="base", settings=settings())

    def test_final_hash_checked_before_parsing_fake_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text(json.dumps(manifest()), encoding="utf-8")
            fake_final = root / "final.jsonl"
            fake_final.write_text("intentionally not valid JSON", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cases hash differs"):
                v5.load_cases(fake_final, split="final", manifest_path=selection,
                              reviewed_final=True, candidate="base", settings=settings())

    def test_once_only_marker_is_exclusive_and_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = v5.claim_final_run("a" * 64, "base", root / "report.json", marker_root=root)
            self.assertTrue(marker.is_file())
            with self.assertRaises(FileExistsError):
                v5.claim_final_run("a" * 64, "base", root / "other.json", marker_root=root)

    def test_no_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            v5.write_report({"untouched": True}, output)
            with self.assertRaises(FileExistsError):
                v5.write_report({"untouched": False}, output)
            self.assertEqual(json.loads(output.read_text()), {"untouched": True})
            runner = runner_module()
            args = runner.parser().parse_args(["--provider", "base", "--model", "explicit", "--output", str(output)])
            with patch.object(runner, "load_cases", side_effect=AssertionError("must not load")):
                with self.assertRaises(FileExistsError):
                    runner.run(args)

    def test_runner_retains_partial_without_exception_secrets(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "validation.jsonl"
            cases.write_text(json.dumps(scenario()) + "\n", encoding="utf-8")
            output = root / "report.json"
            args = runner.parser().parse_args(["--provider", "base", "--model", "explicit", "--cases", str(cases), "--output", str(output)])
            with patch.object(runner, "provenance", return_value=settings()), \
                 patch.object(runner, "create_provider", side_effect=RuntimeError("SECRET_API_KEY")):
                with self.assertRaises(RuntimeError):
                    runner.run(args)
            text = output.read_text()
            self.assertNotIn("SECRET_API_KEY", text)
            self.assertEqual(json.loads(text)["status"], "partial")
            self.assertTrue(output.with_name(output.name + ".calls.jsonl").exists())

    def test_runner_success_preserves_journal_and_raw_calls(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "validation.jsonl"
            case = scenario()
            cases.write_text(json.dumps(case) + "\n", encoding="utf-8")
            output = root / "report.json"
            args = runner.parser().parse_args(["--provider", "base", "--model", "explicit", "--cases", str(cases),
                                              "--output", str(output), "--split", "validation"])
            fake = FakeProvider([case["turns"][0]["gold_extraction"]])
            with patch.object(runner, "provenance", return_value=settings()), \
                 patch.object(runner, "create_provider", return_value=(fake, {"mock": True})), \
                 patch.object(runner, "model_artifacts", return_value={"mock": True}):
                result = runner.run(args)
            persisted = json.loads(output.read_text())
            self.assertEqual(persisted["status"], "complete")
            self.assertEqual(persisted["metrics"]["nonintent"]["f1"], 1)
            journal = Path(result["call_journal"])
            self.assertEqual(result["call_journal_sha256"], v5.sha256(journal))
            self.assertEqual(json.loads(journal.read_text())["raw_output"], json.dumps(case["turns"][0]["gold_extraction"]))

    def test_candidate_rename_cannot_reclaim_same_final_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v5.claim_final_run("a" * 64, "first-name", root / "one.json", marker_root=root, settings_sha256="b" * 64)
            with self.assertRaises(FileExistsError):
                v5.claim_final_run("a" * 64, "new-name", root / "two.json", marker_root=root, settings_sha256="b" * 64)

    def test_final_marker_claim_precedes_model_construction(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            args = runner.parser().parse_args(["--provider", "base", "--model", "explicit", "--output", str(output),
                                              "--split", "final", "--revision", "a" * 40, "--candidate-id", "base"])
            events = []
            def claim(*args, **kwargs):
                events.append("claim")
                return root / "mock-marker.json"
            def create(*args):
                events.append("model")
                raise RuntimeError("mock interruption")
            # Fake cases only: no real final path is read or generated by this test.
            with patch.object(runner, "provenance", return_value=settings()), \
                 patch.object(runner, "load_cases", return_value=([scenario()], {"sha256": "f" * 64, "split": "final"})), \
                 patch.object(runner, "claim_final_run", side_effect=claim), \
                 patch.object(runner, "create_provider", side_effect=create):
                with self.assertRaises(RuntimeError):
                    runner.run(args)
            self.assertEqual(events, ["claim", "model"])
            self.assertEqual(json.loads(output.read_text())["status"], "partial")

    def test_openai_instrumentation_is_mock_only_and_redacts_exception(self):
        async def parse(**kwargs):
            return SimpleNamespace(output_text=json.dumps(extraction()), output_parsed=ExtractorResult.model_validate(extraction()),
                                   usage=SimpleNamespace(input_tokens=20, output_tokens=6))
        responses = SimpleNamespace(parse=parse)
        fake = SimpleNamespace(_client=SimpleNamespace(_client=SimpleNamespace(responses=responses)))
        self.assertTrue(v5.instrument_openai(fake))
        asyncio.run(responses.parse(text_format=ExtractorResult, secret="NOT_RECORDED"))
        traces = fake.drain_traces()
        self.assertEqual(traces[0]["prompt_tokens"], 20)
        self.assertNotIn("NOT_RECORDED", json.dumps(traces))


if __name__ == "__main__":
    unittest.main()
