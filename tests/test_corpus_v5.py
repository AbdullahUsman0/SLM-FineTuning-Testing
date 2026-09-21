"""Only miniature in-memory/temporary fixtures; never load official final data."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from local_slm_lab import corpus_v5 as v5
from local_slm_lab.component_eval import question_request
from forecasting_assistant.domain.models import ExtractorResult, QuestionOutput
from forecasting_assistant.prompts.llmrei_long import validate_question


class CorpusV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = v5.load_schema()
        cls.plan = v5.freeze_plan(20, "v5-miniature")
        # One domain with twenty small source scenarios, plus heldout fixtures.
        rows = [r for r in cls.plan["assignments"] if r["domain"] in {"sales_demand", "synthetic_health_ops"}]
        cls.cases = [v5.build_scenario(r, cls.plan["version"], cls.schema) for r in rows]

    def case(self, category):
        return deepcopy(next(s for s in self.cases if s["category"] == category))

    def test_miniature_structure_and_all_schema_prompts(self):
        self.assertEqual(len(self.schema.slots), 79)
        self.assertEqual(set(v5.CLAUSES) | set(v5.BOOL_CLAUSES), {s.slot_id for s in self.schema.slots})
        report = v5.validate_collection(self.cases, self.plan)
        self.assertEqual(report["exact_model_input_duplicates"], 0)
        for case in self.cases:
            for example in v5.sft_examples(case):
                if example["metadata"]["task"] == "extract":
                    user = json.loads(example["messages"][1]["content"])
                    self.assertEqual(len(user["slot_definitions"]), 79)
                    ExtractorResult.model_validate_json(example["messages"][-1]["content"])
                else:
                    self.assertTrue(validate_question(QuestionOutput.model_validate_json(example["messages"][-1]["content"]), question_request(case, self.schema)))

    def test_frozen_group_split_and_heldout_domain_family(self):
        self.assertEqual(self.plan, v5.freeze_plan(20, "v5-miniature"))
        for row in self.plan["assignments"]:
            if row["domain"] == "synthetic_health_ops":
                self.assertEqual(row["split"], "final")
            self.assertEqual(row["template_family"] == "heldout_handover", row["split"] == "final")
        for case in self.cases:
            self.assertTrue(all(e["metadata"]["split"] == case["split"] for e in v5.sft_examples(case)))

    def test_no_relabeling_or_duplicate_scenario_across_splits(self):
        case = self.case("medium")
        case["split"] = "validation" if case["split"] != "validation" else "train"
        with self.assertRaisesRegex(ValueError, "frozen split"):
            v5.validate_collection([case], self.plan)
        with self.assertRaisesRegex(ValueError, "duplicate scenario"):
            v5.validate_collection([self.cases[0], self.cases[0]], self.plan)

    def test_invalid_evidence_rejected(self):
        case = self.case("medium")
        case["turns"][0]["gold_extraction"]["updates"][0]["evidence_text"] = "not present in the message"
        with self.assertRaisesRegex(ValueError, "evidence"):
            v5.validate_scenario(case)

    def test_invalid_span_rejected(self):
        case = self.case("medium")
        case["turns"][0]["evidence_spans"][0]["end"] += 1
        with self.assertRaisesRegex(ValueError, "evidence"):
            v5.validate_scenario(case)

    def test_number_mutation_rejected(self):
        case = self.case("correction")
        update = case["turns"][-1]["gold_extraction"]["updates"][0]
        update["candidate_value"] = json.dumps({"periods": 999, "unit": "day"})
        with self.assertRaisesRegex(ValueError, "fact-label"):
            v5.validate_scenario(case)

    def test_polarity_mutation_in_text_rejected(self):
        row = {**self.plan["assignments"][0], "category": "medium", "index": 30}
        case = v5.build_scenario(row, "v5-miniature")
        # Mutate a fact's literal clause and evidence together: the atom remains
        # the independent oracle, rather than merely checking substring presence.
        fact = v5.atom(case["scenario_id"], "t1", "contains_sensitive_data", False)
        row["template_family"] = "direct_requirements"
        case["template_family"] = row["template_family"]
        turn = v5.make_turn(row, [fact], {}, 0, "medium", "t1-v0", row["scenario_id"])
        case["source_facts"] = [fact]
        case["turns"] = [turn]
        turn["message"] = turn["message"].replace("contains no", "contains")
        turn["gold_extraction"]["updates"][0]["evidence_text"] = turn["message"]
        turn["evidence_spans"][0].update(text=turn["message"], end=len(turn["message"]))
        with self.assertRaisesRegex(ValueError, "polarity"):
            v5.validate_scenario(case)

    def test_polarity_mutation_in_label_rejected(self):
        case = next(deepcopy(s) for s in self.cases if any(u["slot_id"] in v5.BOOL_CLAUSES for t in s["turns"] for u in t["gold_extraction"]["updates"]))
        update = next(u for t in case["turns"] for u in t["gold_extraction"]["updates"] if u["slot_id"] in v5.BOOL_CLAUSES)
        update["candidate_value"] = json.dumps(not json.loads(update["candidate_value"]))
        with self.assertRaisesRegex(ValueError, "fact-label"):
            v5.validate_scenario(case)

    def test_canonical_numeric_enum_and_boolean_rejections(self):
        bad = [("minimum_training_points", True), ("backtest_folds", 2.5),
               ("forecast_horizon", {"periods": 0, "unit": "day"}),
               ("forecast_horizon", {"periods": 1, "unit": "days"}),
               ("forecast_horizon", {"periods": float("nan"), "unit": "day"}),
               ("contains_sensitive_data", "false"), ("minimum_coverage", 101),
               ("quantiles", [1.5]), ("seasonal_periods", [True]),
               ("file_format", "pdf"), ("target_description", {"value": "description"}),
               ("output_granularity", {"periods": 1, "unit": "day"})]
        for slot, value in bad:
            with self.subTest(slot=slot, value=value), self.assertRaises(ValueError):
                v5.canonical_check(slot, value, self.schema)

    def test_status_and_question_contract_mutations_rejected(self):
        case = self.case("medium")
        case["turns"][0]["gold_extraction"]["updates"][0]["status"] = "unmentioned"
        with self.assertRaises(ValueError):
            v5.validate_scenario(case)
        case = self.case("medium")
        case["ideal_question"] = "What is the value? Also what is the time column?"
        with self.assertRaisesRegex(ValueError, "question contract"):
            v5.validate_scenario(case)

    def test_correction_and_confirmed_conflict_reducer_states(self):
        correction, conflict = self.case("correction"), self.case("confirmed_conflict")
        for case in (correction, conflict):
            v5.validate_scenario(case)
            self.assertEqual(len(case["turns"]), 4)
        self.assertEqual(correction["gold_final_statuses"]["forecast_horizon"], "provided")
        self.assertEqual(conflict["gold_final_statuses"]["forecast_horizon"], "conflicting")
        self.assertEqual(conflict["gold_final_slots"]["forecast_horizon"], conflict["turns"][-1]["context_slots"]["forecast_horizon"])
        self.assertNotEqual(correction["gold_final_slots"]["forecast_horizon"], correction["turns"][-1]["context_slots"]["forecast_horizon"])

    def test_selected_short_and_abstentions(self):
        v5.validate_scenario(self.case("selected_short"))
        for i in range(132, 136):
            row = {**self.plan["assignments"][0], "index": i, "category": "abstention"}
            case = v5.build_scenario(row, "v5-miniature")
            v5.validate_scenario(case)
            self.assertTrue(all(not t["gold_extraction"]["updates"] for t in case["turns"]))
            self.assertIn("problem_statement", case["must_not_infer"])

    def test_all_short_answer_rules_on_miniature_fixtures(self):
        selected = set()
        for i in range(114, 132):
            row = {**self.plan["assignments"][0], "index": i, "category": "selected_short"}
            case = v5.build_scenario(row, "v5-miniature")
            v5.validate_scenario(case)
            selected.add(case["turns"][0]["selected_slot"])
        self.assertEqual(len(selected), 8)

    def test_all_value_types_and_unredacted_clause_evidence(self):
        row = self.plan["assignments"][0]
        values = v5.source_values("sales_demand", 7)
        for slot, value in values.items():
            v5.canonical_check(slot, value, self.schema)
            if slot == "authentication_reference":
                continue  # explicitly documented prompt sanitizer exclusion
            fact = v5.atom(row["scenario_id"], "t1", slot, value)
            for family in ("direct_requirements", "heldout_handover"):
                for variant in range(3):
                    turn = v5.make_turn({**row, "template_family": family}, [fact], {}, variant, "medium", "t1-v0", row["scenario_id"])
                    payload = json.loads(v5.build_v5_extractor_input(turn["message"], v5.turn_state(turn, self.schema), self.schema))
                    self.assertEqual(payload["current_message"], turn["message"])

    def test_parent_group_leakage_rejected(self):
        case = self.case("medium")
        case["turns"][0]["parent_id"] = "another-scenario/t1"
        with self.assertRaisesRegex(ValueError, "parent crosses"):
            v5.validate_scenario(case)

    def test_filename_does_not_imply_columns_or_problem_statement(self):
        # A small unit fixture, not an official source scenario.
        row = self.plan["assignments"][0]
        fact = v5.atom(row["scenario_id"], "t1", "source_reference", "simulated_upload.csv")
        turn = v5.make_turn(row, [fact], {}, 0, "medium", "t1-v0", row["scenario_id"])
        self.assertEqual({u["slot_id"] for u in turn["gold_extraction"]["updates"]}, {"source_reference"})

    def test_free_text_and_unicode_offsets_are_exact(self):
        row = self.plan["assignments"][0]
        value = 'Keep "ZERO" days — Café; do NOT paraphrase'
        fact = v5.atom(row["scenario_id"], "t1", "problem_statement", value)
        turn = v5.make_turn(row, [fact], {}, 0, "medium", "t1-v0", row["scenario_id"])
        self.assertEqual(json.loads(turn["gold_extraction"]["updates"][0]["candidate_value"]), value)
        span = turn["evidence_spans"][0]
        self.assertEqual(turn["message"][span["start"]:span["end"]], span["text"])

    def test_normalized_input_dedup_detects_case_and_punctuation(self):
        self.assertEqual(v5.normalized_input("HELLO, café!"), v5.normalized_input("hello café"))
        cases = self.cases[:2]
        example = next(v5.sft_examples(cases[0]))
        other = deepcopy(example)
        other["messages"][1]["content"] = other["messages"][1]["content"].upper()
        with patch.object(v5, "sft_examples", side_effect=[[example], [other]]):
            with self.assertRaisesRegex(ValueError, "duplicate exact/normalized"):
                v5.validate_collection(cases, self.plan)

    def test_exact_input_dedup(self):
        example = next(v5.sft_examples(self.cases[0]))
        with patch.object(v5, "sft_examples", return_value=[example]):
            with self.assertRaisesRegex(ValueError, "duplicate exact/normalized"):
                v5.validate_collection(self.cases[:2], self.plan)

    def test_overlap_is_measured_and_review_pending_not_zero_claim(self):
        left, right = deepcopy(self.cases[0]), deepcopy(self.cases[0])
        right["scenario_id"] += "-other"
        right["split"] = "final" if left["split"] != "final" else "train"
        report, pairs = v5.overlap_report([left, right])
        self.assertEqual(report["comparisons"], 1)
        self.assertEqual(report["flagged_pairs"], 1)
        self.assertEqual(report["reviewed_pairs"], 0)
        self.assertEqual(pairs[0]["text_jaccard"], 1)

    def test_review_all_final_and_uncertain_and_stratified_train(self):
        decisions = v5.review_decisions(self.cases)
        for case, decision in zip(self.cases, decisions):
            if case["split"] == "final" or case["category"] in {"correction", "confirmed_conflict", "abstention", "selected_short"}:
                self.assertTrue(decision["human_review_required"])
            self.assertEqual(decision["reviewer_status"], "machine_validated_pending_human")
            self.assertEqual(decision["rejection_reasons"], [])


class CorpusV5ArtifactTests(unittest.TestCase):
    def test_temporary_build_lf_seal_hashes_smoke_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public, sealed = root / "public", root / "sealed"
            real_builder = v5.build_scenario
            def ensure_frozen(*args, **kwargs):
                self.assertTrue((public / "split-freeze.json").exists())
                return real_builder(*args, **kwargs)
            with patch.object(v5, "build_scenario", side_effect=ensure_frozen):
                manifest = v5.write_artifacts(public, sealed, per_domain=5, version="v5-miniature", check_dependency=False)
            self.assertEqual(manifest["statistics"]["source_scenarios"], 55)
            self.assertFalse((public / "splits/final.jsonl").exists())
            self.assertFalse((public / "sft/final.jsonl").exists())
            self.assertTrue((sealed / "splits/final.jsonl").exists())
            for file in list(public.rglob("*.jsonl")) + list(sealed.rglob("*.jsonl")):
                data = file.read_bytes()
                self.assertNotIn(b"\r\n", data)
                self.assertTrue(not data or data.endswith(b"\n"))
                data.decode("utf-8")
            result = v5.verify_artifacts(public, sealed)
            self.assertFalse(result["sealed_labels_parsed"])
            frozen_hash = v5._sha256(public / "manifest.json")
            with self.assertRaises(FileExistsError):
                v5.write_artifacts(public, sealed, per_domain=5, check_dependency=False)
            self.assertEqual(v5._sha256(public / "manifest.json"), frozen_hash)
            with (sealed / "sft/final.jsonl").open("ab") as stream:
                stream.write(b" \n")
            with self.assertRaisesRegex(ValueError, "hash/size"):
                v5.verify_artifacts(public, sealed)

    def test_failed_build_marked_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            public, sealed = Path(directory) / "public", Path(directory) / "sealed"
            with patch.object(v5, "build_scenario", side_effect=ValueError("do not print per-case labels")):
                with self.assertRaisesRegex(RuntimeError, "marked failed"):
                    v5.write_artifacts(public, sealed, per_domain=5, check_dependency=False)
            status = json.loads((public / "build-status.json").read_bytes())
            self.assertEqual(status["status"], "failed_do_not_reuse_version")
            self.assertNotIn("per-case labels", json.dumps(status))
            with self.assertRaises(FileExistsError):
                v5.write_artifacts(public, sealed, per_domain=5, check_dependency=False)

    def test_reject_nested_sealed_root(self):
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "public"
            with self.assertRaisesRegex(ValueError, "separate"):
                v5.write_artifacts(public, public / "sealed", per_domain=5, check_dependency=False)


if __name__ == "__main__":
    unittest.main()
