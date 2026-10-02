"""Miniature fixtures only; official final labels are never loaded by tests."""
import gzip
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from local_slm_lab import corpus_v6 as v6


class CorpusV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=v6.freeze_plan(20,"v6-miniature")
        cls.schema=v6.base.load_schema()
        cls.cases=[v6.build_scenario(row,cls.plan["version"],cls.schema) for row in cls.plan["assignments"]
                   if row["domain"] in {"temperature","inflation_prices","crypto_spot","weather_energy"}]

    def test_mixture_and_group_splits_are_frozen(self):
        plan=v6.freeze_plan(100,"v6-miniature")
        self.assertEqual(len(plan["assignments"]),2000)
        self.assertEqual(plan,v6.freeze_plan(100,"v6-miniature"))
        from collections import Counter
        self.assertEqual(Counter(r["domain_family"] for r in plan["assignments"]),
                         {"weather":600,"economics":1200,"combined":200})
        self.assertEqual(Counter(r["split"] for r in plan["assignments"]),
                         {"train":1400,"validation":300,"final":300})
        groups={}
        for row in plan["assignments"]:
            groups.setdefault(row["entity_group"],set()).add(row["split"])
        self.assertTrue(all(len(s)==1 for s in groups.values()))

    def test_miniature_labels_evidence_reducer_and_prompts(self):
        self.assertEqual(len(self.schema.slots),79)
        self.assertEqual(v6.validate_collection(self.cases,self.plan)["exact_model_input_duplicates"],0)
        for case in self.cases:
            self.assertNotEqual(case["reviewer_status"],"human_accepted")
            examples=list(v6.sft_examples(case))
            self.assertEqual(len(examples),len(case["turns"])+1)
            self.assertEqual(examples[0]["metadata"]["domain_family"],case["domain_family"])

    def test_frozen_entity_and_template_cannot_be_relabelled(self):
        for field,value in (("entity_group","different"),("template_family","heldout_handover"),("split","final")):
            case=deepcopy(next(c for c in self.cases if c["split"]=="train"))
            case[field]=value
            with self.assertRaises(ValueError):
                v6.validate_collection([case],self.plan)

    def test_domain_values_are_coherent_not_generic_facility_values(self):
        for row in self.plan["assignments"]:
            values=v6.source_values(row,self.schema)
            self.assertNotIn("facility",values["target_description"])
            if row["domain"].startswith("crypto"):
                self.assertEqual(values["calendar_type"],"calendar")
                self.assertFalse(values["business_days_only"])
            if row["domain"]=="temperature":
                self.assertTrue(values["allow_negative_values"])
                self.assertLess(values["target_bounds"]["min"],0)
            if row["domain"]=="precipitation":
                self.assertFalse(values["allow_negative_values"])
            for slot,value in values.items():
                v6.base.canonical_check(slot,value,self.schema)

    def test_mutated_label_and_text_are_rejected(self):
        case=deepcopy(next(c for c in self.cases if c["category"]=="medium"))
        case["turns"][0]["gold_extraction"]["updates"][1]["candidate_value"]='"invented value"'
        with self.assertRaises(ValueError):
            v6.validate_scenario(case)
        case=deepcopy(self.cases[0]); case["turns"][0]["message"]+=" Fabricated future price."
        with self.assertRaises(ValueError):
            v6.validate_scenario(case)

    def test_deterministic_compression_raw_hash_and_duplicate_rejection(self):
        rows=[{"hello":"world"}]
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/"a.gz",Path(d)/"b.gz"
            record=v6.write_jsonl_gz(a,rows); v6.write_jsonl_gz(b,rows)
            self.assertEqual(a.read_bytes(),b.read_bytes())
            self.assertEqual(record["rows"],1)
            self.assertEqual(v6.read_jsonl_gz(a),rows)
        with self.assertRaisesRegex(ValueError,"duplicate scenario"):
            v6.validate_collection([self.cases[0],self.cases[0]],self.plan)

    def test_pending_or_wrong_corpus_review_is_not_approval(self):
        path=Path(v6.PROJECT_ROOT)/"scripts/prepare-v6-training.py"
        spec=importlib.util.spec_from_file_location("v6_prepare_test",path)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            approval=Path(d)/"approval.json"
            approval.write_text(json.dumps({"corpus_manifest_sha256":"wrong"}))
            with self.assertRaises(ValueError):
                mod.approval_check(approval,"frozen")
            approved={"corpus_manifest_sha256":"frozen","reviewer":"fixture reviewer",
                "reviewed_at_utc":"2026-10-02T00:00:00Z","review_ledger_sha256":"a"*64,
                "training_labels_approved":True,"validation_labels_approved":True,"overlap_review_complete":True,
                "final_labels_used_for_training_or_selection":False}
            approval.write_text(json.dumps(approved))
            self.assertEqual(mod.approval_check(approval,"frozen"),approved)
            approved["overlap_review_complete"]=False
            approval.write_text(json.dumps(approved))
            with self.assertRaises(ValueError):
                mod.approval_check(approval,"frozen")

    def test_artifact_roundtrip_and_tampering(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            public,sealed=Path(d)/"public",Path(d)/"sealed"
            with patch.object(v6,"PROFILES",v6.PROFILES[:1]), patch.dict(v6.PROFILE,{v6.PROFILES[0][0]:v6.PROFILES[0]},clear=True):
                manifest=v6.write_artifacts(public,sealed,per_domain=20,version="v6-fixture",check_dependency=False)
                result=v6.verify_artifacts(public,sealed)
                self.assertFalse(result["final_labels_parsed"])
                self.assertTrue(result["sealed_hashes_checked"])
                self.assertFalse(manifest["trainable"])
                path=public/"sft/train.jsonl.gz"
                payload=path.read_bytes(); path.write_bytes(payload+b"x")
                with self.assertRaisesRegex(ValueError,"hash/size"):
                    v6.verify_artifacts(public)


if __name__=="__main__":
    unittest.main()
