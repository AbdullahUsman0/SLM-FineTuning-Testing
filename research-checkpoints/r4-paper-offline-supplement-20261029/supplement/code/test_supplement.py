"""Supplement controls. Toy review decisions are software fixtures, never human data."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
import socket
import sqlite3
import tempfile
import unittest
import numpy as np
from common import C,OUT,load,prior_states,counts_to_prf,sha
from import_reviews import validate_form,validate_proposal
from coverage_and_intervals import f1_array


class SupplementTests(unittest.TestCase):
    def test_original_primary_and_slot_coverage_reconcile(self):
        result=json.loads((OUT/'analysis/primary_reproduction.json').read_bytes())
        self.assertTrue(result['exactly_matches_publication']);self.assertEqual(result['supported_count'],39);self.assertEqual(result['slot_count'],78)
        with (OUT/'analysis/per_slot_78.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),312)
        with (OUT/'analysis/unknown_slot_ID_predictions.csv').open(encoding='utf-8-sig') as f:unknown=list(csv.DictReader(f))
        for arm in ['v8','openai']:
            for track in ['component','rollout']:
                group=[r for r in rows if r['model']==arm and r['track']==track]
                counts=[sum(int(r[k]) for r in group) for k in ['tp','fp','fn']]
                counts[1]+=sum(int(r['fp']) for r in unknown if r['model']==arm and r['track']==track)
                self.assertEqual(counts_to_prf(counts),result['metrics'][arm][track])

    def test_paired_resampling_recomputes_micro_from_summed_counts(self):
        counts=np.array([[5,1,4],[1,8,0]])
        draw=counts[[0,0,1]].sum(axis=0)
        self.assertAlmostEqual(f1_array(np.array([draw]))[0],22/40)
        self.assertEqual(f1_array(np.array([[0,0,0]]))[0],0)

    def test_posthoc_plan_bound_to_intervals(self):
        result=json.loads((OUT/'analysis/exploratory_paired_intervals.json').read_bytes())
        self.assertTrue(result['posthoc']);self.assertFalse(result['preregistered'])
        self.assertEqual(result['plan_sha256'],sha(OUT/'POSTHOC_ANALYSIS_PLAN.md'))
        self.assertEqual(result['conversation_units'],20);self.assertEqual(result['script_family_units'],10)
        self.assertEqual(result['resamples'],10000)

    def test_visible_input_values_match_reconstructed_actual_priors(self):
        cases,rows,_=load();prior=prior_states(cases,rows)
        for r in rows:
            payload=json.loads(r['current_input']);slots=prior[tuple(r['key'][1:])]['slots']
            for item in payload['confirmed_slots']+payload['unconfirmed_slots']:
                self.assertIn(item['slot_id'],slots)
                expected=slots[item['slot_id']]
                if item['slot_id']!='authentication_reference':self.assertEqual(item['value'],expected['value'])
                self.assertEqual(item['status'],expected['status']);self.assertEqual(item['confirmed_by_user'],expected['confirmed_by_user'])

    def test_blank_forms_never_claim_completed_human_review(self):
        kit=OUT/'human-review'
        for name in ['gold_first_review.csv','second_reviewer_gold.csv','blinded_paired_responses.csv','adjudication.csv']:
            result=validate_form(kit,name,kit/'reviewers'/name,allow_partial=True)
            self.assertEqual(result['completed_rows'],0);self.assertFalse(result['independent_human_review_complete'])
            with self.assertRaises(ValueError):validate_form(kit,name,kit/'reviewers'/name)

    def test_blinding_is_balanced_conversation_stable_and_has_no_metrics(self):
        kit=OUT/'human-review';mapping=json.loads((kit/'coordinator/BLINDING_KEY.json').read_bytes())['mapping']
        self.assertEqual(sum(m['A_model']=='v8' for m in mapping),10)
        for path in (kit/'reviewers').glob('*.csv'):
            self.assertIsNone(re.search(r'\b(?:openai|qwen|gpt-5|slm|v8)\b|latency_seconds|raw_exact_F1|confusion',path.read_text(encoding='utf-8-sig'),re.I))

    def test_oracle_and_unknown_state_findings_preserved(self):
        oracle=json.loads((OUT/'analysis/oracle-20-mismatches.json').read_bytes());self.assertEqual(oracle['mismatch_calls'],20);self.assertTrue(oracle['original_keys_reproduced'])
        self.assertEqual(oracle['recovery_only_mismatch_calls'],20)
        summary=json.loads((OUT/'analysis/error_analysis_summary.json').read_bytes());groups={(r['model'],r['track']):r for r in summary['unknown_extraction_state']}
        self.assertEqual(groups['v8','rollout']['raw_no_actionable_updates'],20);self.assertEqual(groups['v8','rollout']['prior_preserved'],13);self.assertEqual(groups['openai','rollout']['prior_preserved'],15)
        self.assertEqual(len(json.loads((OUT/'analysis/worked-examples.json').read_bytes())),6)
        with (OUT/'analysis/error_taxonomy_counts.csv').open(encoding='utf-8-sig') as f:taxonomy=list(csv.DictReader(f))
        syntax=[r for r in taxonomy if r['category']=='json_valid_failure']
        self.assertEqual(len(syntax),4);self.assertTrue(all(r['unique_affected_calls']=='0' and r['individual_update_or_slot_count']=='' for r in syntax))

    def test_shared_ledger_and_network_are_blocked(self):
        with self.assertRaises(RuntimeError):sqlite3.connect(next(iter(C['original_ledger_hashes'])))
        with socket.socket() as client:
            with self.assertRaises(RuntimeError):client.connect(('203.0.113.1',443))

    def test_proposal_rejects_unknown_slot_bad_evidence_and_nonfinite(self):
        schema={'slot_ids':['target_description'],'statuses':['provided']}
        base={'gold_extraction':{'intent':'create_forecast','intent_confidence':1,'updates':[{'slot_id':'target_description','candidate_value':'"value"','status':'provided','confidence':1,'evidence_text':'value'}],'correction_detected':False,'unsupported_claims':[]}}
        self.assertIsNotNone(validate_proposal(json.dumps(base),schema,'value'))
        for field,value in [('slot_id','invented_slot'),('evidence_text','outside'),('confidence',1.5),('candidate_value','NaN')]:
            obj=deepcopy(base);obj['gold_extraction']['updates'][0][field]=value
            with self.assertRaises(ValueError):validate_proposal(json.dumps(obj),schema,'value')
        obj=deepcopy(base);obj['gold_extraction']['updates']*=2
        with self.assertRaises(ValueError):validate_proposal(json.dumps(obj),schema,'value')
        with self.assertRaises(ValueError):validate_proposal('{"context_state":{},"context_state":{}}',schema,'value')

    def test_toy_import_immutable_rows_duplicates_and_no_auto_application(self):
        # This one-row TEST_ONLY fixture is not any real review form or participant.
        with tempfile.TemporaryDirectory() as temp:
            kit=Path(temp);(kit/'coordinator').mkdir()
            immutable={'row_id':'TEST_ONLY','conversation_id':'TEST_ONLY','turn_id':'test','current_message':'value'}
            mutable=['reviewer_identity','review_decision','proposed_correction_json','rationale']
            reference={'toy.csv':{'kind':'gold','immutable_fields':list(immutable),'mutable_fields':mutable,'rows':{'TEST_ONLY':immutable}}}
            (kit/'coordinator/FORM_REFERENCE_MANIFEST.json').write_text(json.dumps(reference))
            (kit/'coordinator/SCHEMA_VALIDATION.json').write_text(json.dumps({'slot_ids':['target_description'],'statuses':['provided']}))
            row={**immutable,'reviewer_identity':'UNIT_TEST_NOT_HUMAN','review_decision':'approve','proposed_correction_json':'','rationale':'software fixture only'}
            form=kit/'toy.csv'
            def save(items):
                with form.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(items)
            save([row]);result=validate_form(kit,'toy.csv',form)
            self.assertTrue(result['all_rows_completed']);self.assertFalse(result['independent_human_review_complete']);self.assertFalse(result['changes_applied'])
            save([row,row])
            with self.assertRaises(ValueError):validate_form(kit,'toy.csv',form)
            save([{**row,'current_message':'tampered'}])
            with self.assertRaises(ValueError):validate_form(kit,'toy.csv',form)
            save([])
            with self.assertRaises(ValueError):validate_form(kit,'toy.csv',form)


if __name__=='__main__':unittest.main(verbosity=2)
