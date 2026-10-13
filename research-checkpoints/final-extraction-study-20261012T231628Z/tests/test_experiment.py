from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import runtime as rt
import evaluate
from journal import Journal
from shared_policy import SharedPolicy
from registration import validate_human_rows,verify_final
from providers import Hosted
from metrics import paired,bootstrap


def fixture_cases():
    gold={'intent':'create_forecast','intent_confidence':1.0,'correction_detected':False,'unsupported_claims':[],
        'updates':[{'slot_id':'target_column','candidate_value':'"2026"','status':'provided','confidence':1.0,'evidence_text':'2026'}]}
    unknown={**gold,'updates':[]}
    return [{'scenario_id':f'fixture-{i:02d}','source_group':'fixture-group','domain':'offline_fixture',
        'language':'en','template_family':'offline_fixture','turns':[
            {'message':f'Fixture {i}: target column is the literal name 2026.','gold_extraction':gold,'coverage':['literal_strings']},
            {'message':f'Fixture {i}: I do not know the acceptable error.','gold_extraction':unknown,'coverage':['unknown_information']}]} for i in range(20)]


class StudyIntegrity(unittest.TestCase):
    def test_user_deferred_study_blocks_before_freeze_or_API_access(self):
        folder=Path(tempfile.mkdtemp(prefix='study-decision-fixture-'))
        rt.atomic(folder/'EXECUTION_DECISION.json',{'status':'paid_execution_deferred_use_previous_results'})
        with patch.object(rt,'CHECKPOINT',folder),patch.object(rt,'verify_sources') as sources:
            with self.assertRaisesRegex(ValueError,'deferred'):verify_final(folder/'missing-freeze.json')
            with self.assertRaisesRegex(ValueError,'deferred'):Hosted()
            sources.assert_not_called()
            hosted=Hosted.__new__(Hosted)
            with self.assertRaisesRegex(ValueError,'deferred'):hosted.generate('fixture',None,None,None)

    def test_missing_execution_decision_fails_closed(self):
        with patch.object(rt,'CHECKPOINT',Path(tempfile.mkdtemp())):
            with self.assertRaisesRegex(ValueError,'deferred'):rt.require_study_execution()

    def test_pending_human_review_cannot_pass(self):
        row={'scenario_id':'fixture','candidate_sha256':'test-only','decision':'pending','reviewer':'','reviewed_at_utc':''}
        with self.assertRaises(ValueError):validate_human_rows([row],{('fixture',)},'test-only',['scenario_id'])

    def test_review_from_another_cohort_cannot_pass(self):
        row={'scenario_id':'fixture','candidate_sha256':'wrong','decision':'approve','reviewer':'Synthetic unit-test reviewer','reviewed_at_utc':'2026-01-01T00:00:00Z'}
        with self.assertRaises(ValueError):validate_human_rows([row],{('fixture',)},'test-only',['scenario_id'])

    def test_missing_duplicate_review_rows_rejected(self):
        with self.assertRaises(ValueError):validate_human_rows([], {('fixture',)},'test-only',['scenario_id'])
        row={'scenario_id':'fixture'}
        with self.assertRaises(ValueError):validate_human_rows([row,row],{('fixture',)},'test-only',['scenario_id'])

    def test_gate_precedes_model_loading(self):
        with patch.object(evaluate,'verify_final',side_effect=ValueError('review pending')),patch.object(evaluate,'Local') as local:
            with self.assertRaises(ValueError):evaluate.initialize(Path(tempfile.mkdtemp()),Path('test-only-freeze'),{})
            local.assert_not_called()

    def test_distinct_new_runs_cannot_share_raw_receipt_identity(self):
        folder=Path(tempfile.mkdtemp(prefix='study-run-identity-fixtures-'))
        private=folder/'private-fixtures'
        cases=[{**fixture_cases()[0],'scenario_id':f'initialization-fixture-{i}'} for i in range(240)]
        (private/'candidate').mkdir(parents=True)
        (private/'candidate/conversations.jsonl').write_text('\n'.join(rt.dumps(c) for c in cases)+'\n',encoding='utf-8')
        rt.atomic(private/'review/budget-approval.json',{'scope':'Mocked initialization fixture; not user approval','cap_usd':1})
        freeze={'private':str(private),'fingerprint':'mocked-initialization-only'}
        with patch.object(evaluate,'verify_final',return_value=freeze),patch.object(rt,'verify_models'):
            first=evaluate.initialize(folder/'run-a',Path('mocked'),{})[4]
            resumed=evaluate.initialize(folder/'run-a',Path('mocked'),{})[4]
            second=evaluate.initialize(folder/'run-b',Path('mocked'),{})[4]
        self.assertEqual(first,resumed)
        self.assertNotEqual(first['run_id'],second['run_id'])
        source=Journal(folder/'receipt',rt.digest(first));source.save('fixture-call',{'raw_output':'offline fixture'})
        with self.assertRaises(ValueError):Journal(folder/'receipt',rt.digest(second)).load('fixture-call')

    def test_raw_receipt_resume_preserves_literal_and_own_state(self):
        folder=Path(tempfile.mkdtemp(prefix='fresh-study-offline-only-'))
        cases=fixture_cases();schema=rt.load_schema();policy=SharedPolicy();plan=evaluate.plans(cases,schema,policy)
        identity={'planned_calls_per_track':40,'scope':'offline unit-test fixtures; no human approval or final cohort'}
        calls=[]
        class FakeLocal:
            inventory={'scope':'offline mocked generation, not a GPU measurement'}
            def __init__(self,*args):pass
            def generate(self,message,state,schema):
                calls.append(message)
                expected=cases[0]['turns'][1 if 'do not know' in message else 0]['gold_extraction']
                return {'raw_output':rt.dumps(expected),'response_status':'completed','refused':False,
                    'hit_generation_limit':False,'generation_elapsed_seconds':1.25,'cost_usd':0,
                    'prompt_tokens':10,'completion_tokens':10,'reasoning_tokens':0}
        initialize=lambda *args:(cases,plan,schema,policy,identity,{})
        original_grade=evaluate.grade
        with patch.object(evaluate,'initialize',side_effect=initialize),patch.object(evaluate,'Local',FakeLocal),patch.object(rt,'verify_models'):
            with patch.object(evaluate,'grade',side_effect=RuntimeError('simulated interruption after durable raw receipt')):
                with self.assertRaises(RuntimeError):evaluate.run_block(folder,Path('offline-freeze'),{},'v7','fixture-group')
            evaluate.run_block(folder,Path('offline-freeze'),{},'v7','fixture-group')
            generation_count=len(calls)
            evaluate.run_block(folder,Path('offline-freeze'),{},'v7','fixture-group')
            self.assertEqual(len(calls),generation_count)
        self.assertEqual(calls.count(cases[0]['turns'][0]['message']),2) # separate component and rollout calls, not a repeated completed call
        journal=Journal(folder/'records/v7',rt.digest(identity))
        record=journal.load('v7-rollout-fixture-00-02')
        self.assertEqual(record['context']['state']['slots']['target_column']['value'],'2026')
        self.assertEqual(record['predicted_after']['slots']['target_column']['value'],'2026')
        self.assertEqual(record['latency_ms'],1250)
        recovered=journal.load('v7-component-fixture-00-01')
        self.assertTrue(recovered['score_reconstructed_from_journal'])
        self.assertEqual(recovered['latency_ms'],1250) # original generation, never replay execution time
        self.assertEqual(len(list((folder/'records/v7').glob('*.json'))),80)

    def test_changed_context_and_gold_cannot_reuse_journal(self):
        value={'call_id':'one','context_sha256':'context-a','gold_sha256':'gold-a'}
        for context,gold in [('context-b','gold-a'),('context-a','gold-b')]:
            with self.assertRaises(ValueError):evaluate.check_binding(value,'one',context,gold)

    def test_component_context_pairing_guard(self):
        a={'key':['fixture','extract',1],'expected':{'updates':[]},'gold_sha256':'a','cluster_id':'g',
            'track':'component','context':{'message':'same','state':{'slots':{}}}}
        b={**a,'context':{'message':'same','state':{'slots':{'extra':1}}}}
        with self.assertRaises(ValueError):paired([a],[b])
        with self.assertRaises(ValueError):paired([a,a],[a])

    def test_bootstrap_does_not_drop_failed_calls(self):
        gold={'updates':[{'slot_id':'target_column','candidate_value':'x','status':'provided'}]}
        pairs=[]
        for i in range(2):
            a={'cluster_id':str(i),'expected':gold,'emitted':gold,'prediction_valid':True}
            b={**a,'prediction_valid':False}
            pairs.append((a,b))
        result=bootstrap(pairs)
        self.assertEqual(result['delta_left_minus_right'],1.0)
        self.assertEqual(result['ci95'],[1.0,1.0])
        self.assertEqual(result['samples'],10000)


if __name__=='__main__':unittest.main()
