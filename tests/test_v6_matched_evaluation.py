import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('matched_runner',ROOT/'scripts/evaluate-v6-matched.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
analysis_spec=importlib.util.spec_from_file_location('matched_analysis',ROOT/'scripts/analyze-v6-matched.py')
analysis=importlib.util.module_from_spec(analysis_spec); analysis_spec.loader.exec_module(analysis)

class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.schema=runner.load_schema()
        self.cases,_=runner.load_cases(ROOT/'evaluation/v6-matched-20261005/natural.jsonl',split='validation')
        self.gold=runner.build_call_plan(self.cases[:1],self.schema)

    def test_rollout_uses_predicted_state_without_replacing_gold(self):
        state=runner.create_initial_state(self.schema)
        call=runner.rollout_call(self.gold[1],state)
        self.assertIsNone(call['context']['state']['slots']['forecast_horizon']['value'])
        self.assertEqual(call['expected'],self.gold[1]['expected'])
        self.assertEqual(call['gold_after'],self.gold[1]['gold_after'])
        self.assertIsNotNone(self.gold[1]['context']['state']['slots']['forecast_horizon']['value'])

    def test_failed_output_preserves_state_and_records_user_turn(self):
        state=runner.create_initial_state(self.schema)
        message='Please do not guess a forecast horizon.'
        result=runner.advance_rollout(state,{'prediction_valid':False,'context':{'message':message}},self.schema,1)
        self.assertIsNone(result.slots['forecast_horizon'].value)
        self.assertEqual(result.turns[-1].user_message,message)

    def test_successful_correction_changes_predicted_state(self):
        state=runner.create_initial_state(self.schema)
        for index,call in enumerate(self.gold[:2],1):
            state=runner.advance_rollout(state,{'prediction_valid':True,'predicted':call['expected'],'context':call['context']},self.schema,index)
        self.assertEqual(state.slots['forecast_horizon'].value,{'periods':8,'unit':'day'})

    def test_no_sealed_or_train_labels_and_all_gold_transitions_preflight(self):
        self.assertEqual(len(self.cases),32)
        self.assertTrue(all(s['split']=='validation' for s in self.cases))
        plan=runner.build_call_plan(self.cases,self.schema)
        self.assertEqual(len(plan),96)
        self.assertEqual(sum(bool(c['expected']['correction_detected']) for c in plan),32)
        self.assertTrue(all(not c['expected']['updates'] for c in plan[2::3]))

    def test_wrong_value_does_not_become_wrong_slot_name(self):
        from copy import deepcopy
        expected=self.gold[0]['expected']; emitted=deepcopy(expected)
        emitted['updates'][1]['candidate_value']='invented unit'
        record={'task':'extract','expected':expected,'emitted':emitted,'prediction_valid':True,
                'context':self.gold[0]['context']}
        metrics=analysis.extra_metrics([record])
        self.assertEqual(metrics['slot_name_prf']['f1'],1)
        self.assertLess(metrics['slot_value_prf_without_status']['f1'],1)
        self.assertEqual(metrics['wrong_values_for_stated_slots'],1)
        self.assertEqual(metrics['new_unmentioned_slot_count'],0)

    def test_failed_unknown_call_is_not_certified_safe(self):
        record={'task':'extract','expected':self.gold[2]['expected'],'emitted':None,'prediction_valid':False,
                'context':self.gold[2]['context'],'test_kind':'unknown','forbidden_slots':self.gold[2]['forbidden_slots']}
        metrics=analysis.extra_metrics([record])
        self.assertEqual(metrics['unknown_no_updates_and_valid']['denominator'],1)
        self.assertEqual(metrics['unknown_no_updates_and_valid']['numerator'],0)

    def test_unscored_warmup_file_uses_validation_labels(self):
        warmup,_=runner.load_cases(ROOT/'corpus-v5/v5-20260919-r1/splits/smoke.jsonl',split='validation')
        self.assertTrue(warmup)
        self.assertEqual(len(runner.build_call_plan(warmup[:1],self.schema)[:1]),1)

    def test_unknown_must_retain_intent_to_retain_state(self):
        from copy import deepcopy
        before=self.gold[2]['context']; after=deepcopy(self.gold[2]['gold_after'])
        after['intent']='ambiguous'
        record={'task':'extract','expected':self.gold[2]['expected'],'emitted':{'updates':[]},
                'prediction_valid':True,'context':before,'test_kind':'unknown',
                'predicted_after':after,'forbidden_slots':self.gold[2]['forbidden_slots']}
        metrics=analysis.extra_metrics([record])
        self.assertEqual(metrics['unknown_no_updates_and_valid']['numerator'],1)
        self.assertEqual(metrics['unknown_retains_prior_state_and_valid']['numerator'],0)

    def test_summary_uses_schema_snapshot_for_known_slot_ids(self):
        # Construct the scorer row from frozen gold; no model is required.
        call=self.gold[0]
        record={'task':'extract','key':call['key'],'scenario_id':call['scenario_id'],
                'expected':call['expected'],'predicted':call['expected'],'emitted':call['expected'],
                'raw_value':call['expected'],'prediction_valid':True,'provider_success':True,
                'context':call['context'],'forbidden_slots':[],'latency_ms':1,
                'raw_json_valid':True,'raw_schema_valid':True}
        metrics=analysis.summarize([record])
        self.assertEqual(metrics['unknown_slot_count'],0)
        record['emitted']={'updates':[dict(call['expected']['updates'][0],slot_id='made_up_id')]}
        record['raw_value']=record['emitted']
        metrics=analysis.summarize([record])
        self.assertEqual(metrics['unknown_slot_count'],1)

    def test_readable_content_does_not_certify_an_invalid_wire_response(self):
        call=self.gold[0]
        record={'task':'extract','expected':call['expected'],'emitted':call['expected'],
                'prediction_valid':False,'raw_json_valid':True,'context':call['context']}
        metrics=analysis.extra_metrics([record])
        self.assertEqual(metrics['slot_value_prf_without_status']['f1'],0)
        self.assertEqual(metrics['readable_slot_value_prf']['f1'],1)

if __name__=='__main__': unittest.main()
