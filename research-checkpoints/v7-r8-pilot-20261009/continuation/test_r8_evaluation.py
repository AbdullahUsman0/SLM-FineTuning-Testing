"""CPU-only oracle and fault tests of the new evaluation/state machinery."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from evaluate_r8 import (ComparisonPolicy,PolicyTransformersProvider,build_plans,load_inputs,one_call,
                         advance,create_initial_state,digest,score_records,state_snapshot)
from evaluate_r8 import Provider
from finish_r8 import matched,bootstrap,numeric_canonical,diagnostic_f1
from local_slm_lab.v7_prompts import build_v7_extractor_instructions

RUN=Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v7-r8-pilot-20261010T002204Z')

class FakeProvider(PolicyTransformersProvider):
    def __init__(self,policy,raw,hit_limit=False):
        self.policy=policy;self.raw=raw;self.hit_limit=hit_limit;self._traces=[];self.last_drained_traces=[]
    def _generate_raw(self,system,user):
        assert system==build_v7_extractor_instructions()
        return self.raw,1,1,self.hit_limit
    async def extract(self,message,state):
        from forecasting_assistant.domain.models import ExtractorResult
        return self._structured('extract',build_v7_extractor_instructions(),'test-only',ExtractorResult)

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema,cls.policy,cls.cases,cls.plans=load_inputs(RUN)
    def invoke(self,case,call,raw,track='component',state=None,hit_limit=False):
        provider=FakeProvider(self.policy,raw,hit_limit)
        return asyncio.run(one_call(provider,case,call,track,state,self.schema,self.policy))
    def test_exact_validation_sft_contexts_and_entity_clusters(self):
        self.assertEqual(sum(map(len,self.plans.values())),1040)
        self.assertEqual(len({c['entity_group'] for c in self.cases}),20)
        self.assertTrue(all(c['split']=='validation' for c in self.cases))
    def test_actual_provider_routes_the_matching_v7_prompt(self):
        c=self.cases[0];call=self.plans[c['scenario_id']][0]
        provider=object.__new__(Provider);provider.schema=self.schema;provider.policy=self.policy
        provider._traces=[];provider.last_drained_traces=[]
        provider._torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
        seen=[]
        def generate(system,user):
            seen.append(system)
            self.assertEqual(json.loads(user)['current_message'],call['context']['message'])
            return json.dumps(c['turns'][0]['gold_extraction']),1,1,False
        provider._generate_raw=generate
        result=asyncio.run(provider.extract(call['context']['message'],call['state']))
        self.assertEqual(seen,[build_v7_extractor_instructions()]);self.assertTrue(result.updates)
    def test_gold_oracle_component_and_rollout_for_all_1040_calls(self):
        for track in ('component','rollout'):
            for case in self.cases:
                state=create_initial_state(self.schema)
                for turn,call in zip(case['turns'],self.plans[case['scenario_id']]):
                    record,state=self.invoke(case,call,json.dumps(turn['gold_extraction']),track,state)
                    self.assertTrue(record['prediction_valid'])
                    self.assertTrue(record['evidence_grounded'])
                    self.assertTrue(record['transition_correct'],(case['scenario_id'],turn['turn_id'],track))
                    self.assertEqual(score_records([record])['nonintent']['fn'],0)
    def test_failed_json_kept_in_denominator_and_prior_preserved(self):
        c=self.cases[0];call=self.plans[c['scenario_id']][0];state=create_initial_state(self.schema)
        before=state_snapshot(state)['slots']
        record,state=self.invoke(c,call,'{bad','rollout',state)
        self.assertFalse(record['prediction_valid']);self.assertEqual(score_records([record])['calls']['extract'],1)
        self.assertGreater(score_records([record])['nonintent']['fn'],0)
        self.assertEqual(state_snapshot(state)['slots'],before);self.assertEqual(len(state.turns),1)
    def test_duplicate_slots_and_generation_limit_rejected(self):
        c=self.cases[0];call=self.plans[c['scenario_id']][0];gold=deepcopy(c['turns'][0]['gold_extraction'])
        gold['updates'].append(deepcopy(gold['updates'][0]))
        r,_=self.invoke(c,call,json.dumps(gold));self.assertFalse(r['prediction_valid'])
        r,_=self.invoke(c,call,json.dumps(c['turns'][0]['gold_extraction']),hit_limit=True)
        self.assertTrue(r['raw_json_valid']);self.assertFalse(r['prediction_valid'])
    def test_unmatched_gold_or_component_context_rejected(self):
        c=self.cases[0];call=self.plans[c['scenario_id']][0]
        r,_=self.invoke(c,call,json.dumps(c['turns'][0]['gold_extraction']))
        b=deepcopy(r);b['expected']['updates'][0]['status']='inferred'
        with self.assertRaises(ValueError):matched([r],[b],'component')
        b=deepcopy(r);b['context']['message']='changed'
        with self.assertRaises(ValueError):matched([r],[b],'component')
    def test_string_literal_2026_never_double_decoded(self):
        c=next(c for c in self.cases if c['category']=='encoding_literal')
        call=next(x for x in self.plans[c['scenario_id']] if x['behavior']=='literal_identifier')
        gold=deepcopy(c['turns'][call['key'][2]-1]['gold_extraction'])
        target=next(u for u in gold['updates'] if u['slot_id']=='target_column')
        target['candidate_value']=json.dumps('2026')
        record,_=self.invoke(c,call,json.dumps(gold))
        value=next(u['candidate_value'] for u in record['predicted']['updates'] if u['slot_id']=='target_column')
        self.assertIsInstance(value,str);self.assertEqual(value,'2026')
        state=advance(call['state'].model_copy(deep=True),record,self.policy,self.schema)
        self.assertIsInstance(state.slots['target_column'].value,str)
    def test_ungrounded_wire_is_reported_and_reducer_preserves_prior(self):
        c=self.cases[0];call=self.plans[c['scenario_id']][0];gold=deepcopy(c['turns'][0]['gold_extraction'])
        gold['updates'][0]['evidence_text']='This is absent from the user message.'
        state=create_initial_state(self.schema);before=state_snapshot(state)['slots']
        r,state=self.invoke(c,call,json.dumps(gold),'rollout',state)
        self.assertTrue(r['prediction_valid']);self.assertFalse(r['evidence_grounded']);self.assertFalse(r['transition_correct'])
        self.assertEqual(state_snapshot(state)['slots'],before)
    def test_paired_bootstrap_keeps_failure_penalty_and_groups(self):
        c=self.cases[0];call=self.plans[c['scenario_id']][0]
        bad,_=self.invoke(c,call,'{bad');good,_=self.invoke(c,call,json.dumps(c['turns'][0]['gold_extraction']))
        result=bootstrap(matched([bad],[good],'component'))
        self.assertEqual(result['cluster_count'],1);self.assertEqual(result['delta_right_minus_left'],1)
        self.assertEqual(result['ci95'],[1,1])
    def test_numeric_diagnostics_preserve_string_and_boolean_types(self):
        self.assertEqual(numeric_canonical({'periods':1.0}),{'periods':1})
        self.assertIsInstance(numeric_canonical('2026'),str)
        self.assertIsInstance(numeric_canonical(True),bool)
        c=self.cases[0];call=self.plans[c['scenario_id']][0]
        record,_=self.invoke(c,call,json.dumps(c['turns'][0]['gold_extraction']))
        changed=deepcopy(record)
        for u in changed['emitted']['updates']:
            if isinstance(u['candidate_value'],dict) and 'periods' in u['candidate_value']:
                u['candidate_value']['periods']=float(u['candidate_value']['periods'])
        self.assertEqual(diagnostic_f1([changed],'values_and_status'),1)

if __name__=='__main__':unittest.main()
