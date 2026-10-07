"""Contract and token-mask tests without loading model weights."""
import importlib.util
import json
from pathlib import Path
import unittest

from forecasting_assistant.domain.models import ExtractorResult
from local_slm_lab.v5_provider import strict_json_object, validate_raw_output
from local_slm_lab.v6_constrained_provider import NativeGrammarLogitsProcessor, V6ConstrainedProvider, COMPILE_OPTIONS

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('export_schemas',ROOT/'scripts/export-v5-output-schemas.py')
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)


class ConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xgrammar as xgr
        cls.xgr=xgr
        cls.schema=export.build_schemas()['extractor-result']
        info=xgr.TokenizerInfo(['a','b','<eos>'],xgr.VocabType.RAW,stop_token_ids=[2])
        cls.compiler=xgr.GrammarCompiler(info,max_threads=1)
        cls.compiled=cls.compiler.compile_json_schema(cls.schema,**COMPILE_OPTIONS)

    def wire(self):
        return {'intent':'create_forecast','intent_confidence':1,'updates':[{
            'slot_id':'target_unit','candidate_value':json.dumps('litres'),'status':'provided',
            'confidence':1,'evidence_text':'litres'}],'correction_detected':False,'unsupported_claims':[]}

    def accepted(self,text):
        matcher=self.xgr.GrammarMatcher(self.compiled)
        return matcher.accept_string(text) and matcher.is_completed()

    def test_correct_contract_and_explicit_fixed_object_key_order(self):
        wire=self.wire();self.assertTrue(self.accepted(json.dumps(wire)))
        wire=dict(reversed(list(wire.items())))
        wire['updates'][0]=dict(reversed(list(wire['updates'][0].items())))
        self.assertFalse(self.accepted(json.dumps(wire)))

    def test_extra_and_merged_fields_are_rejected_before_emission(self):
        for extra in ({'type':'string'},{'update_type':'string'},{'target_unit':'litres'}):
            wire=self.wire();wire['updates'][0].update(extra)
            self.assertFalse(self.accepted(json.dumps(wire)))

    def test_duplicate_object_keys_are_rejected(self):
        raw=json.dumps(self.wire()).replace('"evidence_text": "litres"','"evidence_text": "litres", "evidence_text": "litres"')
        self.assertFalse(self.accepted(raw))

    def test_missing_keys_and_unknown_slot_names_are_rejected(self):
        wire=self.wire();wire.pop('correction_detected')
        self.assertFalse(self.accepted(json.dumps(wire)))
        wire=self.wire();wire['updates'][0]['slot_id']='invented_slot'
        self.assertFalse(self.accepted(json.dumps(wire)))

    def test_inner_json_and_duplicate_slots_still_need_application_validation(self):
        for duplicate in (False,True):
            wire=self.wire()
            if duplicate:wire['updates'].append(dict(wire['updates'][0]))
            else:wire['updates'][0]['candidate_value']='not json text'
            raw=json.dumps(wire);self.assertTrue(self.accepted(raw))
            with self.assertRaises(ValueError):validate_raw_output(strict_json_object(raw),ExtractorResult)

    def test_mask_prevents_highest_scoring_invalid_token_and_allows_stop(self):
        import torch
        compiled=self.compiler.compile_regex('a')
        processor=NativeGrammarLogitsProcessor(compiled)
        scores=torch.tensor([[1.,100.,0.]])
        result=processor(torch.tensor([[999]]),scores)
        self.assertEqual(result.argmax(-1).item(),0)
        result=processor(torch.tensor([[999,0]]),torch.tensor([[100.,100.,1.]]))
        self.assertEqual(result.argmax(-1).item(),2)

    def test_invalid_constrained_response_is_never_repaired_or_retried(self):
        provider=object.__new__(V6ConstrainedProvider)
        provider.constraints_enabled=True;provider.compiled_extractor=self.compiled
        provider.schema_compile_ms=0;provider._traces=[];calls=[]
        def generate(system,user,logits_processor=None):
            calls.append(logits_processor);return 'not json',10,4,False
        provider._generate_raw=generate
        with self.assertRaises(RuntimeError):provider._structured('extract','system','user',ExtractorResult)
        self.assertEqual(len(calls),1)
        trace=provider.drain_traces()[0];self.assertEqual(trace['retry_count'],0)
        self.assertFalse(trace['raw_schema_valid'])

if __name__=='__main__':unittest.main()
