import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import runtime as rt
from shared_policy import SharedPolicy,admit_for_provider
from providers import decode_hosted,hosted_request
from journal import Journal,Budget


def response(candidate='"field_z"',evidence='field_z'):
    return {'intent':'create_forecast','intent_confidence':1.0,'updates':[
        {'slot_id':'target_column','candidate_value':candidate,'status':'provided','confidence':1.0,'evidence_text':evidence}],
        'correction_detected':False,'unsupported_claims':[]}


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.policy=SharedPolicy();cls.schema=rt.load_schema()

    def parse(self,value,message='field_z',**flags):
        return self.policy.parse(rt.dumps(value) if isinstance(value,dict) else value,message,**flags)

    def test_ordinary_string(self):
        self.assertEqual(self.parse(response('field_z')).predicted['updates'][0]['candidate_value'],'field_z')

    def test_encoded_literals_no_double_decode(self):
        for value in ['2026','true','null','[1,2]','{"a":1}','001']:
            with self.subTest(value=value):
                admission=self.parse(response(rt.dumps(value),value),value)
                self.assertTrue(admission.admitted)
                self.assertEqual(admission.to_model().updates[0].candidate_value,value)
                self.assertIsInstance(admission.to_model().updates[0].candidate_value,str)

    def test_numbers_objects_lists(self):
        for value in [2026,True,None,[1,2],{'a':[{'b':2}]}]:
            admission=self.parse(response(rt.dumps(value)))
            self.assertTrue(admission.admitted)
            self.assertEqual(admission.to_model().updates[0].candidate_value,value)

    def test_duplicate_slots(self):
        value=response();value['updates']*=2
        admission=self.parse(value)
        self.assertEqual(admission.error_code,'duplicate_slot');self.assertFalse(admission.admitted)

    def test_duplicate_outer_and_nested_keys(self):
        raw=rt.dumps(response()).replace('"intent":"create_forecast"','"intent":"create_forecast","intent":"ambiguous"')
        self.assertFalse(self.parse(raw).structural.json_valid)
        self.assertFalse(self.parse(response('{"x":1,"x":2}')).admitted)

    def test_nonfinite_constants_and_overflow(self):
        for literal in ['NaN','Infinity','-Infinity','1e999']:
            self.assertFalse(self.parse(response(literal)).admitted)
            raw=rt.dumps(response()).replace('"intent_confidence":1.0','"intent_confidence":'+literal)
            self.assertFalse(self.parse(raw).admitted)

    def test_closed_required_fields(self):
        for field in response():
            value=response();value.pop(field)
            self.assertFalse(self.parse(value).admitted)
        value=response();value['unexpected']='x';self.assertFalse(self.parse(value).admitted)
        for field in response()['updates'][0]:
            value=response();value['updates'][0].pop(field);self.assertFalse(self.parse(value).admitted)
        value=response();value['updates'][0]['extra']=1;self.assertFalse(self.parse(value).admitted)

    def test_slot_and_status_enums(self):
        value=response();value['updates'][0]['status']='guessed';self.assertFalse(self.parse(value).admitted)
        value=response();value['updates'][0]['slot_id']='field';self.assertFalse(self.parse(value).admitted)

    def test_no_truncation_or_markdown_repair(self):
        raw=rt.dumps(response())
        for text in [raw[:-1],'```json\n'+raw+'\n```','prefix '+raw,raw+raw]:
            self.assertFalse(self.parse(text).admitted)

    def test_exact_evidence(self):
        self.assertTrue(self.parse(response(evidence='Field\n  z'),'Field\n  z').admitted)
        self.assertFalse(self.parse(response(evidence='Field z'),'Field\n  z').admitted)
        self.assertFalse(self.parse(response(evidence='field z'),'Field z').admitted)
        self.assertFalse(self.parse(response(evidence=''),'Field z').admitted)

    def test_refusal_incomplete_and_token_limit(self):
        for flags in [{'refused':True},{'response_status':'incomplete'},{'hit_generation_limit':True}]:
            self.assertFalse(self.parse(response(),**flags).admitted)

    def test_all_transports_share_admission(self):
        raw=rt.dumps(response('"2026"','2026'))
        values=[admit_for_provider(arm,raw,'2026').audit() for arm in ['v6','v7','openai','manual']]
        self.assertTrue(all(value==values[0] for value in values))

    def test_full_schema_not_reduced(self):self.assertEqual(len(self.schema.slots),79)

    def test_unknown_initial_and_established_intent(self):
        state=rt.create_initial_state(self.schema)
        empty={'intent':'ambiguous','intent_confidence':1.0,'updates':[],'correction_detected':False,'unsupported_claims':[]}
        after,accepted,_=self.policy.advance(state,self.parse(empty,'Unknown'),self.schema,1,'Unknown')
        self.assertTrue(accepted);self.assertEqual(rt.semantic_state(after),rt.semantic_state(state))
        seed=self.parse(response())
        state,accepted,_=self.policy.advance(state,seed,self.schema,1,'field_z');self.assertTrue(accepted)
        empty['intent']='create_forecast'
        after,accepted,_=self.policy.advance(state,self.parse(empty,'Unknown'),self.schema,2,'Unknown')
        self.assertTrue(accepted);self.assertEqual(rt.semantic_state(after),rt.semantic_state(state))
        self.assertNotEqual(after.slots['intent'].source_turn,state.slots['intent'].source_turn)

    def test_rejected_evidence_preserves_state(self):
        state=rt.create_initial_state(self.schema)
        after,accepted,_=self.policy.advance(state,self.parse(response(),'different'),self.schema,1,'different')
        self.assertFalse(accepted);self.assertEqual(rt.semantic_state(after),rt.semantic_state(state))

    def test_hosted_refusal_and_reasoning_accounting(self):
        value=decode_hosted({'status':'incomplete','incomplete_details':{'reason':'max_output_tokens'},
            'output':[{'type':'message','content':[{'type':'refusal','refusal':'x'}]}],
            'usage':{'input_tokens':100,'output_tokens':50,'input_tokens_details':{'cached_tokens':20},'output_tokens_details':{'reasoning_tokens':40}}})
        self.assertTrue(value['refused']);self.assertTrue(value['hit_generation_limit']);self.assertEqual(value['reasoning_tokens'],40)
        self.assertAlmostEqual(value['cost_usd'],(80*.25+20*.025+50*2)/1e6)

    def test_hosted_request_same_logical_prompt(self):
        state=rt.create_initial_state(self.schema)
        request=hosted_request('field_z',state,self.schema,self.policy)
        self.assertEqual(request['input'][0]['content'],rt.instructions())
        self.assertEqual(request['input'][1]['content'],rt.context('field_z',state,self.schema))
        self.assertTrue(request['text']['format']['strict'])

    def test_failed_readable_emissions_remain_FP(self):
        value=response();value['updates']*=2
        admission=self.parse(value)
        record={'expected':response(),'emitted':self.policy.emitted(rt.dumps(value),admission),'prediction_valid':False}
        self.assertEqual(rt.confusion(record,nonintent=True),(0,2,1))


class Integrity(unittest.TestCase):
    def setUp(self):
        self.folder=Path(tempfile.mkdtemp(prefix='fresh-study-contract-'))
        # Small test artifacts are deliberately retained; no recursive deletion.

    def test_journal_identity_checksum_and_no_overwrite(self):
        journal=Journal(self.folder/'raw','identity-a');journal.save('one',{'output':'x'})
        self.assertEqual(journal.load('one'),{'output':'x'})
        with self.assertRaises(FileExistsError):journal.save('one',{'output':'y'})
        with self.assertRaises(ValueError):Journal(self.folder/'raw','identity-b').load('one')
        path=self.folder/'raw/one.json';value=rt.read(path);value['payload']['output']='edited';rt.atomic(path,value)
        with self.assertRaises(ValueError):journal.load('one')

    def test_budget_reservation_survives_interruption(self):
        path=self.folder/'budget.json'
        budget=Budget(path,identity='test',cap_usd=1,attempt_bound=.6,max_attempts=3)
        attempt=budget.reserve('one')
        resumed=Budget(path,identity='test',cap_usd=1,attempt_bound=.6,max_attempts=3)
        with self.assertRaises(RuntimeError):resumed.reserve('two')
        resumed.reconcile(attempt,.1);resumed.reserve('two')
        self.assertAlmostEqual(resumed.exposure,.7)

    def test_per_call_retry_limit_and_changed_budget(self):
        path=self.folder/'budget.json';b=Budget(path,identity='test',cap_usd=2,attempt_bound=.1,max_attempts=10)
        b.reserve('one');b.reserve('one')
        with self.assertRaises(RuntimeError):b.reserve('one')
        with self.assertRaises(ValueError):Budget(path,identity='test',cap_usd=3,attempt_bound=.1,max_attempts=10)


if __name__=='__main__':unittest.main()
