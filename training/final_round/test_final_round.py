"""CPU checks for admission, scoring, journaling, detachment and corpus invariants."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse,importlib.util,json,os,subprocess,sys,time,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_tools import CORPUS,ROOT,sha,rows,verify,use_fpy,dumps,admit,strict_json
from evaluate_development import select_cases,counts,prf,read_journal,decoded_tuples
from run import run_lock,background,train_command
FPY_AVAILABLE=False

class FinalRoundTests(unittest.TestCase):
    def test_frozen_corpus_full_structural_verification(self):
        result=verify()
        self.assertEqual(result['splits']['train']['scenarios'],3600)
        self.assertEqual(result['splits']['validation']['scenarios'],800)
        self.assertEqual(result['splits']['train']['examples'],17860)
        self.assertEqual(result['splits']['validation']['examples'],3980)
        self.assertEqual(result['positive_slot_count'],78)
        self.assertFalse(result['human_review_complete']);self.assertFalse(result['final_test'])

    def test_diagnostic_scope_is_balanced_and_validation_only(self):
        cases=select_cases(list(rows(CORPUS/'splits/validation.jsonl.gz')))
        self.assertEqual(len(cases),100);self.assertTrue(all(c['split']=='validation' for c in cases))
        self.assertEqual(sum(c['origin']=='new_failure_targeted_controlled_synthetic' for c in cases),20)

    def test_unknowns_and_corrections_have_no_unrelated_updates(self):
        focused=[c for c in rows(CORPUS/'splits/train.jsonl.gz') if c['origin']=='new_failure_targeted_controlled_synthetic']
        self.assertEqual(len(focused),800)
        for c in focused:
            correction=c['turns'][2]['gold_extraction']
            self.assertTrue(correction['correction_detected'])
            self.assertEqual([u['slot_id'] for u in correction['updates']],['forecast_horizon'])
            self.assertEqual(c['turns'][3]['gold_extraction']['updates'],[])

    def test_history_has_both_positive_and_abstention_contrasts(self):
        cases=[c for c in rows(CORPUS/'splits/train.jsonl.gz') if c['category']=='cadence_role_isolation']
        self.assertEqual(len(cases),200)
        positives=sum(any(u['slot_id']=='expected_history_length' for u in c['turns'][1]['gold_extraction']['updates']) for c in cases)
        self.assertEqual(positives,100)
        self.assertEqual(sum('expected_history_length' in c['turns'][1]['forbidden_slots'] for c in cases),100)

    def test_safe_references_are_identifiers_only(self):
        cases=list(rows(CORPUS/'splits/train.jsonl.gz'));refs=[t for c in cases for t in c['turns'] if t['behavior']=='safe_stored_reference']
        self.assertEqual(len(refs),100)
        for t in refs:
            self.assertTrue(t['message'].startswith('secret://synthetic-train/'))
            self.assertEqual(json.loads(t['gold_extraction']['updates'][0]['candidate_value']),t['message'])

    def test_rejected_matching_response_gets_zero_true_positives(self):
        gold=[('forecast_horizon','7','provided')]
        self.assertEqual(counts(gold,gold,False),[0,1,1])
        self.assertEqual(prf([[0,1,1]])['f1'],0.)

    def test_duplicate_emissions_are_counted(self):
        gold=[('frequency','daily','provided')]
        self.assertEqual(counts(gold,gold*2,False),[0,2,1])

    def test_wire_string_values_decode_only_once(self):
        obj={'updates':[{'slot_id':'target_column','candidate_value':'"true"','status':'provided'}]}
        self.assertEqual(decoded_tuples(obj,True),[('target_column','"true"','provided')])

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
        for raw in ['{"x":1,"x":2}','NaN','Infinity','1e9999']:
            with self.assertRaises(ValueError):strict_json(raw)

    def admission_fixture(self):
        return {'intent':'create_forecast','intent_confidence':1.,'updates':[{'slot_id':'frequency','candidate_value':'{"periods":1,"unit":"hour"}',
            'status':'provided','confidence':1.,'evidence_text':'Observations are hourly.'}],'correction_detected':False,'unsupported_claims':[]}

    def require_fpy(self):
        if not FPY_AVAILABLE:self.skipTest('Pass --fpy <dedicated pinned checkout> for admission boundary tests')

    def validate_fixture(self,obj,message='Observations are hourly.'):
        self.require_fpy();return admit(dumps(obj),message,json.loads((CORPUS/'response-schema.json').read_bytes()))

    def test_admission_rejects_rewritten_or_prior_only_evidence(self):
        self.require_fpy()
        for evidence in ['Observations  are hourly.','observations are hourly.','Prior message','']:
            obj=self.admission_fixture();obj['updates'][0]['evidence_text']=evidence
            with self.assertRaises(ValueError):self.validate_fixture(obj)

    def test_admission_rejects_duplicate_slots(self):
        self.require_fpy();obj=self.admission_fixture();obj['updates']*=2
        with self.assertRaises(ValueError):self.validate_fixture(obj)

    def test_admission_rejects_extra_fields_and_unknown_slots(self):
        self.require_fpy();obj=self.admission_fixture();obj['extra']='x'
        with self.assertRaises(Exception):self.validate_fixture(obj)
        obj=self.admission_fixture();obj['updates'][0]['slot_id']='invented_field'
        with self.assertRaises(Exception):self.validate_fixture(obj)

    def test_admission_does_not_default_missing_required_field(self):
        self.require_fpy();obj=self.admission_fixture();del obj['correction_detected']
        with self.assertRaises(Exception):self.validate_fixture(obj)

    def test_admission_preserves_literal_string_after_one_decode(self):
        self.require_fpy();obj=self.admission_fixture();obj['updates'][0].update(slot_id='target_column',candidate_value='"true"')
        result=self.validate_fixture(obj)
        self.assertIs(type(result.updates[0].candidate_value),str)
        self.assertEqual(result.updates[0].candidate_value,'true')

    def test_journal_recovers_partial_tail_and_preserves_it(self):
        with TemporaryDirectory() as td:
            p=Path(td)/'calls.jsonl';record={'key':['rollout','a','t1'],'raw_output':'x'}
            p.write_bytes((json.dumps(record)+'\n{"key":').encode())
            self.assertEqual(len(read_journal(p)),1)
            self.assertEqual(p.read_bytes(),(json.dumps(record)+'\n').encode())
            self.assertEqual(len(list(Path(td).glob('*.incomplete-tail-*'))),1)

    def test_journal_adds_newline_after_complete_final_record(self):
        with TemporaryDirectory() as td:
            p=Path(td)/'calls.jsonl';p.write_text('{"key":["component","a","t1"]}',encoding='utf-8')
            self.assertEqual(len(read_journal(p)),1);self.assertTrue(p.read_bytes().endswith(b'\n'))

    def test_duplicate_journal_key_fails_closed(self):
        with TemporaryDirectory() as td:
            p=Path(td)/'calls.jsonl';p.write_text('{"key":["x"]}\n'*2,encoding='utf-8')
            with self.assertRaises(ValueError):read_journal(p)

    def test_live_worker_lock_excludes_second_worker(self):
        with TemporaryDirectory() as td:
            p=Path(td)/'worker.lock'
            with run_lock(p):
                with self.assertRaises(OSError):
                    with run_lock(p):pass
            with run_lock(p):pass

    def test_resume_is_auto_only_after_a_real_manifest(self):
        with TemporaryDirectory() as td:
            run=Path(td);c={'python':sys.executable,'inputs':{'train':{'path':'t'},'validation':{'path':'v'}},'seed':42}
            self.assertNotIn('--resume-from-checkpoint',train_command(run,c))
            (run/'full').mkdir();(run/'full/run-manifest.json').write_text('{}')
            self.assertEqual(train_command(run,c)[-2:],['--resume-from-checkpoint','auto'])

    def test_detached_worker_survives_launcher_exit(self):
        with TemporaryDirectory() as td:
            root=Path(td);stamp=root/'finished.txt';log=root/'worker.log'
            child='import time;from pathlib import Path;time.sleep(1);Path('+repr(str(stamp))+').write_text("completed")'
            launcher='import sys;sys.path.insert(0,'+repr(str(Path(__file__).resolve().parent))+');from run import background;background([sys.executable,"-c",'+repr(child)+'],'+repr(str(log))+')'
            subprocess.run([sys.executable,'-c',launcher],check=True,timeout=15)
            until=time.monotonic()+10
            while not stamp.exists() and time.monotonic()<until:time.sleep(.1)
            self.assertEqual(stamp.read_text(),'completed')

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--fpy',type=Path);known,remaining=parser.parse_known_args()
    if known.fpy:use_fpy(known.fpy);FPY_AVAILABLE=True
    unittest.main(argv=[sys.argv[0],*remaining],verbosity=2)
