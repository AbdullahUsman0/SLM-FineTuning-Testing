"""Recovery-policy tests use no GPU, model, labels, training launches or network."""
import importlib.util, sys, tempfile, time, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('watchdog',Path(__file__).with_name('hourly-watchdog-r2.py'))
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

class RecoveryPolicy(unittest.TestCase):
    def test_live_worker_never_restarts(self):
        self.assertEqual(w.decision(True,False,False,False,False),'active')
        self.assertEqual(w.decision(True,False,False,False,True),'stalled_requires_review')
    def test_orphan_trainer_prevents_restart(self):
        self.assertEqual(w.decision(False,True,False,False,True),'stalled_requires_review')
    def test_dead_worker_recovers_same_phase(self):
        self.assertEqual(w.decision(False,False,False,False,True),'recover_training')
        self.assertEqual(w.decision(False,False,True,False,True),'recover_development')
    def test_completed_worker_does_not_duplicate_v6_diagnostic(self):
        self.assertEqual(w.decision(False,False,True,False,False,True),'development_finished_followup_pending')
    def test_finished_stops(self):
        self.assertEqual(w.decision(False,False,True,True,False),'finished')
    def test_pid_alone_is_not_process_identity(self):
        script=Path('D:/SLM/repo/run.py');run=Path('D:/SLM/run')
        samples=[{'pid':1,'cmdline':['python.exe','unrelated.py']},
                 {'pid':2,'cmdline':['python.exe',str(script),'worker','--run',str(run)]},
                 {'pid':3,'cmdline':['python.exe',str(script),'worker','--run','D:/SLM/other-run']},
                 {'pid':4,'cmdline':['python.exe',str(script),'start','--run',str(run)]}]
        self.assertEqual(w.matching(samples,script,'--run',run,'worker'),[2])
    def test_selects_verified_checkpoint_skips_corrupt_newer(self):
        class Trainer:
            checkpoint_step=staticmethod(lambda p:int(p.name.split('-')[-1]))
            @staticmethod
            def verify_checkpoint(path,fingerprint):
                if path.name!='checkpoint-120': raise ValueError('hash mismatch')
                self.assertEqual(fingerprint,'fixed');return {'global_step':120}
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'full/checkpoint-120').mkdir(parents=True);(run/'full/checkpoint-130').mkdir()
            chosen,receipt,rejected=w.latest_verified(Trainer,run,'fixed')
            self.assertEqual(chosen.name,'checkpoint-120');self.assertEqual(receipt['global_step'],120)
            self.assertEqual(len(rejected),1)
    def test_no_checkpoint_refuses_restart_from_zero(self):
        class Trainer: checkpoint_step=staticmethod(lambda p:0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'restart from zero is forbidden'):
                w.latest_verified(Trainer,Path(tmp),'fixed')
    def test_resume_does_not_report_old_unsaved_steps_as_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);log=run/'training-console.log'
            log.write_text('133/2234 [old]\nResume decision: verified D:/run/full/checkpoint-130 (step 130)\nLoading weights\n')
            state=w.progress(run)
            self.assertIsNone(state['last_console_optimizer_step'])
            self.assertEqual(state['resume_checkpoint_step'],130)
            self.assertFalse(state['new_steps_observed_after_resume'])
            with log.open('a') as f:f.write('131/2234 [new]\n')
            self.assertEqual(w.progress(run)['last_console_optimizer_step'],131)
            self.assertTrue(w.progress(run)['new_steps_observed_after_resume'])
    def test_hidden_launcher_survives_parent_and_quotes_paths(self):
        with tempfile.TemporaryDirectory(prefix='SLM monitor test ') as tmp:
            proof=Path(tmp)/'child completed.txt'
            code="import time;from pathlib import Path;time.sleep(4);Path("+repr(str(proof))+").write_text('ok')"
            started=time.monotonic()
            result=w.launch_hidden({'control':tmp,'repo':tmp},[sys.executable,'-B','-c',code],'fixture')
            self.assertGreater(result['launcher_pid'],0)
            self.assertFalse(proof.exists(),'Launcher waited for child completion instead of detaching')
            self.assertLess(time.monotonic()-started,4)
            for attempt in range(100):
                if proof.exists(): break
                time.sleep(.1)
            self.assertEqual(proof.read_text(),'ok')

if __name__=='__main__': unittest.main(verbosity=2)
