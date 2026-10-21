"""Recovery-policy tests use no GPU, model, labels, training launches or network."""
import importlib.util, json, sys, tempfile, time, unittest
from contextlib import nullcontext
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
spec=importlib.util.spec_from_file_location('watchdog',Path(__file__).with_name('hourly-watchdog-r4.py'))
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

class RecoveryPolicy(unittest.TestCase):
    def test_manual_recoveries_do_not_consume_automatic_slots(self):
        stamp=datetime.now(timezone.utc).isoformat();history=[{'at':stamp,'manual_review':True} for _ in range(5)]
        self.assertEqual(w.recent_automatic_recoveries(history,time.time()),[])
        automatic={'at':stamp,'automatic_recovery':True};history.append(automatic)
        self.assertEqual(w.recent_automatic_recoveries(history,time.time()),[automatic])
    def test_unlabelled_and_conflicting_entries_remain_conservatively_counted(self):
        stamp=datetime.now(timezone.utc).isoformat()
        history=[{'at':stamp},{'at':stamp,'manual_review':True,'automatic_recovery':True}]
        self.assertEqual(w.recent_automatic_recoveries(history,time.time()),history)
    def test_rolling_24_hour_boundary_and_clock_rollback(self):
        current=datetime(2026,10,21,7,20,tzinfo=timezone.utc)
        old={'at':(current-timedelta(hours=24)).isoformat()}
        recent={'at':(current-timedelta(hours=23,minutes=59)).isoformat()}
        future={'at':(current+timedelta(hours=1)).isoformat()}
        self.assertEqual(w.recent_automatic_recoveries([old,recent,future],current.timestamp()),[recent,future])
    def test_full_check_resumes_after_manual_attempts_but_blocks_after_two_automatic(self):
        for automatic_count in (0,2):
            with self.subTest(automatic_count=automatic_count),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);repo=root/'repo';run=root/'same-run';control=root/'control'
                run.mkdir();control.mkdir();stamp=datetime.now(timezone.utc).isoformat()
                history=[{'at':stamp,'manual_review':True} for _ in range(3)]
                history += [{'at':stamp,'automatic_recovery':True} for _ in range(automatic_count)]
                journal=control/'hourly-recoveries.jsonl';original=''.join(json.dumps(x)+'\n' for x in history)
                journal.write_text(original,encoding='utf-8')
                (run/'status.json').write_text(json.dumps({'status':'running','pid':99}),encoding='utf-8')
                (control/'followup-status.json').write_text(json.dumps({'status':'waiting_for_supplied_worker'}),encoding='utf-8')
                follow_source=control/'followup.py';follow_config=control/'followup-config.json'
                companion={'pid':2,'cmdline':['python',str(follow_source),'--config',str(follow_config)]}
                alive_worker={'pid':3,'cmdline':['python',str(repo/'training/final_round/run.py'),'worker','--run',str(run)]}
                trainer=SimpleNamespace(load_manifest=lambda p:{'run_fingerprint':'fixed'})
                helper=SimpleNamespace(check_config=lambda p:{'python':'pinned-python'},helper=lambda:trainer,run_lock=lambda p:nullcontext())
                config={'repo':str(repo),'run':str(run),'control':str(control),'run_fingerprint':'fixed',
                    'watchdog_sha256':w.sha(w.__file__),'followup_source':str(follow_source),'followup_config':str(follow_config),
                    'stall_seconds':3600,'max_recoveries_per_24h':2}
                with patch.dict(sys.modules,{'run':helper}),patch.object(w,'latest_verified',return_value=(run/'full/checkpoint-780',{'global_step':780,'sha256':{'file':'sha'}},[])),patch.object(w,'progress',return_value={'last_output_age_seconds':10}),patch.object(w,'processes',side_effect=[[companion],[companion,alive_worker]]),patch.object(w,'launch_hidden',return_value={'pid':3}) as launch,patch.object(w.time,'sleep'):
                    result=w.check(config)
                self.assertTrue(journal.read_text(encoding='utf-8').startswith(original),'Recovery history must not be reset')
                if automatic_count==0:
                    self.assertEqual(result['status'],'same_run_recovery_launched_progress_not_yet_verified')
                    launch.assert_called_once();command=launch.call_args.args[1]
                    self.assertEqual(command[-1],str(run));self.assertIn('start',command);self.assertNotIn('prepare',command)
                    self.assertEqual(len(w.recent_automatic_recoveries([json.loads(s) for s in journal.read_text().splitlines()],time.time())),1)
                else:
                    self.assertEqual(result['status'],'blocked_preserved');self.assertIn('Recovery limit reached',result['reason'])
                    launch.assert_not_called();self.assertEqual(journal.read_text(encoding='utf-8'),original)
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
            config=json.loads(Path(__file__).with_name('wmi-isolation-config.json').read_bytes())
            result=w.launch_hidden(config,[sys.executable,'-B','-c',code],'fixture')
            self.assertGreater(result['pid'],0)
            self.assertEqual(result['isolation']['console_window'],0)
            self.assertEqual(result['isolation']['provider_name'].lower(),'wmiprvse.exe')
            self.assertFalse(proof.exists(),'Launcher waited for child completion instead of detaching')
            self.assertLess(time.monotonic()-started,4)
            for attempt in range(100):
                if proof.exists(): break
                time.sleep(.1)
            self.assertEqual(proof.read_text(),'ok')

if __name__=='__main__': unittest.main(verbosity=2)
