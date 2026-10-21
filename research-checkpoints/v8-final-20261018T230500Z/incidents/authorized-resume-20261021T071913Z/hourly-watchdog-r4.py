"""One hourly, local-only check of the existing fixed experiment. No new runs."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, re, subprocess, sys, time, threading
from datetime import datetime, timezone
from pathlib import Path
import psutil

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def norm(value): return os.path.normcase(os.path.abspath(str(value)))
def argument(command, name, value):
    return any(item==name and i+1<len(command) and norm(command[i+1])==norm(value)
               for i,item in enumerate(command))
def matching(processes, script, flag=None, value=None, verb=None):
    return [p['pid'] for p in processes
            if any(norm(a)==norm(script) for a in p['cmdline'])
            and (verb is None or verb in p['cmdline'])
            and (flag is None or argument(p['cmdline'],flag,value))]
def processes():
    result=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe'):
                if p.info['cmdline'] is None: raise RuntimeError('Cannot inspect Python process identity')
                result.append(p.info)
        except psutil.NoSuchProcess: pass
    return result
def decision(worker_alive, trainer_alive, complete, workflow_done, stale, worker_complete=False):
    if workflow_done: return 'finished'
    if worker_alive or trainer_alive:
        return 'stalled_requires_review' if stale else 'active'
    if worker_complete: return 'development_finished_followup_pending'
    return 'recover_development' if complete else 'recover_training'
def latest_verified(trainer, run, fingerprint):
    rejected=[]
    for path in sorted((run/'full').glob('checkpoint-*'),key=trainer.checkpoint_step,reverse=True):
        try: return path,trainer.verify_checkpoint(path,fingerprint),rejected
        except (OSError,ValueError) as error: rejected.append({'checkpoint':str(path),'reason':str(error)})
    raise RuntimeError('No complete SHA-verified checkpoint; restart from zero is forbidden')
def recent_automatic_recoveries(history, now_timestamp):
    # Reviewed manual resumes do not spend automatic retry slots. Preserve old
    # unlabelled entries as automatic, and conservatively count future clocks.
    return [item for item in history
            if (item.get('manual_review') is not True or item.get('automatic_recovery') is True)
            and now_timestamp-datetime.fromisoformat(item['at']).timestamp()<86400]
def progress(run):
    files=[run/'training-console.log',run/'full/events.jsonl',run/'development-console.log']
    files += list((run/'development-evaluation').glob('calls.jsonl'))
    files += list((run/'v6-same-development-diagnostic').glob('calls.jsonl'))
    stamps=[p.stat().st_mtime for p in files if p.exists()]
    log=run/'training-console.log'; step=None; resumed_from=None
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-262144)); body=f.read().decode('utf-8',errors='replace')
        resumes=list(re.finditer(r'Resume decision: verified [^\r\n]*\(step (\d+)\)',body))
        if resumes:
            resumed_from=int(resumes[-1].group(1));body=body[resumes[-1].end():]
        values=re.findall(r'(\d+)/2234\s*\[',body)
        if values: step=int(values[-1])
    return {'last_output_age_seconds':round(time.time()-max(stamps),1) if stamps else None,
            'last_console_optimizer_step':step,'resume_checkpoint_step':resumed_from,
            'new_steps_observed_after_resume':step is not None and (resumed_from is None or step>resumed_from)}
def launch_hidden(config, command, label):
    control=Path(config['control']);source=control/'wmi-launch.py'
    if sha(source)!=config['wmi_launch_sha256']:raise ValueError('Isolated launcher changed')
    spec=importlib.util.spec_from_file_location('frozen_wmi_launcher',source)
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    return launcher.launch(config,command,label)
def check(config):
    repo=Path(config['repo']);run=Path(config['run']);control=Path(config['control'])
    if sha(__file__)!=config['watchdog_sha256']: raise ValueError('Monitor source changed')
    sys.path.insert(0,str(repo/'training/final_round'))
    import run as worker
    with worker.run_lock(control/'hourly-watchdog-r4.lock'):
        record={'at':now(),'run':str(run),'monitor_pid':os.getpid(),'actions':[],
                'paid_API_calls':0,'training_settings_changed':False}
        try:
            c=worker.check_config(run)
            trainer=worker.helper(); manifest=trainer.load_manifest(run/'full')
            if manifest['run_fingerprint']!=config['run_fingerprint']: raise ValueError('Run identity mismatch')
            checkpoint,receipt,rejected=latest_verified(trainer,run,config['run_fingerprint'])
            record.update(verified_checkpoint=str(checkpoint),verified_step=receipt['global_step'],
                          checkpoint_artifacts_verified=len(receipt['sha256']),rejected_checkpoints=rejected)
            live=processes(); status=read(run/'status.json'); follow=read(control/'followup-status.json')
            workers=matching(live,repo/'training/final_round/run.py','--run',run,'worker')
            trainers=matching(live,repo/'training/train_lora.py','--output',run/'full')
            diagnostic=matching(live,repo/'training/final_round/evaluate_development.py','--output',run/'development-evaluation')
            v6_diagnostic=matching(live,repo/'training/final_round/evaluate_development.py','--output',run/'v6-same-development-diagnostic')
            companions=matching(live,config['followup_source'],'--config',config['followup_config'])
            observation=progress(run); age=observation['last_output_age_seconds']
            record.update(worker_pids=workers,trainer_pids=trainers,diagnostic_pids=diagnostic,
                          companion_pids=companions,v6_diagnostic_pids=v6_diagnostic,progress=observation,worker_status=status,followup_status=follow)
            # An alive trainer/diagnostic is never killed, including during long evaluation.
            action=decision(bool(workers),bool(trainers or diagnostic),(run/'training-completion.json').exists(),
                            follow.get('status')=='complete_and_pushed',age is not None and age>config['stall_seconds'],
                            status.get('status')=='complete')
            record['status']=action
            if action in ('recover_training','recover_development'):
                history=control/'hourly-recoveries.jsonl'
                old=[json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
                recent=recent_automatic_recoveries(old,time.time())
                record['automatic_recovery_attempts_last_24h']=len(recent)
                record['maximum_automatic_recoveries_per_24h']=config['max_recoveries_per_24h']
                if len(recent)>=config['max_recoveries_per_24h']: raise RuntimeError('Recovery limit reached; inspect saved failure before another attempt')
                # Never trust a PID-only status. The supplied OS lock protects the full workflow.
                with worker.run_lock(run/'worker.lock'): pass
                if trainers or diagnostic: raise RuntimeError('Live child prevents recovery')
                incident=control/('hourly-recovery-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
                incident.mkdir();write(incident/'before.json',record)
                attempt={'at':now(),'checkpoint':str(checkpoint),'step':receipt['global_step'],'incident':str(incident),'automatic_recovery':True}
                with history.open('a',encoding='utf-8') as f:f.write(json.dumps(attempt)+'\n')
                # The original worker chooses verified auto-resume; no prepare, new path or counter edits.
                command=[c['python'],'-B',str(repo/'training/final_round/run.py'),'start','--run',str(run)]
                record['actions'].append(launch_hidden(config,command,'hourly-worker'))
                time.sleep(10)
                live=processes()
                workers=matching(live,repo/'training/final_round/run.py','--run',run,'worker')
                if not workers: raise RuntimeError('Recovery worker did not stay alive; inspect unique logs')
                record['recovery_worker_pids']=workers
                record['status']='same_run_recovery_launched_progress_not_yet_verified'
            if action!='finished' and not companions and v6_diagnostic:
                record['companion_note']='Orphan v6 diagnostic is alive; no concurrent diagnostic will be launched'
            elif action!='finished' and not companions:
                stopped=follow.get('status')=='stopped_preserved'
                recoverable=not stopped or str(follow.get('reason','')).startswith('Supplied worker failed')
                if not recoverable:
                    record['companion_note']='Preserved scientific/publication failure needs review; no blind retry'
                else:
                    with worker.run_lock(control/'followup.lock'): pass
                    if sha(config['followup_source'])!=read(config['followup_config'])['followup_sha256']:
                        raise ValueError('Followup source changed')
                    command=[c['python'],'-B','-u',config['followup_source'],'--config',config['followup_config']]
                    record['actions'].append(launch_hidden(config,command,'hourly-followup'))
            record['finished_at']=now()
        except Exception as error:
            record.update(status='blocked_preserved',error_type=type(error).__name__,reason=str(error),finished_at=now())
        write(control/'hourly-last-check-r4.json',record)
        with (control/'hourly-checks-r4.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
        return record
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--loop',action='store_true');a=p.parse_args()
    config=read(a.config)
    if a.loop:
        # Separate daemon lock prevents duplicate logon launches in the fallback mode.
        sys.path.insert(0,str(Path(config['repo'])/'training/final_round'));from run import run_lock
        with run_lock(Path(config['control'])/'hourly-daemon.lock'):
            while True:
                record=check(config)
                if record['status']=='finished': return
                time.sleep(config['interval_seconds'])
    else:
        record=check(config);print(json.dumps(record,indent=2))
        if record['status']=='blocked_preserved': sys.exit(1)
if __name__=='__main__':main()
