"""Resume the existing verified run under the user's previous-setup instruction."""
import hashlib,importlib.util,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

control=Path(__file__).resolve().parent;active=Path('D:/SLM/active-v8-final.json');a=json.loads(active.read_bytes())
repo=Path('D:/SLM/SLM-FineTuning-Testing');run=Path(a['run'])
sys.path.insert(0,str(repo/'training/final_round'));import run as worker
config=worker.check_config(run);trainer=worker.helper();manifest=trainer.load_manifest(run/'full')
spec=importlib.util.spec_from_file_location('monitor',a['hourly_monitor_source']);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
live=monitor.processes()
if monitor.matching(live,repo/'training/final_round/run.py','--run',run,'worker'):raise RuntimeError('Existing live worker; no duplicate launch')
if monitor.matching(live,repo/'training/train_lora.py','--output',run/'full'):raise RuntimeError('Existing live trainer; no duplicate launch')
checkpoint,receipt,rejected=monitor.latest_verified(trainer,run,manifest['run_fingerprint'])
with worker.run_lock(run/'worker.lock'):pass
power=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True).strip())
if power!=140:raise RuntimeError('GPU operating limit differs from the previous 140W setup; no silent substitution')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');incident=control/('authorized-resume-'+stamp);incident.mkdir()
before={'at_UTC':datetime.now(timezone.utc).isoformat(),'user_instruction':'continue training with previous setup',
    'run':str(run),'checkpoint':str(checkpoint),'checkpoint_step':receipt['global_step'],
    'verified_sha256':receipt['sha256'],'run_fingerprint':manifest['run_fingerprint'],
    'progress_before':monitor.progress(run),'GPU_limit_W':power,'GPU_settings_changed':False,
    'training_settings_changed':False,'source_input_locks_verified':True,'rejected_checkpoints':rejected,
    'automatic_retry_cap_changed':False}
(incident/'authorization-and-verification.json').write_text(json.dumps(before,indent=2)+'\n',encoding='utf-8')
for name,path in [('worker-before.json',run/'status.json'),('followup-before.json',control/'followup-status.json'),('monitor-before.json',Path(a['hourly_check']))]:
    (incident/name).write_bytes(path.read_bytes())
isolation=json.loads((control/'wmi-isolation-config.json').read_bytes())
if hashlib.sha256((control/'wmi-launch.py').read_bytes()).hexdigest()!=isolation['wmi_launch_sha256']:raise ValueError('Launcher changed')
s=importlib.util.spec_from_file_location('wmi_launcher',control/'wmi-launch.py');launcher=importlib.util.module_from_spec(s);s.loader.exec_module(launcher)
attempt={'at':datetime.now(timezone.utc).isoformat(),'checkpoint':str(checkpoint),'step':receipt['global_step'],
    'incident':str(incident),'reason':'User explicitly requested continuing training with previous setup after hardware review',
    'manual_review':True,'user_authorized':True}
with (control/'hourly-recoveries.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(attempt)+'\n')
started=launcher.launch(isolation,[config['python'],'-B',str(repo/'training/final_round/run.py'),'start','--run',str(run)],'authorized-worker')
(incident/'worker-launch.json').write_text(json.dumps(started,indent=2)+'\n',encoding='utf-8')
if not monitor.matching(monitor.processes(),a['followup_source'],'--config',a['followup_config']):
    with worker.run_lock(control/'followup.lock'):pass
    follow=launcher.launch(isolation,[config['python'],'-B','-u',a['followup_source'],'--config',a['followup_config']],'authorized-followup')
    (incident/'followup-launch.json').write_text(json.dumps(follow,indent=2)+'\n',encoding='utf-8')
a.update(status='user_authorized_previous_setup_resume_launched_progress_pending',
    verified_checkpoint_step=receipt['global_step'],verified_checkpoint=str(checkpoint),
    latest_authorized_resume=str(incident),previous_setup_continuation_authorized=True)
active.write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8')
subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
    "Start-ScheduledTask -TaskName 'SLM-v8-final-hourly-20261018T230500Z'"],check=True)
print(json.dumps({'incident':str(incident),'checkpoint_step_verified':receipt['global_step'],
    'artifacts_verified':len(receipt['sha256']),'same_run':str(run),'GPU_limit_W':power,'resume_launched':True}))
