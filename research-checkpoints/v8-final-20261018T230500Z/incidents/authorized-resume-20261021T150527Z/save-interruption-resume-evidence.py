"""Record a verified same-run recovery, without changing training or its monitor."""
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

control = Path(__file__).resolve().parent
active = Path('D:/SLM/active-v8-final.json')
a = json.loads(active.read_bytes())
repo = Path('D:/SLM/SLM-FineTuning-Testing')
run = Path(a['run'])
incident = Path(a['latest_authorized_resume'])
before = json.loads((incident / 'authorization-and-verification.json').read_bytes())
incident_review = (incident / 'incident-review.md').read_text(encoding='utf-8').strip()
if not incident_review:
    raise RuntimeError('An incident-specific factual review is required')
sys.path.insert(0, str(repo / 'training/final_round'))
import run as worker

worker.check_config(run)
trainer = worker.helper()
manifest = trainer.load_manifest(run / 'full')
spec = importlib.util.spec_from_file_location('monitor', a['hourly_monitor_source'])
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
resumed = trainer.verify_checkpoint(Path(before['checkpoint']), manifest['run_fingerprint'])
latest, verified, rejected = monitor.latest_verified(trainer, run, manifest['run_fingerprint'])
progress = monitor.progress(run)
previous_high_watermark = before['progress_before'].get('last_console_optimizer_step')
live = monitor.processes()
workers = monitor.matching(live, repo / 'training/final_round/run.py', '--run', run, 'worker')
trainers = monitor.matching(live, repo / 'training/train_lora.py', '--output', run / 'full')
if not workers or not trainers:
    raise RuntimeError('Worker and trainer must both be alive')
if progress['resume_checkpoint_step'] != resumed['global_step'] or not progress['new_steps_observed_after_resume']:
    raise RuntimeError('A newly advanced optimizer step after this resume is required')
power = float(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=power.limit', '--format=csv,noheader,nounits'], text=True).strip())
if power != 140:
    raise RuntimeError('Power differs from user-approved previous setup')

dest = repo / a['results_relative'] / 'incidents' / incident.name
dest.mkdir(exist_ok=False)

def write(name, value):
    (dest / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

for path in incident.iterdir():
    if path.is_file() and path.suffix == '.json':
        shutil.copyfile(path, dest / path.name)
    elif path.is_file() and path.suffix in {'.txt', '.md'}:
        # Normalize only public excerpts, preserving the original logs locally.
        content = '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8-sig').splitlines()).rstrip() + '\n'
        (dest / path.name).write_text(content, encoding='utf-8')
write('resume-confirmation.json', {
    'at_UTC': datetime.now(timezone.utc).isoformat(), 'run': str(run),
    'run_fingerprint': manifest['run_fingerprint'], 'progress': progress,
    'previous_observed_high_watermark': previous_high_watermark,
    'previous_high_watermark_recovered': (
        progress['last_console_optimizer_step'] >= previous_high_watermark
        if isinstance(previous_high_watermark, int) else None),
    'resumed_checkpoint': before['checkpoint'], 'resumed_step': resumed['global_step'],
    'resumed_artifacts_sha256_verified': resumed['sha256'],
    'latest_checkpoint': str(latest), 'latest_verified_step': verified['global_step'],
    'latest_artifacts_sha256_verified': verified['sha256'], 'rejected_checkpoints': rejected,
    'worker_pids': workers, 'trainer_pids': trainers, 'GPU_power_limit_W': power,
    'source_and_input_locks_verified': True, 'training_settings_changed': False,
    'step_counter_edited': False, 'automatic_retry_cap_changed': False,
    'checkpoint_directories_retained': len(list((run / 'full').glob('checkpoint-*'))),
    'development_results_available': False, 'final_adapter_available': False,
    'paid_API_calls': 0, 'final_test_labels_accessed': False,
})
write('monitor-after.json', json.loads(Path(a['hourly_check']).read_bytes()))
follow = json.loads((control / 'followup-status.json').read_bytes())
write('followup-after.json', follow)
proofs = {}
for pid in workers + trainers:
    path = control / 'console-isolation-proofs' / f'{pid}.json'
    if path.exists():
        proofs[str(pid)] = json.loads(path.read_bytes())
write('console-isolation-proofs.json', proofs)
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,power.limit,temperature.gpu,utilization.gpu,memory.used', '--format=csv'], text=True)
(dest / 'gpu.csv').write_text(gpu.rstrip() + '\n', encoding='utf-8')
with (run / 'training-console.log').open('rb') as log:
    log.seek(0, 2)
    log.seek(max(0, log.tell() - 65536))
    tail = log.read()
(dest / 'training-console-tail.log.gz').write_bytes(gzip.compress(tail, mtime=0))
shutil.copyfile(__file__, dest / Path(__file__).name)
shutil.copyfile(control / 'resume-authorized-existing-run.py', dest / 'resume-authorized-existing-run.py')
(dest / 'README.md').write_text(f'''# Interrupted process: verified checkpoint-{resumed['global_step']} continuation

{incident_review}

The hourly monitor had used both automatic recovery attempts in its rolling 24-hour window and correctly preserved the run at its existing cap. After reviewing the logs, the user's standing instruction, "continue training with previous setup", was followed through a separately recorded, manually reviewed resume. The retry cap and recovery history were not reset or increased.

All 11 checkpoint artifact hashes, the original run fingerprint, and frozen source/input locks were verified before resuming the SAME run from checkpoint {resumed['global_step']}. The original supplied background worker and existing console-isolated launcher were reused. The power limit remained 140W. Training settings, dataset, model revision, epoch count and optimizer step counter were not changed. Fresh optimizer progress beyond the resumed checkpoint is recorded in resume-confirmation.json. Initial startup output showing zero is not treated as a new run or as verified progress.

The followup companion is waiting for training completion and will then perform the existing local development workflow. The hourly Windows task remains enabled. Current operation does not establish that the underlying interruption or system reliability issue is resolved. No model quality or winner is claimed: final adapter and measured development results remain pending.

Full logs, adapters and optimizer checkpoints remain outside Git. This snapshot includes metadata and compressed log excerpts only, with no base weights, optimizer states, credentials, caches, final-test labels or paid API calls. This auditor does not change the training or monitor source.
''', encoding='utf-8')
write('artifact-manifest.json', {'files_sha256': {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(dest.iterdir()) if p.is_file()
}})
a.update(status='same_run_resume_fresh_progress_verified',
         last_observed_step=progress['last_console_optimizer_step'],
         current_worker_PID=json.loads((run / 'status.json').read_bytes())['pid'],
         current_followup_PID=follow['PID'],
         latest_authorized_resume_publication=str(dest))
active.write_text(json.dumps(a, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'results': str(dest), 'fresh_step': progress['last_console_optimizer_step'],
                  'resumed_step': resumed['global_step'], 'latest_verified_step': verified['global_step']}))
