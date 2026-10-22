"""Snapshot a verified automatic same-run recovery after fresh optimizer progress."""
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

control = Path(__file__).resolve().parent
active = Path('D:/SLM/active-v8-final.json')
a = json.loads(active.read_bytes())
repo = Path('D:/SLM/SLM-FineTuning-Testing')
run = Path(a['run'])
attempts = [json.loads(line) for line in (control / 'hourly-recoveries.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
attempt = next(item for item in reversed(attempts) if item.get('automatic_recovery') is True)
incident = Path(attempt['incident'])
before = json.loads((incident / 'before.json').read_bytes())
review = (incident / 'incident-review.md').read_text(encoding='utf-8').strip()
if not review:
    raise RuntimeError('Incident-specific review required')
sys.path.insert(0, str(repo / 'training/final_round'))
import run as worker

worker.check_config(run)
trainer = worker.helper()
manifest = trainer.load_manifest(run / 'full')
resumed = trainer.verify_checkpoint(Path(attempt['checkpoint']), manifest['run_fingerprint'])
spec = importlib.util.spec_from_file_location('monitor', a['hourly_monitor_source'])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
progress = m.progress(run)
live = m.processes()
workers = m.matching(live, repo / 'training/final_round/run.py', '--run', run, 'worker')
trainers = m.matching(live, repo / 'training/train_lora.py', '--output', run / 'full')
if not workers or not trainers or progress['resume_checkpoint_step'] != resumed['global_step'] or not progress['new_steps_observed_after_resume']:
    raise RuntimeError('Fresh optimizer progress in the live recovered run is required')
latest, verified, rejected = m.latest_verified(trainer, run, manifest['run_fingerprint'])
dest = repo / a['results_relative'] / 'incidents' / incident.name
dest.mkdir(exist_ok=False)

def write(name, value):
    (dest / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

for path in incident.iterdir():
    if path.is_file() and path.suffix == '.json':
        shutil.copyfile(path, dest / path.name)
    elif path.is_file() and path.suffix == '.md':
        (dest / path.name).write_text(path.read_text(encoding='utf-8').rstrip() + '\n', encoding='utf-8')
write('automatic-recovery-attempt.json', attempt)
write('resume-confirmation.json', {
    'at_UTC': datetime.now(timezone.utc).isoformat(), 'run': str(run),
    'run_fingerprint': manifest['run_fingerprint'], 'progress': progress,
    'resumed_step': resumed['global_step'], 'resumed_artifacts_sha256_verified': resumed['sha256'],
    'latest_checkpoint': str(latest), 'latest_verified_step': verified['global_step'],
    'latest_artifacts_sha256_verified': verified['sha256'], 'rejected_checkpoints': rejected,
    'previous_observed_high_watermark': before['progress']['last_console_optimizer_step'],
    'previous_high_watermark_recovered': progress['last_console_optimizer_step'] >= before['progress']['last_console_optimizer_step'],
    'worker_pids': workers, 'trainer_pids': trainers, 'source_and_input_locks_verified': True,
    'training_settings_changed': False, 'counter_edited': False,
    'additional_worker_launched_by_this_auditor': False, 'automatic_retry_cap_changed': False,
    'final_test_labels_accessed': False, 'paid_API_calls': 0,
})
write('monitor-after.json', json.loads(Path(a['hourly_check']).read_bytes()))
follow = json.loads((control / 'followup-status.json').read_bytes())
write('followup-after.json', follow)
with (run / 'training-console.log').open('rb') as stream:
    stream.seek(0, 2)
    stream.seek(max(0, stream.tell() - 65536))
    tail = stream.read()
(dest / 'training-console-tail.log.gz').write_bytes(gzip.compress(tail, mtime=0))
shutil.copyfile(__file__, dest / Path(__file__).name)
(dest / 'README.md').write_text(f'''# Automatic recovery from checkpoint {resumed['global_step']}

{review}

The existing hourly monitor launched the original supplied worker on the SAME run, using automatic selection of its latest complete checkpoint. No duplicate worker or new experiment was launched during the status inspection. All 11 resumed checkpoint file hashes, original run fingerprint and frozen source/input locks were independently verified. Fresh optimizer progress beyond the restored checkpoint is recorded in resume-confirmation.json; this is distinct from recovering the earlier unsaved high-water mark.

The original two-epoch training settings, pinned base and dataset remain unchanged. No step counter or checkpoint was edited. The monitor's two automatic attempts per rolling 24 hours and its history were not reset. Final adapter, remaining validation, development comparison and packaging are pending. The reboot cause remains unresolved; successful checkpoint restoration does not establish that system reliability is fixed.

Full checkpoints and logs remain local. This snapshot contains metadata and a compressed log excerpt, with no base weights, optimizer states, credentials, caches, final-test labels or paid API calls.
''', encoding='utf-8')
write('artifact-manifest.json', {'files_sha256': {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(dest.iterdir()) if path.is_file()
}})
a.update(status='automatic_same_run_recovery_fresh_progress_verified',
         last_observed_step=progress['last_console_optimizer_step'],
         previous_observed_high_watermark=before['progress']['last_console_optimizer_step'],
         verified_checkpoint_step=verified['global_step'], verified_checkpoint=str(latest),
         current_worker_PID=json.loads((run / 'status.json').read_bytes())['pid'],
         current_followup_PID=follow['PID'], latest_automatic_recovery_publication=str(dest))
active.write_text(json.dumps(a, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'record': str(dest), 'fresh_step': progress['last_console_optimizer_step'],
                  'resumed_step': resumed['global_step'], 'verified_step': verified['global_step']}))
