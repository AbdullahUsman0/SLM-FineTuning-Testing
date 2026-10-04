"""Durable, isolated, development-only background v6 training."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(sys.argv[1]).resolve()
CONFIG = json.loads((RUN / 'launch-config.json').read_text(encoding='utf-8'))
REPO = Path(CONFIG['repository']).resolve()
PYTHON = CONFIG['python']
os.chdir(REPO)
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location('durable_trainer', REPO / 'training/train_lora.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(name, value):
    temporary = RUN / ('.' + name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, RUN / name)


def guard():
    assert trainer.git_revision(REPO)['revision'] == CONFIG['repository_revision'], 'isolated source revision changed'
    assert trainer.git_revision(REPO.parent / 'fpy')['revision'] == CONFIG['fpy_revision'], 'fpy source revision changed'
    for name, expected in CONFIG['source_sha256'].items():
        assert trainer.sha256(REPO / name) == expected, 'source changed: ' + name
    for name, expected in CONFIG['fpy_source_sha256'].items():
        assert trainer.sha256(REPO.parent / 'fpy' / name) == expected, 'fpy source changed: ' + name
    inputs = json.loads((RUN / 'inputs/input-lock.json').read_text(encoding='utf-8'))
    for name, expected in inputs['files'].items():
        assert trainer.sha256(RUN / 'inputs' / name) == expected['sha256'], 'input changed: ' + name
    for name, expected in CONFIG['run_file_sha256'].items():
        assert trainer.sha256(RUN / name) == expected, 'run configuration changed: ' + name


def locks():
    held = []
    for path in (RUN / 'supervisor.lock', RUN.parent / 'gpu-training.lock'):
        handle = path.open('a+b')
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b'1')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return None
        held.append(handle)
    return held


def main():
    held = locks()
    if held is None or (RUN / 'completion.json').exists():
        return
    guard()
    import psutil
    # An orphaned Trainer can survive a supervisor crash. Wait for it instead
    # of creating a second Trainer that writes into the same directory.
    while True:
        active = []
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if process.info['name'] == 'python.exe' and any(
                    Path(arg).name == 'train_lora.py' for arg in process.info['cmdline'] or []
                ):
                    active.append(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not active:
            break
        save('monitor-status.json', {'status': 'waiting_for_existing_trainer', 'at_utc': now(), 'pids': active})
        time.sleep(15)
    env = {**os.environ, 'HF_HOME': r'D:\SLM\hf-cache', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1',
           'TOKENIZERS_PARALLELISM': 'false'}
    output = RUN / 'full'
    if (output / 'evaluation.json').exists() and (output / 'best-adapter/adapter_model.safetensors').exists():
        print('Trainer already finished; verifying saved artifacts', flush=True)
    else:
        # A Windows restart before checkpoint 10 can leave only the run manifest.
        # Preserve that attempt, then restart fresh from the same stock base/seed.
        if output.exists() and any(output.iterdir()):
            manifest = trainer.load_manifest(output)
            checkpoint = trainer.latest_checkpoint(output, manifest['run_fingerprint'])
            if checkpoint is None:
                assert output.parent.resolve() == RUN and output.name == 'full'
                archived = RUN / ('interrupted-before-first-checkpoint-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
                output.rename(archived)
                print('Preserved interrupted pre-checkpoint attempt: ' + str(archived), flush=True)
        command = [PYTHON, '-u', str(REPO / 'training/train_lora.py'), '--model', CONFIG['model'],
                   '--revision', CONFIG['model_revision'], '--train', str(RUN / 'inputs/train.jsonl'),
                   '--validation', str(RUN / 'inputs/validation.jsonl'), '--output', str(output),
                   '--epochs', '2', '--learning-rate', '5e-5', '--save-steps', '10',
                   '--early-stopping-patience', '0', '--max-length', str(CONFIG['max_length']),
                   '--seed', '42', '--resume-from-checkpoint', 'auto', '--minimum-resume-step', '0']
        verified = {}
        started = now()
        with (RUN / 'full-console.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            save('process.json', {'supervisor_pid': os.getpid(), 'trainer_pid': child.pid,
                 'started_utc': started, 'command': command})
            while True:
                status = {'status': 'training' if child.poll() is None else 'trainer_exited', 'at_utc': now(),
                          'supervisor_pid': os.getpid(), 'trainer_pid': child.pid, 'returncode': child.poll()}
                status['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
                manifest = None
                if (output / trainer.MANIFEST_NAME).exists():
                    try:
                        manifest = trainer.load_manifest(output)
                    except (OSError, ValueError):
                        status['manifest'] = 'write in progress or invalid; inspect trainer log'
                if manifest:
                    for checkpoint in output.glob('checkpoint-*'):
                        if (checkpoint / trainer.COMPLETION_MARKER).exists() and checkpoint.name not in verified:
                            marker = trainer.verify_checkpoint(checkpoint, manifest['run_fingerprint'])
                            verified[checkpoint.name] = {'step': marker['global_step'], 'verified_utc': now()}
                            save('verified-checkpoints.json', verified)
                    if verified:
                        latest = max(verified, key=lambda name: verified[name]['step'])
                        state = trainer.read_json(output / latest / 'trainer_state.json')
                        status.update(latest_verified_checkpoint=latest, step=state['global_step'],
                                      total_steps=state.get('max_steps'), epoch=state.get('epoch'))
                save('monitor-status.json', status)
                if child.poll() is not None:
                    break
                time.sleep(15)
        save('process-result.json', {'started_utc': started, 'ended_utc': now(), 'returncode': child.returncode})
        if child.returncode:
            raise RuntimeError('Trainer failed; inspect full-console.log. Checkpoints are retained.')
    guard()
    manifest = trainer.load_manifest(output)
    steps = []
    for checkpoint in output.glob('checkpoint-*'):
        marker = trainer.verify_checkpoint(checkpoint, manifest['run_fingerprint'])
        steps.append(marker['global_step'])
    last = trainer.latest_checkpoint(output, manifest['run_fingerprint'])
    assert last is not None
    adapter = output / 'best-adapter'
    from safetensors.torch import load_file
    import torch
    tensors = load_file(str(adapter / 'adapter_model.safetensors'))
    reference = load_file(str(last / 'adapter_model.safetensors'))
    assert set(tensors) == set(reference) and all(torch.isfinite(v).all() for v in tensors.values())
    assert all(torch.equal(v, reference[k]) for k, v in tensors.items()), 'final adapter differs from final checkpoint'
    files = {name: {'sha256': trainer.sha256(adapter / name), 'bytes': (adapter / name).stat().st_size}
             for name in ('adapter_model.safetensors', 'adapter_config.json')}
    with (RUN / 'smoke-console.log').open('ab', buffering=0) as log:
        smoke = subprocess.run([PYTHON, '-u', str(RUN / 'adapter_smoke_v6.py'), str(RUN)],
                               cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    result = {'status': 'complete' if smoke.returncode == 0 else 'training_complete_smoke_failed',
              'completed_utc': now(), 'repository_revision': CONFIG['repository_revision'],
              'run_fingerprint': manifest['run_fingerprint'], 'checkpoint_steps': sorted(steps),
              'adapter': files, 'tensor_count': len(tensors), 'matches_final_checkpoint': True,
              'evaluation': trainer.read_json(output / 'evaluation.json'), 'smoke_returncode': smoke.returncode,
              'selection': 'final step, behavioral selection pending', 'independent_human_review': 'pending',
              'matched_behavioral_pilot': 'not run; explicit full experimental training requested',
              'final_labels_accessed': False}
    save('completion.json', result)
    (RUN / 'TRAINING_COMPLETION_SUMMARY.md').write_text('# V6 experimental training completion\n\n'
        'Fresh stock-base specialist adapter; no independent human-review claim.\n'
        'Teacher-forced loss and loading smoke do not establish task quality.\n\n```json\n'
        + json.dumps(result, indent=2) + '\n```\n', encoding='utf-8')
    save('monitor-status.json', result)
    subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                    "Disable-ScheduledTask -TaskName '" + CONFIG['task_name'] + "' | Out-Null"],
                   creationflags=subprocess.CREATE_NO_WINDOW)


if __name__ == '__main__':
    try:
        if '--check-only' in sys.argv:
            guard()
            print('Pinned isolated source and all input/run hashes verified')
        else:
            main()
    except Exception as exc:
        save('monitor-status.json', {'status': 'failed', 'at_utc': now(), 'error': str(exc)})
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
