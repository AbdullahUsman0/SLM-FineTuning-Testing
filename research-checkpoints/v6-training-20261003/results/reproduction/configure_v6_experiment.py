import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

main = Path(r'D:\SLM\SLM-FineTuning-Testing')
repo = Path(r'D:\SLM\SLM-v6-training-20261003')
run = Path(Path(r'D:\SLM\active-v6-training-run.txt').read_text().strip())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


for name in ('supervise_v6_experiment.py', 'adapter_smoke_v6.py', 'prepare_v6_experiment.py', 'resume_v6_training.ps1'):
    shutil.copyfile(Path(r'D:\SLM\setup') / name, run / name)
lengths = json.loads((run / 'token-lengths.json').read_text(encoding='utf-8'))
snapshot = Path(r'D:\SLM\hf-cache\hub\models--Qwen--Qwen3.5-2B\snapshots\15852e8c16360a2fea060d615a32b45270f8a8fc')
weight = snapshot / 'model.safetensors-00001-of-00001.safetensors'
weight_hash = sha(weight)
assert weight_hash == 'aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1'
config = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'repository': str(repo),
    'repository_revision': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'fpy_revision': subprocess.check_output(['git', '-C', r'D:\SLM\fpy', 'rev-parse', 'HEAD'], text=True).strip(),
    'python': str(main / '.venv/Scripts/python.exe'),
    'model': 'Qwen/Qwen3.5-2B', 'model_revision': '15852e8c16360a2fea060d615a32b45270f8a8fc',
    'base_weight_sha256': weight_hash, 'base_weight_bytes': weight.stat().st_size,
    'max_length': lengths['max_length'], 'epochs': 2, 'learning_rate': 5e-5, 'save_steps': 10,
    'task_name': 'FYP-v6-20261003-recovery',
    'source_sha256': {name: sha(repo / name) for name in ('training/train_lora.py',
        'local_slm_lab/v5_provider.py', 'local_slm_lab/peft_provider.py', 'local_slm_lab/v5_prompts.py',
        'local_slm_lab/slm_prompts.py', 'local_slm_lab/providers.py')},
    'fpy_source_sha256': {p.relative_to(repo.parent / 'fpy').as_posix(): sha(p)
                         for p in sorted((repo.parent / 'fpy/src/forecasting_assistant').rglob('*.py'))},
    'run_file_sha256': {name: sha(run / name) for name in ('authorization.json', 'inputs/input-lock.json',
        'token-lengths.json', 'supervise_v6_experiment.py', 'adapter_smoke_v6.py', 'resume_v6_training.ps1')},
    'final_labels_accessed': False, 'adapter_initialization': 'fresh stock base, not v5 adapter'}
(run / 'launch-config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
(run / 'RUN_NOTES.md').write_text(
    '# V6 experimental run\n\n'
    'Explicit user request authorizes training; independent corpus and overlap review remain pending.\n'
    'The 10,219 overlap flags remain unresolved; no semantic leakage or generalization clearance is claimed.\n'
    'Development artifacts were machine-verified and copied byte-for-byte without accessing final labels.\n'
    'A fresh specialist starts from stock Qwen/Qwen3.5-2B, not the previous generalist.\n'
    'Matched behavioral pilot not run; user requested full background experimental fine-tuning.\n'
    'Two epochs, LR5e-5, effective batch16, max_length3584, checkpoint every10, no truncation or early stopping.\n'
    'Final step adapter is not a behaviorally selected winner. Teacher-forced loss is not task-quality evidence.\n'
    'Training is pinned in an isolated detached worktree; checkpoints, weights, environment and cache are local.\n'
    'On reboot, the login recovery task starts the same guarded supervisor and resumes a hash-verified checkpoint.\n',
    encoding='utf-8')
print(json.dumps({'run': str(run), 'source_commit': config['repository_revision'],
                  'max_length': config['max_length'], 'verified_base_weight_sha256': weight_hash}, indent=2))
