"""Audit and package completed v6 experiment records; never stage model files."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO = Path(r'D:\SLM\SLM-FineTuning-Testing')
RUN = Path(Path(r'D:\SLM\active-v6-training-run.txt').read_text().strip())
DEST = REPO / 'research-checkpoints/v6-training-20261003/results'
CONFIG = json.loads((RUN / 'launch-config.json').read_text(encoding='utf-8'))
SOURCE = Path(CONFIG['repository'])
spec = importlib.util.spec_from_file_location('completed_v6_trainer', SOURCE / 'training/train_lora.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def git_head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    if DEST.exists():
        raise FileExistsError('Use a new package directory; retain any previous partial package')
    completed = read(RUN / 'completion.json')
    manifest = trainer.load_manifest(RUN / 'full')
    lock = read(RUN / 'inputs/input-lock.json')
    state = read(RUN / 'full/checkpoint-726/trainer_state.json')
    assert completed['status'] == 'complete' and completed['smoke_returncode'] == 0
    assert state['global_step'] == 726 and state['epoch'] == 2.0
    assert completed['run_fingerprint'] == manifest['run_fingerprint']
    assert manifest['immutable']['base_model'] == 'Qwen/Qwen3.5-2B'
    assert manifest['immutable']['model_revision'] == '15852e8c16360a2fea060d615a32b45270f8a8fc'
    assert git_head(SOURCE) == CONFIG['repository_revision']
    assert git_head(SOURCE.parent / 'fpy') == CONFIG['fpy_revision']
    assert lock['corpus_manifest_sha256'] == '0782956c587b5996a2df3f7b799fb0b9e572942b063c8fc1b623ea691fc5aeb9'
    for name, expected in CONFIG['source_sha256'].items():
        assert trainer.sha256(SOURCE / name) == expected, 'source differs: ' + name
    for name, expected in CONFIG['fpy_source_sha256'].items():
        assert trainer.sha256(SOURCE.parent / 'fpy' / name) == expected, 'fpy differs: ' + name
    for name, expected in CONFIG['run_file_sha256'].items():
        assert trainer.sha256(RUN / name) == expected, 'run file differs: ' + name
    for name, info in lock['files'].items():
        assert trainer.sha256(RUN / 'inputs' / name) == info['sha256'], 'input differs: ' + name
    attestations = {}
    expected_steps = list(range(10, 721, 10)) + [726]
    for index, step in enumerate(expected_steps, 1):
        checkpoint = RUN / 'full' / f'checkpoint-{step}'
        attestations[checkpoint.name] = trainer.verify_checkpoint(checkpoint, manifest['run_fingerprint'])
        if index % 10 == 0 or index == len(expected_steps):
            print(f'Independently verified {index}/{len(expected_steps)} checkpoints', flush=True)
    assert sorted(trainer.checkpoint_step(p) for p in (RUN / 'full').glob('checkpoint-*')) == expected_steps
    assert completed['checkpoint_steps'] == expected_steps
    adapter = RUN / 'full/best-adapter'
    for name, info in completed['adapter'].items():
        assert trainer.sha256(adapter / name) == info['sha256'] and (adapter / name).stat().st_size == info['bytes']
    from safetensors.torch import load_file
    import torch
    tensors = load_file(str(adapter / 'adapter_model.safetensors'))
    reference = load_file(str(RUN / 'full/checkpoint-726/adapter_model.safetensors'))
    assert set(tensors) == set(reference) and len(tensors) == completed['tensor_count']
    assert all(torch.isfinite(value).all() and torch.equal(value, reference[name]) for name, value in tensors.items())
    events = [json.loads(line) for line in (RUN / 'full/events.jsonl').read_text(encoding='utf-8').splitlines()]
    started = next(e['at'] for e in events if e['event'] == 'run_started')
    attempts = [{k: e.get(k) for k in ('at', 'event', 'resume_from_checkpoint')}
                for e in events if e['event'] in ('run_started', 'run_resumed')]
    last_window = [e for e in events if e['event'] == 'metrics' and 'loss' in e.get('metrics', {})][-1]
    epoch_evaluations = [e for e in events if e['event'] == 'metrics' and 'eval_loss' in e.get('metrics', {})]
    elapsed = (datetime.fromisoformat(completed['completed_utc']) - datetime.fromisoformat(started)).total_seconds()
    audit = {'status': 'passed', 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
             'source_commit': CONFIG['repository_revision'], 'fpy_commit': CONFIG['fpy_revision'],
             'run_fingerprint': manifest['run_fingerprint'], 'corpus_manifest_sha256': lock['corpus_manifest_sha256'],
             'optimizer_steps': 726, 'epochs': 2.0, 'checkpoint_count': len(attestations),
             'checkpoint_steps': expected_steps, 'all_checkpoint_file_hashes_verified': True,
             'adapter_file_hashes_verified': True, 'adapter_tensor_count': len(tensors),
             'all_adapter_tensors_finite': True, 'matches_checkpoint_726_exactly': True,
             'source_and_input_hashes_verified': True, 'attempts': attempts,
             'wall_seconds_including_interruptions_and_verification': elapsed,
             'last_logged_training_window': last_window, 'epoch_and_standalone_evaluations': epoch_evaluations,
             'training_loss_accounting_note': 'The Trainer aggregate train_loss follows resume accounting and is not an uninterrupted two-epoch mean. Report the last logged training window separately.',
             'independent_human_review': 'pending', 'overlap_review_complete': False,
             'unresolved_overlap_pairs': 10219, 'behavioral_evaluation': 'pending',
             'final_labels_accessed': False, 'paid_api_calls': False}
    DEST.mkdir(parents=True)
    source_records = {}
    for source, name in (
        ('completion.json', 'completion.json'), ('adapter-smoke.json', 'adapter-smoke.json'),
        ('full/evaluation.json', 'evaluation.json'), ('full/run-manifest.json', 'run-manifest.json'),
        ('full/checkpoint-726/trainer_state.json', 'trainer-state.json'), ('launch-config.json', 'launch-config.json'),
        ('authorization.json', 'authorization.json'), ('inputs/input-lock.json', 'input-lock.json'),
        ('token-lengths.json', 'token-lengths.json'), ('interruption-resume-260.json', 'interruption-resume-260.json'),
        ('process.json', 'last-process.json'), ('process-result.json', 'last-process-result.json'),
        ('verified-checkpoints.json', 'supervisor-verified-checkpoints.json')):
        path = RUN / source
        write(DEST / name, read(path))
        source_records[name] = {'local_source': source, 'source_sha256': trainer.sha256(path), 'source_bytes': path.stat().st_size,
                                'packaging': 'JSON value preserved; UTF-8, LF, indentation normalized'}
    for source, name in (('RUN_NOTES.md', 'run-notes.md'),
                         ('TRAINING_COMPLETION_SUMMARY.md', 'original-completion-summary.md')):
        path = RUN / source
        (DEST / name).write_text(path.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
        source_records[name] = {'local_source': source, 'source_sha256': trainer.sha256(path), 'source_bytes': path.stat().st_size,
                                'packaging': 'UTF-8 text, LF normalized'}
    for source, name in (('full/events.jsonl', 'training-events.jsonl.gz'),
                         ('full-console.log', 'training-console.log.gz'), ('smoke-console.log', 'smoke-console.log.gz')):
        path = RUN / source
        with path.open('rb') as inp, (DEST / name).open('wb') as out:
            with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as compressed:
                shutil.copyfileobj(inp, compressed)
        with gzip.open(DEST / name, 'rb') as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == trainer.sha256(path)
        source_records[name] = {'local_source': source, 'source_sha256': trainer.sha256(path), 'source_bytes': path.stat().st_size,
                                'packaging': 'lossless deterministic gzip, original bytes preserved'}
    reproduction = DEST / 'reproduction'
    reproduction.mkdir()
    for name in ('prepare_v6_experiment.py', 'supervise_v6_experiment.py', 'resume_v6_training.ps1', 'adapter_smoke_v6.py'):
        shutil.copyfile(RUN / name, reproduction / name)
        source_records['reproduction/' + name] = {'local_source': name, 'source_sha256': trainer.sha256(RUN / name),
                                                 'packaging': 'original executed source bytes'}
    for name in ('configure_v6_experiment.py', 'v6_training_status.py'):
        path = Path(r'D:\SLM\setup') / name
        shutil.copyfile(path, reproduction / name)
        source_records['reproduction/' + name] = {'local_source': str(path), 'source_sha256': trainer.sha256(path),
                                                 'packaging': 'local orchestration/status helper source bytes'}
    write(DEST / 'completion-audit.json', audit)
    write(DEST / 'checkpoint-attestations.json', attestations)
    write(DEST / 'source-records.json', source_records)
    with (DEST / 'metrics-history.csv').open('w', encoding='utf-8', newline='') as handle:
        columns = ['at_utc', 'step', 'epoch', 'loss', 'eval_loss', 'train_loss', 'grad_norm', 'learning_rate',
                   'entropy', 'eval_entropy', 'mean_token_accuracy', 'eval_mean_token_accuracy', 'num_tokens',
                   'eval_num_tokens', 'train_runtime', 'eval_runtime']
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for event in events:
            if event['event'] == 'metrics':
                row = {'at_utc': event['at'], 'step': event.get('step'), 'epoch': event.get('epoch'), **event['metrics']}
                writer.writerow({key: row.get(key, '') for key in columns})
    finished_local = datetime.fromisoformat(completed['completed_utc']).astimezone(ZoneInfo('America/Los_Angeles'))
    summary = f'''# Completed v6 specialist training

Training and verification completed **{finished_local.strftime('%Y-%m-%d %H:%M:%S %Z')}**:
**726 optimizer steps, two epochs, 73 retained and independently hash-verified checkpoints**.
The final adapter's 372 tensors are finite and exactly equal checkpoint 726.
Selection is the final step; behavioral checkpoint selection remains pending.

| Measurement | Value |
| --- | ---: |
| Training / validation examples | 5,800 / 1,253 |
| Epoch 1 teacher-forced validation loss | 0.000744799617677927 |
| Epoch 2 / standalone validation loss | 0.00006398998812073842 |
| Last logged training window loss, step 725 | {last_window['metrics']['loss']:.12g} |
| Standalone validation token accuracy | 0.9999930703059635 |
| Adapter weights + config bytes | {sum(v['bytes'] for v in completed['adapter'].values())} |
| Wall elapsed from trainer start through verification | {elapsed / 3600:.3f} hours |

The wall time includes Windows interruption, recovery, validation, and verification.
The original attempt reached step 261. Recovery resumed the attested checkpoint
260 with the same source, input hashes and hyperparameters, replaying one step.
The Trainer's aggregate `train_loss` resets its accumulator on resume and divides
by restored global step; it is not an uninterrupted two-epoch average.

The pinned stock model is Qwen/Qwen3.5-2B at
`15852e8c16360a2fea060d615a32b45270f8a8fc`; this is a fresh specialist adapter.
Training code came from isolated commit `{CONFIG['repository_revision']}` with
fpy `{CONFIG['fpy_revision']}`. The corpus is frozen `v6-20261002-r4`,
30% weather / 60% economics / 10% combined, using the existing 79-slot extraction
contract. See [input lock](input-lock.json), [launch configuration](launch-config.json)
and [immutable trainer manifest](run-manifest.json) for all pins and settings.
No completion labels were truncated; maximum training length was 3,385 tokens
under the 3,584 limit. All model input files remain reproducible from the committed corpus.

## What the loading smoke showed

The adapter loaded on CUDA and returned valid extraction JSON for:

> I want to forecast daily maximum temperature in Celsius for the next 7 days using a CSV file.

It produced only `target_description`, containing the entire phrase from
"daily maximum temperature" through "CSV file". It did not return separate
frequency, unit, forecast-horizon or file-format updates. The exact response is in
[adapter-smoke.json](adapter-smoke.json). A successful loading smoke and low
teacher-forced loss therefore do not establish natural-language extraction quality.
This single observation is consistent with a gap between template-like supervision
and natural unquoted phrasing; that is an inference, not a measured generalization score.

Behavioral validation is **pending**. Next evaluation should compare stock, v5 and
v6 on identical v6 development cases with family/subdomain and entity-cluster
metrics, check v5 retention, and freeze independently authored natural-language
requests. Training-budget-matched controls are needed to isolate domain effects.
No numerical weather/economics prediction score or behavioral F1 is claimed here.

## Study records

- [Independent completion audit](completion-audit.json), [completion](completion.json),
  [teacher-forced evaluation](evaluation.json), [trainer state](trainer-state.json).
- [All metrics as CSV](metrics-history.csv), [full training events](training-events.jsonl.gz),
  [full training console](training-console.log.gz), [smoke console](smoke-console.log.gz).
- [Checkpoint attestations](checkpoint-attestations.json), [input token lengths](token-lengths.json),
  [recovery record](interruption-resume-260.json), [run notes](run-notes.md).
- [User authorization](authorization.json), [local source records](source-records.json),
  [publication hashes](artifact-hashes.json), [execution helper source](reproduction/).

The `.gz` logs are lossless; decompress with Python `gzip.open(path, 'rt', encoding='utf-8')`.
JSON values are preserved with normalized LF text; original local-byte hashes are
in `source-records.json`. `artifact-hashes.json` hashes the published Git files.
The execution helpers retain their original local paths for provenance. Adapt
them to a new environment and a new run directory before replaying; do not reuse
the old immutable launch configuration or overwrite the frozen experiment.

Independent human corpus review and 10,219 fuzzy-overlap flags remain unresolved.
Explicit user authorization permitted this experimental run; no formal human-review
approval or zero semantic leakage claim is fabricated. The sealed final set and
paid APIs were unused. The recovery task is now disabled and the trainer has exited.

Adapter location: `{adapter}`. Weights, checkpoints, optimizer/RNG state,
runtime environments, caches, and secrets stay outside Git. The corpus already
contains the verified development data; it is not duplicated in this publication.
'''
    (DEST / 'README.md').write_text(summary, encoding='utf-8', newline='\n')
    # Public text from the exact Windows execution scripts preserves CRLF bytes;
    # Git treats CR as an end-of-line byte through the accompanying attributes.
    forbidden_extensions = {'.safetensors', '.pt', '.pth', '.bin', '.ckpt'}
    for path in DEST.rglob('*'):
        if path.is_file():
            assert path.suffix not in forbidden_extensions and path.stat().st_size < 10_000_000
            if path.suffix != '.gz':
                content = path.read_text(encoding='utf-8')
                assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{20,}', content), 'credential-like string in publication'
    hashes = {p.relative_to(DEST).as_posix(): {'bytes': p.stat().st_size, 'sha256': trainer.sha256(p)}
              for p in sorted(DEST.rglob('*')) if p.is_file()}
    write(DEST / 'artifact-hashes.json', hashes)
    for name, info in hashes.items():
        assert trainer.sha256(DEST / name) == info['sha256']
    print(json.dumps({'status': 'packaged_and_audited', 'output': str(DEST), 'artifacts': len(hashes) + 1,
                     'checkpoints': len(attestations), 'final_validation_loss': completed['evaluation']['eval_loss']}), flush=True)


if __name__ == '__main__':
    main()
