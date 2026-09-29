#!/usr/bin/env python3
"""Build the post-training Colab notebook for structured-output research."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/Qwen_2B_v5_Structured_Output_Research_Colab.ipynb"
SOURCE_COMMIT = "774ba274c54e9a0a36a8648160a15e15a15bb002"
FPY_COMMIT = "04d52c015d1e3ecdefe92b87116f209361509b4b"


def lines(value: str) -> list[str]:
    return value.strip().splitlines(keepends=True)


def markdown(value: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines(value)}


def code(value: str, cell_id: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": lines(value)}


def main() -> None:
    cells = [
        markdown("""
# Qwen3.5 2B v5 structured-output research

Run this notebook **after** the final v5 adapter finishes training. It does not retrain or modify the adapter. It prepares and runs the six research arms described in `evaluation/v5-structured-output/README.md`.

Validation is durable per scenario: reconnecting and rerunning an arm skips verified completed scenarios. The primary randomized order is **B, A, C**. Arms D and F are disabled by default because D requires roughly 79 model calls per extraction turn and F incurs API charges. The sealed final set is never opened by this notebook.
""", "research-title"),
        code("""
from pathlib import Path
from google.colab import drive
import hashlib, json, os, shutil, subprocess, sys

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')

# EDIT this one path to the completed portable training run.
TRAIN_RUN = Path('/content/drive/MyDrive/FYP-model-runs/qwen35-2b-lora-v5-portable-10step-r1')
ADAPTER = TRAIN_RUN / 'full/best-adapter'
STUDY_DIR = TRAIN_RUN / 'structured-output-study-v2'
STUDY_DIR.mkdir(parents=True, exist_ok=True)

PROJECT = Path('/content/SLM-FineTuning-Testing')
FPY = Path('/content/fpy')
SOURCE_COMMIT = '774ba274c54e9a0a36a8648160a15e15a15bb002'
FPY_COMMIT = '04d52c015d1e3ecdefe92b87116f209361509b4b'
MODEL = 'Qwen/Qwen3.5-2B'
MODEL_REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
DTYPE = 'float16'
MAX_NEW_TOKENS = 1024

# Expensive optional baselines remain opt-in.
RUN_SLOT_WISE_D = False
RUN_OPENAI_F = False
OPENAI_MODEL = 'gpt-5-mini-2025-08-07'  # explicit historical comparison snapshot

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def command(args, *, log=None, env=None):
    print('Running:', ' '.join(map(str, args)))
    if log is None:
        subprocess.run(list(map(str, args)), check=True, env=env)
        return
    with Path(log).open('a', encoding='utf-8', buffering=1) as handle:
        process = subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in process.stdout:
            print(line, end='')
            handle.write(line)
        if process.wait() != 0:
            raise RuntimeError(f'Command failed; see {log}')

try:
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                   '--format=csv,noheader'], text=True).strip()
except Exception as exc:
    raise RuntimeError('Select a CUDA GPU runtime for local arms A-D.') from exc
print('GPU:', gpu)
print('Training run:', TRAIN_RUN)
print('Adapter:', ADAPTER)
print('Study output:', STUDY_DIR)
""", "research-config"),
        code("""
# Recreate the exact source and dependency revisions.
if PROJECT.exists():
    current = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
    assert current == SOURCE_COMMIT, f'Existing project is at {current}; restart the runtime.'
else:
    command(['git', 'clone', 'https://github.com/AbdullahUsman0/SLM-FineTuning-Testing.git', PROJECT])
    command(['git', '-C', PROJECT, 'checkout', '--detach', SOURCE_COMMIT])

if FPY.exists():
    current = subprocess.check_output(['git', '-C', str(FPY), 'rev-parse', 'HEAD'], text=True).strip()
    assert current == FPY_COMMIT, f'Existing fpy is at {current}; restart the runtime.'
else:
    command(['git', 'clone', 'https://github.com/int-abd-5/fpy.git', FPY])
    command(['git', '-C', FPY, 'checkout', '--detach', FPY_COMMIT])

command([sys.executable, '-m', 'pip', 'install', '-q', '-r',
         PROJECT / 'training/requirements-structured-output.txt'])
command([sys.executable, '-m', 'pip', 'install', '-q', '-e', FPY])

assert subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE_COMMIT
assert subprocess.check_output(['git', '-C', str(FPY), 'rev-parse', 'HEAD'], text=True).strip() == FPY_COMMIT
print('Pinned source and dependencies installed.')
""", "research-source"),
        code("""
# Freeze the selected adapter before any experiment.
required = [ADAPTER / 'adapter_model.safetensors', ADAPTER / 'adapter_config.json']
missing = [str(path) for path in required if not path.is_file()]
assert not missing, f'Completed best-adapter is missing: {missing}'

freeze = {
    'adapter_path': str(ADAPTER),
    'files': {path.name: {'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in required},
    'base_model': MODEL,
    'base_revision': MODEL_REVISION,
    'source_commit': SOURCE_COMMIT,
    'fpy_commit': FPY_COMMIT,
}
freeze_path = STUDY_DIR / 'adapter-freeze.json'
if freeze_path.exists():
    assert json.loads(freeze_path.read_text()) == freeze, 'Adapter changed after the study was initialized'
else:
    freeze_path.write_text(json.dumps(freeze, indent=2) + '\\n')
print(json.dumps(freeze, indent=2))

# Confirm the deterministic frozen cohort and offline implementation tests.
command([sys.executable, '-B', '-m', 'unittest',
         'tests.test_structured_output_providers', 'tests.test_structured_output_runner',
         'tests.test_structured_output_study'], log=STUDY_DIR / 'offline-tests.log')
""", "research-freeze"),
        code("""
# Helpers. Each local arm runs in a fresh process so GPU memory is released cleanly.
SMOKE = PROJECT / 'corpus-v5/v5-20260919-r1/splits/smoke.jsonl'

def local_arguments(arm):
    return ['--arm', arm, '--model', MODEL, '--revision', MODEL_REVISION,
            '--adapter', ADAPTER, '--dtype', DTYPE, '--device', 'cuda',
            '--max-new-tokens', str(MAX_NEW_TOKENS)]

def smoke_arm(arm):
    output = STUDY_DIR / f'arm-{arm.lower()}-smoke.json'
    if output.exists() and json.loads(output.read_text()).get('status') == 'complete':
        print('SKIP complete:', output)
        return
    args = [sys.executable, PROJECT / 'scripts/evaluate-structured-output.py',
            '--arm', arm, '--split', 'smoke', '--cases', SMOKE,
            '--output', output, '--skip-warmup']
    if arm in 'ABCD':
        args += local_arguments(arm)[2:]
    if arm == 'D':
        args += ['--allow-high-call-count']
    command(args, log=STUDY_DIR / f'arm-{arm.lower()}-smoke-console.log')

def validation_arm(arm):
    run_dir = STUDY_DIR / f'arm-{arm.lower()}'
    args = [sys.executable, PROJECT / 'scripts/evaluate-structured-output-resumable.py',
            '--arm', arm, '--run-dir', run_dir]
    if arm in 'ABCD':
        args += local_arguments(arm)[2:]
    if arm == 'D':
        args += ['--allow-high-call-count']
    command(args, log=STUDY_DIR / f'arm-{arm.lower()}-validation-console.log')
""", "research-helpers"),
        code("""
# Diagnostic gate: primary local arms in randomized order B, A, C, then rules E.
for arm in ('B', 'A', 'C', 'E'):
    smoke_arm(arm)
if RUN_SLOT_WISE_D:
    smoke_arm('D')
print('Smoke gate complete. Inspect the four smoke reports before validation.')
""", "research-smoke"),
        code("""
# Arm B validation: exact Arm-A first attempt plus at most one deterministic validation retry.
validation_arm('B')
""", "research-validation-b"),
        code("""
# Arm A validation: current prompt-only JSON + strict parser + Pydantic, no retry.
validation_arm('A')
""", "research-validation-a"),
        code("""
# Arm C validation: XGrammar JSON Schema constrained decoding + local Pydantic.
validation_arm('C')
""", "research-validation-c"),
        code("""
# Arm E validation: deterministic conservative rules, no model calls.
validation_arm('E')
""", "research-validation-e"),
        code("""
# Optional Arm D. This is intentionally disabled because it is about 79 model calls per extraction turn.
if not RUN_SLOT_WISE_D:
    print('Arm D skipped. Set RUN_SLOT_WISE_D=True only after accepting the workload.')
else:
    validation_arm('D')
""", "research-validation-d"),
        code("""
# Optional paid Arm F. The key is read from Colab Secrets and removed immediately afterward.
if not RUN_OPENAI_F:
    print('Arm F skipped. Set RUN_OPENAI_F=True after reviewing the paid validation budget.')
else:
    from google.colab import userdata
    api_key = userdata.get('OPENAI_API_KEY')
    assert api_key, 'Add a fresh OPENAI_API_KEY to Colab Secrets'
    env = os.environ.copy()
    env['OPENAI_API_KEY'] = api_key
    env['V5_OPENAI_COST_APPROVED'] = 'yes'
    try:
        command([
            sys.executable, PROJECT / 'scripts/evaluate-structured-output-resumable.py',
            '--arm', 'F', '--model', OPENAI_MODEL, '--max-new-tokens', str(MAX_NEW_TOKENS),
            '--openai-cost-approved', '--run-dir', STUDY_DIR / 'arm-f',
        ], log=STUDY_DIR / 'arm-f-validation-console.log', env=env)
    finally:
        env.pop('OPENAI_API_KEY', None)
        api_key = None
""", "research-validation-f"),
        code("""
# Paired comparisons. Existing complete comparisons are never overwritten.
comparison_dir = STUDY_DIR / 'comparisons'
comparison_dir.mkdir(exist_ok=True)

def compare(left, right):
    output = comparison_dir / f'{left.lower()}-vs-{right.lower()}.json'
    if output.exists():
        print('SKIP existing:', output)
        return
    left_report = STUDY_DIR / f'arm-{left.lower()}/combined-report.json'
    right_report = STUDY_DIR / f'arm-{right.lower()}/combined-report.json'
    if not (left_report.is_file() and right_report.is_file()):
        print('WAITING:', left, right)
        return
    command([sys.executable, PROJECT / 'scripts/compare-structured-output.py',
             '--left', left_report, '--right', right_report, '--output', output])

for pair in (('A', 'B'), ('A', 'C'), ('A', 'E'), ('A', 'D'), ('A', 'F')):
    compare(*pair)
""", "research-compare"),
        code("""
# Durable inventory for the paper handoff.
summary_path = STUDY_DIR / 'research-completion-summary.json'
artifacts = {}
for path in sorted(STUDY_DIR.rglob('*')):
    if path.is_file() and path != summary_path and path.suffix in {'.json', '.log'}:
        artifacts[str(path.relative_to(STUDY_DIR))] = {
            'bytes': path.stat().st_size,
            'sha256': sha256(path),
        }
summary = {
    'status': 'implementation_ready_measurements_' + (
        'complete' if all((STUDY_DIR / f'arm-{arm.lower()}/combined-report.json').is_file() for arm in 'ABCE')
        else 'pending'
    ),
    'adapter_freeze_sha256': sha256(STUDY_DIR / 'adapter-freeze.json'),
    'required_primary_reports': [f'arm-{arm.lower()}/combined-report.json' for arm in 'ABC'],
    'secondary_reports': [f'arm-{arm.lower()}/combined-report.json' for arm in 'DEF'],
    'artifacts': artifacts,
}
summary_path.write_text(json.dumps(summary, indent=2) + '\\n')
print(json.dumps(summary, indent=2))
""", "research-summary"),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
