#!/usr/bin/env python3
"""Build the portable ten-step v5 Colab notebook from the reviewed base notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/Qwen_2B_LoRA_v5_Resumable_Colab.ipynb"
OUTPUT = ROOT / "notebooks/Qwen_2B_LoRA_v5_Portable_10Step_Colab.ipynb"


def lines(value: str) -> list[str]:
    return value.strip().splitlines(keepends=True)


def code(value: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines(value),
    }


def markdown(value: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines(value)}


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    original = notebook["cells"]

    title = markdown(
        """
# Qwen3.5 2B LoRA — v5 portable training with 10-step checkpoints

This notebook trains the reviewed **general-domain v5 r1 requirements extractor**. It writes a complete, hash-attested checkpoint to Google Drive every **10 optimizer steps** and refuses an accidental restart when `MINIMUM_RESUME_STEP` is positive.

A 10-step interval reduces how much progress can be lost. It does **not** make each checkpoint smaller: exact resume requires the adapter, optimizer, scheduler, trainer state, and RNG state. Expect substantial Drive usage. The transfer cell packages the latest verified checkpoint into one portable `.tar` file for another Google account.

Run the cells in order. For a fresh run leave `MINIMUM_RESUME_STEP = 0`. After transferring `checkpoint-N`, set it to `N`; training stops before loading weights if that checkpoint is missing or corrupt. Never manufacture a checkpoint from a progress-bar number. The sealed final test is never loaded.
""",
        "portable-title",
    )

    config = code(
        """
from pathlib import Path
from google.colab import drive
import hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')

RUN_NAME = 'qwen35-2b-lora-v5-portable-10step-r1'
RUN = Path('/content/drive/MyDrive/FYP-model-runs') / RUN_NAME
TRANSFER_DIR = Path('/content/drive/MyDrive/FYP-model-transfer')
RUN.mkdir(parents=True, exist_ok=True)
TRANSFER_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_INTERVAL = 10
# Fresh run: 0. After moving checkpoint-N to another account: N.
MINIMUM_RESUME_STEP = 0
# Destination account only: set this to the transferred .tar path.
IMPORT_BUNDLE = None
# IMPORT_BUNDLE = '/content/drive/MyDrive/FYP-model-transfer/qwen35-2b-lora-v5-portable-10step-r1-step-70.tar'

PROJECT = Path('/content/slm-v5-portable')
FPY = Path('/content/fpy-v5-portable')
REPO_COMMIT = '7469e8b516474d684f5859f1e9a6a09078af944b'
FPY_COMMIT = '04d52c015d1e3ecdefe92b87116f209361509b4b'
MODEL = 'Qwen/Qwen3.5-2B'
MODEL_REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
EXPECTED_MANIFEST_SHA256 = '7e3c6f83bd39a73a386dbbf5e2a7932c45b68af46eaf6641e56474f2cc9dd021'
MAX_LENGTH = 3584

try:
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True
    ).strip()
except Exception as exc:
    raise RuntimeError('Select a CUDA GPU runtime before continuing.') from exc
print('GPU:', gpu)
print('Run:', RUN)
print('Checkpoint interval:', CHECKPOINT_INTERVAL)
print('Required resume floor:', MINIMUM_RESUME_STEP)
print('Import bundle:', IMPORT_BUNDLE)
""",
        "portable-config",
    )

    import_cell = code(
        """
# Optional safe import on a different Google account. Run before setup/data cells.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

if IMPORT_BUNDLE:
    bundle = Path(IMPORT_BUNDLE)
    assert bundle.is_file(), f'Missing transfer bundle: {bundle}'
    existing = list(RUN.iterdir())
    assert not existing, f'Import destination must be empty; found: {[p.name for p in existing]}'
    with tempfile.TemporaryDirectory(dir='/content') as temporary:
        staging = Path(temporary)
        with tarfile.open(bundle, 'r') as archive:
            for member in archive.getmembers():
                target = (staging / member.name).resolve()
                assert str(target).startswith(str(staging.resolve()) + os.sep), member.name
                assert not (member.issym() or member.islnk()), member.name
            archive.extractall(staging, filter='data')
        transfer_manifest = json.loads((staging / 'transfer-manifest.json').read_text())
        for relative, expected in transfer_manifest['files'].items():
            path = staging / relative
            assert path.is_file(), f'Missing bundled file: {relative}'
            assert sha256(path) == expected['sha256'], f'Hash mismatch: {relative}'
        for item in staging.iterdir():
            if item.name != 'transfer-manifest.json':
                shutil.move(str(item), str(RUN / item.name))
    print('Imported and hash-verified:', bundle)
else:
    print('No transfer bundle selected.')
""",
        "portable-import",
    )

    source_cell = original[2]
    source_cell["id"] = "portable-source"
    source_text = "".join(source_cell["source"])
    source_text = source_text.replace(
        "'license': 'Apache-2.0',", "'license': 'Apache-2.0',\n    'checkpoint_interval_steps': CHECKPOINT_INTERVAL,"
    )
    source_cell["source"] = lines(source_text)
    source_cell["execution_count"] = None
    source_cell["outputs"] = []

    data_cell = original[3]
    data_cell["id"] = "portable-data"
    data_cell["execution_count"] = None
    data_cell["outputs"] = []

    audit_cell = code(
        """
# Mandatory resume audit. This performs full hash verification before training.
sys.path.insert(0, str(PROJECT))
from training.train_lora import checkpoint_step, latest_checkpoint, load_manifest, verify_checkpoint

output = RUN / 'full'
manifest_path = output / 'run-manifest.json'
if manifest_path.is_file():
    manifest = load_manifest(output)
    fingerprint = manifest['run_fingerprint']
    selected = latest_checkpoint(output, fingerprint)
else:
    manifest = None
    fingerprint = None
    selected = None

contents = sorted(p.name for p in output.iterdir()) if output.exists() else []
print('Output contents:', contents)
print('Verified checkpoints:')
if output.exists():
    for path in sorted(output.glob('checkpoint-*'), key=lambda p: checkpoint_step(p)):
        try:
            marker = verify_checkpoint(path, fingerprint)
            print(' ', path.name, marker['completed_at'])
        except Exception as exc:
            print(' INVALID', path.name, type(exc).__name__, str(exc))

if MINIMUM_RESUME_STEP > 0:
    assert selected is not None, f'Expected checkpoint >= {MINIMUM_RESUME_STEP}, but none is verified'
    assert checkpoint_step(selected) >= MINIMUM_RESUME_STEP, (
        f'Latest verified checkpoint is {checkpoint_step(selected)}, expected >= {MINIMUM_RESUME_STEP}'
    )
if contents and selected is None:
    raise RuntimeError('Run folder is occupied but has no verified checkpoint. Use a new RUN_NAME or import the correct bundle.')

print('RESUME DECISION:', f'{selected} (step {checkpoint_step(selected)})' if selected else 'FRESH START AT STEP 0')
""",
        "portable-audit",
    )

    train_cell = original[4]
    train_cell["id"] = "portable-train"
    train_text = "".join(train_cell["source"])
    train_text = train_text.replace("output.mkdir(exist_ok=True)\n", "")
    train_text = train_text.replace("'--save-steps', '25',", "'--save-steps', str(CHECKPOINT_INTERVAL),")
    train_text = train_text.replace(
        "'--resume-from-checkpoint', 'auto',",
        "'--resume-from-checkpoint', 'auto',\n    '--minimum-resume-step', str(MINIMUM_RESUME_STEP),",
    )
    train_text = train_text.replace(
        "=== FULL TRAINING START OR RESUME ===", "=== PORTABLE 10-STEP TRAINING START OR RESUME ==="
    ).replace(
        "All 25-step durability checkpoints", "All 10-step durability checkpoints"
    )
    train_cell["source"] = lines(train_text)
    train_cell["execution_count"] = None
    train_cell["outputs"] = []

    export_cell = code(
        """
# Export the latest verified checkpoint as one cross-account transfer file.
# This can run after an interruption from a CPU-only Colab runtime.
import io
from datetime import datetime, timezone

output = RUN / 'full'
manifest = load_manifest(output)
selected = latest_checkpoint(output, manifest['run_fingerprint'])
assert selected is not None, 'No complete checkpoint is available to export'
verify_checkpoint(selected, manifest['run_fingerprint'])
step = checkpoint_step(selected)
required_resume_files = {
    'adapter_config.json', 'optimizer.pt', 'scheduler.pt',
    'trainer_state.json', 'training_args.bin', 'rng_state.pth',
    'checkpoint-complete.json',
}
missing_resume_files = sorted(name for name in required_resume_files if not (selected / name).is_file())
assert not missing_resume_files, f'Checkpoint cannot resume exactly; missing: {missing_resume_files}'

relative_paths = []
for relative in [
    Path('source-lock.json'), Path('human-review-approval.json'),
    Path('length-preflight.json'), Path('inputs/train.jsonl'),
    Path('inputs/validation.jsonl'), Path('full/run-manifest.json'),
]:
    if (RUN / relative).is_file():
        relative_paths.append(relative)
for path in sorted(selected.rglob('*')):
    if path.is_file():
        relative_paths.append(path.relative_to(RUN))

transfer_manifest = {
    'schema_version': 1,
    'run_name': RUN_NAME,
    'checkpoint_step': step,
    'run_fingerprint': manifest['run_fingerprint'],
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'files': {},
}
for relative in relative_paths:
    path = RUN / relative
    transfer_manifest['files'][relative.as_posix()] = {
        'bytes': path.stat().st_size, 'sha256': sha256(path),
    }

bundle = TRANSFER_DIR / f'{RUN_NAME}-step-{step}.tar'
temporary_bundle = TRANSFER_DIR / f'.{RUN_NAME}-step-{step}-{os.getpid()}.tar.tmp'
with tarfile.open(temporary_bundle, 'w') as archive:
    payload = (json.dumps(transfer_manifest, indent=2) + '\\n').encode()
    info = tarfile.TarInfo('transfer-manifest.json')
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))
    for relative in relative_paths:
        archive.add(RUN / relative, arcname=relative.as_posix(), recursive=False)
temporary_bundle.replace(bundle)

receipt = {
    'bundle': str(bundle), 'bytes': bundle.stat().st_size,
    'sha256': sha256(bundle), 'checkpoint_step': step,
}
bundle.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\\n')
print(json.dumps(receipt, indent=2))
print(f'On the new account set IMPORT_BUNDLE to this file and MINIMUM_RESUME_STEP = {step}.')
""",
        "portable-export",
    )

    summary_cell = original[5]
    summary_cell["id"] = "portable-summary"
    summary_text = "".join(summary_cell["source"])
    summary_text = summary_text.replace("'checkpoint_interval_steps': 25", "'checkpoint_interval_steps': CHECKPOINT_INTERVAL")
    summary_cell["source"] = lines(summary_text)
    summary_cell["execution_count"] = None
    summary_cell["outputs"] = []

    after = markdown(
        """
## After training

Send `training-completion-summary.json` and the last 150 lines of `full-console.log` back to Codex. The next stage selects the behavioral checkpoint on v5 validation and then runs the structured-output experiments one arm at a time. Do not run the sealed final set during method development.
""",
        "portable-after",
    )

    notebook["cells"] = [
        title, config, import_cell, source_cell, data_cell, audit_cell,
        train_cell, export_cell, summary_cell, after,
    ]
    notebook.setdefault("metadata", {}).setdefault("colab", {})["name"] = OUTPUT.name
    notebook["nbformat_minor"] = 5
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
