# Qwen3.5-2B LoRA: reproducible first training run

This is the first 2B training candidate. It uses the **tracked v3 extraction-only corpus**, because the v4 paraphrase cache and exact generated SFT files are not yet available in Git. It is a new run, not a reproduction of the historical 0.8B v3 run. The tracked Git blobs use LF endings, while the v3 corpus manifest records hashes of the same records with CRLF endings. The historical v3 run input hashes match the LF Git blobs.

Use a Colab CUDA GPU and mount Drive. An L4 or larger GPU gives more memory headroom. A Tesla T4 completed this 2B, 3072-token, two-epoch setup on 2026-09-19; runtime and availability may vary. The trainer checks lengths and stops if any example would be truncated. Do not add `--allow-truncation` merely to make a run start.

## 1. Setup and freeze inputs

```python
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from google.colab import drive

PROJECT = Path("/content/local-slm-lab")
FPY = Path("/content/fpy")
if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")

def run(*args, cwd=None):
    print("Running:", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)

if not (PROJECT / ".git").is_dir():
    run("git", "clone", "https://github.com/AbdullahUsman0/SLM-FineTuning-Testing.git", PROJECT)
run("git", "-C", PROJECT, "fetch", "origin", "7cc08f39272038f4c84ef4d41e64f436ce778742")
run("git", "-C", PROJECT, "checkout", "--detach", "7cc08f39272038f4c84ef4d41e64f436ce778742")

if not (FPY / ".git").is_dir():
    run("git", "clone", "https://github.com/int-abd-5/fpy.git", FPY)
run("git", "-C", FPY, "fetch", "origin", "04d52c015d1e3ecdefe92b87116f209361509b4b")
run("git", "-C", FPY, "checkout", "--detach", "04d52c015d1e3ecdefe92b87116f209361509b4b")

# The old verifier expects CRLF hashes recorded by the Windows-built manifest.
# Colab checks out LF Git blobs. Convert only after confirming each blob's CRLF
# form matches the manifest; preserve current files on Drive before replacing.
PIN = "7cc08f39272038f4c84ef4d41e64f436ce778742"
head = subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip()
assert head == PIN, f"Unexpected repository commit: {head}"

def pinned_bytes(relative):
    return subprocess.check_output(["git", "-C", str(PROJECT), "show", f"{PIN}:corpus-v3/{relative}"])

manifest_bytes = pinned_bytes("manifest.json")
manifest = json.loads(manifest_bytes)
expected_files = {"manifest.json": manifest_bytes}
for relative, info in manifest["files"].items():
    git_data = pinned_bytes(relative)
    data = git_data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert hashlib.sha256(data).hexdigest() == info["sha256"], f"Pinned Git artifact mismatch: {relative}"
    expected_files[relative] = data

changed = [(relative, data) for relative, data in expected_files.items()
           if not (PROJECT / "corpus-v3" / relative).is_file()
           or (PROJECT / "corpus-v3" / relative).read_bytes() != data]
if changed:
    backup_root = Path("/content/drive/MyDrive/FYP-model-runs/preflight-corpus-v3-backup") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for relative, data in changed:
        source = PROJECT / "corpus-v3" / relative
        if source.is_file():
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup)
            print(f"Backed up changed {relative} to {backup}")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
        print(f"Restored pinned {relative}")

run(sys.executable, "-m", "pip", "install", "-r", PROJECT / "training/requirements.txt")
run(sys.executable, "-m", "pip", "install", "-e", FPY)
run(sys.executable, "scripts/verify-corpus-v3.py", cwd=PROJECT)

expected = {
    "corpus-v3/sft/train.jsonl": "f43e88f8c24db0701d3e62df2f7e1a04a8cda93e736809fe24ddecaced833674",
    "corpus-v3/sft/validation.jsonl": "93c39338ad1f85a003d69a3728de5ecc2f79d29c715085487a2eed6ede3f9bad",
}
for relative, wanted in expected.items():
    actual = hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest()
    assert actual == wanted, f"Corpus hash changed: {relative}: {actual}"
print("Frozen corpus hashes verified")
```

## 2. Train a fresh 2B adapter

```python
import hashlib
import subprocess
import sys
from pathlib import Path
from google.colab import drive

PROJECT = Path("/content/local-slm-lab")
assert (PROJECT / "training/train_lora.py").is_file(), "Project checkout missing; run setup first"
if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")

for relative, wanted in {
    "corpus-v3/sft/train.jsonl": "f43e88f8c24db0701d3e62df2f7e1a04a8cda93e736809fe24ddecaced833674",
    "corpus-v3/sft/validation.jsonl": "93c39338ad1f85a003d69a3728de5ecc2f79d29c715085487a2eed6ede3f9bad",
}.items():
    path = PROJECT / relative
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == wanted, (
        f"Corpus not verified: {relative}; run the setup or repair cell first"
    )

import torch
assert torch.cuda.is_available(), "CUDA GPU unavailable; select a GPU runtime"
print("Training GPU:", torch.cuda.get_device_name(0))

def run(*args, cwd=None):
    print("Running:", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)

OUTPUT = Path("/content/drive/MyDrive/FYP-model-runs/qwen35-2b-lora-v3-repro-20260918")
OUTPUT.mkdir(parents=True, exist_ok=True)

run(
    sys.executable, "training/train_lora.py",
    "--model", "Qwen/Qwen3.5-2B",
    "--train", "corpus-v3/sft/train.jsonl",
    "--validation", "corpus-v3/sft/validation.jsonl",
    "--output", OUTPUT,
    "--epochs", "2",
    "--learning-rate", "5e-5",
    "--early-stopping-patience", "1",
    "--max-length", "3072",
    "--resume-from-checkpoint", "auto",
    cwd=PROJECT,
)
```

The output path is deliberately separate from every 0.8B run. The trainer writes epoch checkpoints, `run-manifest.json`, `best-adapter/`, and `evaluation.json` to Drive. If the session stops after a complete checkpoint, rerun the training cell to resume. Inspect errors before changing batch size, sequence length, or precision; any such change creates a new experiment configuration.

## 3. Verify and evaluate on validation scenarios

```python
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/content/local-slm-lab")
OUTPUT = Path("/content/drive/MyDrive/FYP-model-runs/qwen35-2b-lora-v3-repro-20260918")

def run(*args, cwd=None):
    print("Running:", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)

adapter = OUTPUT / "best-adapter" / "adapter_model.safetensors"
assert adapter.is_file(), f"Missing adapter: {adapter}"
print("Adapter bytes:", adapter.stat().st_size)
print("Adapter SHA-256:", hashlib.sha256(adapter.read_bytes()).hexdigest())
print(json.dumps(json.loads((OUTPUT / "evaluation.json").read_text()), indent=2))

run(
    sys.executable, "scripts/evaluate-peft.py",
    "--variant", "custom",
    "--adapter", OUTPUT / "best-adapter",
    "--base-model", "Qwen/Qwen3.5-2B",
    "--cases", "corpus-v3/splits/validation.jsonl",
    "--device", "cuda",
    "--dtype", "float16",
    "--output", OUTPUT / "validation-component-report.json",
    cwd=PROJECT,
)
```

Save the validation report and adapter hash with the run. Use validation results for model or prompt choices. A new untouched final test set is needed because ten existing v3 test scenarios were already used for exploratory 0.8B/2B selection.
