# Colab training: Qwen3.5-0.8B LoRA v3

Use GitHub for code and the versioned corpus. Write checkpoints directly to
Google Drive so a free Colab runtime can disconnect without losing completed
epochs.

Select **Runtime > Change runtime type > T4 GPU**, then run this setup cell:

```python
import subprocess
import sys
from pathlib import Path
from google.colab import drive

PROJECT_DIR = Path("/content/local-slm-lab")
FPY_DIR = Path("/content/fpy")

if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")

def run(command, cwd=None):
    print("Running:", " ".join(map(str, command)))
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)

if not (PROJECT_DIR / ".git").is_dir():
    run(["git", "clone", "https://github.com/AbdullahUsman0/SLM-FineTuning-Testing.git", PROJECT_DIR])
else:
    run(["git", "-C", PROJECT_DIR, "pull", "--ff-only", "origin", "main"])

if not (FPY_DIR / ".git").is_dir():
    run(["git", "clone", "https://github.com/int-abd-5/fpy.git", FPY_DIR])

run(["git", "-C", FPY_DIR, "fetch", "origin", "04d52c015d1e3ecdefe92b87116f209361509b4b"])
run(["git", "-C", FPY_DIR, "checkout", "--detach", "04d52c015d1e3ecdefe92b87116f209361509b4b"])
run([sys.executable, "-m", "pip", "install", "-r", PROJECT_DIR / "training" / "requirements.txt"])
run([sys.executable, "-m", "pip", "install", "-e", FPY_DIR])
run([sys.executable, "scripts/build-corpus-v3.py"], cwd=PROJECT_DIR)
run([sys.executable, "scripts/verify-corpus-v3.py"], cwd=PROJECT_DIR)
run([sys.executable, "-m", "unittest", "tests.test_corpus_v3", "tests.test_training_helpers", "-v"], cwd=PROJECT_DIR)
```

Then run this training cell:

```python
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path("/content/local-slm-lab")
DRIVE_OUTPUT = Path("/content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v3")
DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable,
    "training/train_lora.py",
    "--train", "corpus-v3/sft/train.jsonl",
    "--validation", "corpus-v3/sft/validation.jsonl",
    "--output", str(DRIVE_OUTPUT),
    "--epochs", "3",
    "--learning-rate", "5e-5",
    "--early-stopping-patience", "1",
    "--max-length", "3072",
    "--resume-from-checkpoint", "auto",
]
print("Starting or resuming:", " ".join(command))
subprocess.run(command, cwd=PROJECT_DIR, check=True)
```

Do not initialize v3 from either older adapter. This is a fresh LoRA adapter on
the same `Qwen/Qwen3.5-0.8B` base; v1 and v2 remain unchanged as experimental
baselines. The trainer evaluates and saves each epoch, reloads the lowest
validation-loss checkpoint, and writes it to `best-adapter`.

After training, verify the saved result:

```python
import json
from pathlib import Path

root = Path("/content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v3")
print("Best adapter exists:", (root / "best-adapter" / "adapter_model.safetensors").is_file())
print("Checkpoints:", sorted(path.name for path in root.glob("checkpoint-*")))
print(json.dumps(json.loads((root / "evaluation.json").read_text()), indent=2))
```
