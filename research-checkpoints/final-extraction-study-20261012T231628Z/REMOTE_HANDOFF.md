# Handoff to another Codex session

Repository: `AbdullahUsman0/SLM-FineTuning-Testing`, branch `experiment/v5-grounded-20260919`. Preparation started from `15d8c8703b6c2312356aecc920aeb8a5ac31afb2`; use `git log -1` after pulling to identify the publication commit. Work lives under `research-checkpoints/final-extraction-study-20261012T231628Z`. Read `EXECUTION_DECISION.json`, `README.md`, `PAPER_RESULTS.md`, `IMPLEMENTATION_CHANGES.md` and `PREPARATION_COMPLETION.json` first.

**Current instruction: use existing evidence; $0 new paid API budget; no training; no new final-cohort runs.** There is no active evaluation process to monitor. Registration and hosted access fail before loading/network calls while the study is deferred. Do not interpret the old proposed budget as approval, or previous training-dataset approval as review of the new evaluation candidate.

With a clean checkout, update:

```powershell
Set-Location D:\SLM\SLM-FineTuning-Testing
git status --short
git pull --ff-only origin experiment/v5-grounded-20260919
git rev-parse HEAD
```

Preserve unrelated work if the checkout is dirty. No reset/discard, sealed-label access, paid API calls or new training.

Completed: old matched evidence replay, pilot failures/metrics replay, verified adapters/base/dependencies, separate CUDA smoke, shared parser and annotation rules, pending private candidate/review, durable runner and offline tests. The new study has no final freeze, scored outcomes or independently completed human review. The machine-readable historical 31-case manifest also records incomplete human review.

## Public downloads and identities

- [V7 adapter prerelease](https://github.com/AbdullahUsman0/SLM-FineTuning-Testing/releases/tag/v7-r8-pilot-adapter-20261010).
- [V7 adapter ZIP](https://github.com/AbdullahUsman0/SLM-FineTuning-Testing/releases/download/v7-r8-pilot-adapter-20261010/qwen35-2b-lora-v7-r8-pilot-20261010T002204Z-adapter.zip).
- [V7 ZIP checksum](https://github.com/AbdullahUsman0/SLM-FineTuning-Testing/releases/download/v7-r8-pilot-adapter-20261010/qwen35-2b-lora-v7-r8-pilot-20261010T002204Z-adapter.zip.sha256).
- [Pinned r8 source bundle](https://github.com/AbdullahUsman0/SLM-FineTuning-Testing/raw/6bd184364580c4e4108d27b2efb81ee01a27a400/research-checkpoints/v7-approved-policy-20261009/v7-approved-policy-development-20261009-r8.zip).

V7 ZIP SHA-256: `df34422e42fe7c03a4cac3160876061592b37e3e2ae974c5c97c099742a59da8`.

V7 weights: `a9ab464ae78f62e94b3dbb30b3d763563e5b7802dc437bb009acfae9b2a1a1db`; config: `fcbed9d64ab43e0a8fe5382bd7db03cf73cea6e6281d9d0950fe4537678c5f1e`.

V6 weights: `e7add7b8b02d24615ed89067c7761c3985a4a6afb7c84ca2a72e3b06fd84e74d`; config: `38f60de9678324e91fa882ed86c70d09e4a39b27ae28884374ddc3e6fece15ef`.

Pinned r8 ZIP SHA-256: `9f8424895c4f7591b65a5f1aaa7e87104f8035671c272177cd8b6a5bb8bba22b`.

Base: Qwen/Qwen3.5-2B at `15852e8c16360a2fea060d615a32b45270f8a8fc`; pinned fpy at `04d52c015d1e3ecdefe92b87116f209361509b4b`. Verify every artifact before loading; `models-freeze.json` is authoritative. Do not replace the pilot with a later checkpoint.

## Training-PC paths and verification

Python: `D:\SLM\SLM-FineTuning-Testing\.venv\Scripts\python.exe`. Bundle: `D:\SLM\v7-approved-policy-20261009-r8-review`. Private study: `D:\SLM\FYP-evaluation-private\final-extraction-study-20261012T231628Z`. Its `runtime-config.json` identifies both verified adapters and cached base weights. V7 was extracted into a new private `models\v7-extracted\best-adapter` folder.

These paths refer to the training PC; they do not exist automatically on another PC. Download the public adapter ZIP via HTTPS. If independent review resumes later, transfer the private review ZIP securely via AnyDesk, verify its checksum, and preserve its existing pending review state. Do not publish the private conversations, labels or generator. A portable private review package excludes weights, credentials and caches.

Recorded GPU environment: RTX A4000, torch 2.11.0+cu128, transformers 5.15.0, peft 0.20.0, accelerate 1.14.0, safetensors 0.8.0. Parser versions and source hashes are in `runtime-lock.json`. Do not upgrade the working environment for this evidence-only task. The recorded loader uses BF16 base weights and default FP32 LoRA parameters, all on CUDA.

```powershell
$study = 'D:\SLM\SLM-FineTuning-Testing\research-checkpoints\final-extraction-study-20261012T231628Z'
$python = 'D:\SLM\SLM-FineTuning-Testing\.venv\Scripts\python.exe'
& $python -B -m unittest discover -s "$study\tests" -v
Get-Content "$study\EXECUTION_DECISION.json"
Get-Content "$study\PREPARATION_COMPLETION.json"
```

Optional interactive inference on the recorded training PC, using your own messages and the shared new policy:

```powershell
& $python -B "$study\code\manual.py" --arm v7 --config 'D:\SLM\FYP-evaluation-private\final-extraction-study-20261012T231628Z\runtime-config.json'
```

This launches local inference only when explicitly run. It may reject responses admitted by the historical parser because evidence matching is now exact. Do not enter private final-cohort examples. On another PC, first install/verify the recorded runtime and bundle and create a local configuration with correct adapter/base/cache paths; the training-PC command cannot locate missing remote files.

If a future registered run is authorized, the prepared CLI entry points are `code/registration.py`, `code/schedule.py`, `code/evaluate.py` and `code/verify_results.py`. No start command is recommended under the current decision. During an authorized run, `Get-Content <run>\console.log -Tail 20` shows progress; creating `<run>\stop.request` stops at the next call boundary. Keep completed raw receipts and resume the same registered run identity. Do not repair semantic failures or silently rerun completed calls.
