# Final GPU training handoff

Read `REMOTE_GPU_PROMPT.md` for the complete agent instructions. Dataset: `corpus-v8/v8-final-20261018-r1`. The frozen manifest SHA-256 is:

`abedd97e642152786db56d3e98a0f997d1c241bf6549365c56ddd01ac5cab2e9`

This authorizes one experimental full-data training run and local development diagnostics. It is not a human annotation approval or permission for paid APIs/final-test execution. Read `EXPERIMENT_PLAN.md` for the fixed schedule and limitations.

## Quick start on the Windows GPU PC

Use the existing CUDA training environment after checking the pins in `requirements.txt`. Do not replace CUDA PyTorch with a CPU wheel. The historical working runtime used Python 3.12, torch 2.11.0+cu128, transformers 5.15.0, peft 0.20.0, TRL 1.9.2 and tokenizers 0.22.2. Record the actual environment. If rebuilding, use a dedicated environment and an official CUDA wheel compatible with the driver.

Create a dedicated dependency checkout; do not reset a sibling fpy with local work:

```powershell
Set-Location D:\SLM\SLM-FineTuning-Testing
git clone https://github.com/int-abd-5/fpy.git runtime/fpy-final-pinned
git -C runtime/fpy-final-pinned checkout --detach 04d52c015d1e3ecdefe92b87116f209361509b4b
$env:HF_HOME = 'D:\SLM\hf-cache'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:PYTHONUNBUFFERED = '1'

& .\.venv\Scripts\python.exe -B training/final_round/test_final_round.py --fpy runtime/fpy-final-pinned
& .\.venv\Scripts\python.exe -B training/final_round/data_tools.py `
  --fpy runtime/fpy-final-pinned

$run = 'D:\SLM\FYP-model-runs\qwen35-2b-lora-v8-final-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
& .\.venv\Scripts\python.exe -B training/final_round/run.py prepare `
  --fpy runtime/fpy-final-pinned --run $run
& .\.venv\Scripts\python.exe -B training/final_round/run.py start --run $run
& .\.venv\Scripts\python.exe -B training/final_round/run.py status --run $run
```

Reuse an existing dedicated dependency checkout only after verifying its exact commit and clean tracked state. Keep the exact `$run` path in an active-run pointer file outside Git. `prepare` refuses existing directories, checks at least 80 GiB free, requires a single BF16 CUDA GPU, validates all gold trajectories, materializes byte-verified inputs and tokenizes every example without loading model weights. It refuses truncation. An error during prepare is not a started training run; inspect the error before choosing a new output folder.

The fixed run is two epochs with a fresh stock base and LoRA, seed 42, effective batch 16 and checkpoints every ten optimizer steps. The expected schedule is 2,234 optimizer steps. All checkpoints are retained, so disk requirements are substantial. Checkpoints include optimizer, scheduler, RNG state and per-file completion hashes. Do not confuse a small adapter ZIP with a resumable training checkpoint.

## Background operation and resuming

`start` launches a detached worker, records its PID and redirects logs to files. The Windows worker requests `ES_CONTINUOUS | ES_SYSTEM_REQUIRED` for its lifetime; the display can turn off and Windows can be locked. This prevents ordinary idle sleep, not user-requested sleep, shutdown, restart, logout or every laptop-lid policy. Do not close a lid configured to sleep. A GPU server must stay powered and awake. Disconnecting AnyDesk or locking the client laptop does not require the client to run the training process. The remote job, once launched and verified, runs on the GPU PC.

On Linux the worker starts a separate session. Use the server's supported sleep inhibition/service policy if it has automatic suspend; this code does not assert that every login manager allows jobs to survive logout. Screen locking and logging out are different operations.

Verify that global step advances and at least one complete checkpoint exists before leaving the machine. Read `status.json`, `training-console.log`, `full/events.jsonl` and `worker-console.log`. A PID/status receipt alone is not proof of a live or healthy training process.

To resume after a failure, inspect logs, retain the exact code/environment/input files, then call **start on the same run path**. If a training manifest exists, the worker passes `--resume-from-checkpoint auto`; the unchanged trainer chooses the latest complete, hash-verified checkpoint and refuses an unsafe restart from zero. A failed run without any complete checkpoint cannot be resumed; preserve it and report the limitation rather than changing displayed progress. OS file locks reject a simultaneous second worker.

The launch config retains the non-secret model-cache location, selected CUDA device and explicit offline flags. A resumed worker restores these even when started from a different terminal environment. It does not record API credentials.

After technical training completion, the worker automatically runs a resumable 100-conversation local development diagnostic in both component and rollout tracks. It verifies adapter loading, journals raw outputs, and writes structural, exact F1, state, correction and language/domain summaries. It does not call OpenAI or consume a final test. The worker's final status is complete only after both phases finish.

## Expected saved evidence

- `inventory.json`, `preflight.json`, immutable launch config and hashes.
- `inputs/` with exact uncompressed train/validation bytes.
- `full/run-manifest.json`, `events.jsonl`, all verified `checkpoint-*` directories, `evaluation.json` and final-step `best-adapter/`.
- `training-completion.json` with adapter hashes and verified checkpoint steps.
- `development-evaluation/freeze.json`, `calls.jsonl`, `scored-records.jsonl.gz`, `runtime.json` and `summary.json`.
- Status and separate supervisor/training/development logs.

Do not name training loss/token accuracy as extraction accuracy. Do not call the final-step folder a behaviorally selected winner. Summarize failures as well as improvements. New results require their own cohort/policy identifiers in the paper.

The same development diagnostic can be run on the existing v6 adapter in a separate output folder:

```powershell
& .\.venv\Scripts\python.exe -B training/final_round/evaluate_development.py `
  --fpy runtime/fpy-final-pinned `
  --adapter D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter `
  --output "$run\v6-same-development-diagnostic"
```

This is an inspected development comparison, not a final OpenAI comparison. The previous hosted execution budget remains zero. Use the separate prospective study only after independent review and a new explicit spending decision.

## Verification boundaries

This preparation machine performed complete CPU-side dataset/reducer checks and tested detached child-process survival after launcher exit. It did not run GPU training, tokenizer preflight or model inference for this new corpus. Those are mandatory remote checks. Current synthetic labels and language/overlap judgments still need independent human review before strong publication claims.

Background API references: [Python subprocess](https://docs.python.org/3/library/subprocess.html) and [Microsoft SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate).
