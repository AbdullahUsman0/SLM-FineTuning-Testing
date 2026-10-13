# Existing evidence and offline study preparation

The user chose **previous results for now**, with **$0 authorized for new paid API work**. No additional training or fresh final-cohort evaluation is running. `EXECUTION_DECISION.json` disables registration, scored runners and hosted-provider access before model loading or network requests. The earlier `budget-request.json` remains an unapproved planning record, superseded by that decision.

Read [PAPER_RESULTS.md](PAPER_RESULTS.md) for the defensible research conclusion, [FAILURE_REVIEW.md](FAILURE_REVIEW.md) for verified errors, and [TEACHER_EXPLANATION.md](TEACHER_EXPLANATION.md) for English and Roman Urdu explanations. [summary.csv](summary.csv) and [summary.json](summary.json) preserve separate cohort IDs and comparability groups.

The 31-conversation historical comparison does not establish a v6–OpenAI winner or equivalence. V7 improved exact F1 over v6 on a different synthetic pilot, but failed its promotion gate because extra slot flags increased. There is no matched v7–OpenAI result. These models extract forecasting requirements; these scores do not measure forecast accuracy.

## What is completed

- Verified the historical checkpoint's 18 recorded file hashes, reproduced its 744 raw admissions/primary scores, and checked matched messages, gold and component contexts under the unchanged historical policy.
- Verified the pilot result archive and replayed all six pilot exact metrics under original scoring. Reproduced evidence, structural failures and selected language/category diagnostics.
- Verified v6 provenance, v7 ZIP and adapter files, pinned base weights, parser dependencies and the pinned fpy snapshot. See `models-freeze.json`, `runtime-lock.json`, and `environment.json`.
- Added one new versioned admission path for local arms, hosted arms and optional manual chat, preserving the old policy files. Exact current-message evidence is now a hard gate for the prospective study.
- Prepared a private 240-conversation candidate, rubric, review sheet, ambiguity queue and overlap audit. This is agent-authored, structurally validated, **pending independent human review**, and unconsumed by model scoring.
- Completed CUDA readiness checks on separate fixtures, with BF16 base weights and PEFT's default FP32 LoRA parameters for both arms. All model parameters were on CUDA; no CPU offload. These fixtures are not a final quality evaluation.
- Prepared a durable raw-response journal, budget reservations, resume checks, grouped schedule and independent result verifier. Offline tests exercise interruption recovery and parser behavior; paid execution has not been validated in this new study.

## Preservation and publication

This public package contains code, policy, aggregate candidate metadata, old public evidence and reports. New conversations, proposed labels, review materials and their deterministic generator remain outside Git at:

`D:\SLM\FYP-evaluation-private\final-extraction-study-20261012T231628Z`

That path exists on the training PC. It does not imply a copy exists in another PC's Downloads. No base weights, adapters, optimizer checkpoints, credentials or caches are included here. Existing adapter release links are in [REMOTE_HANDOFF.md](REMOTE_HANDOFF.md).

The public offline ZIP includes its own per-file manifest. Its adjacent `.sha256` verifies the ZIP. The separate private review ZIP is for secure transfer only; do not upload it to a public release. `PREPARATION_COMPLETION.json` records readiness and pending work, without claiming a completed final evaluation.

## Offline verification

On the recorded training PC, with the pinned r8 bundle present:

```powershell
Set-Location D:\SLM\SLM-FineTuning-Testing
$study = 'research-checkpoints\final-extraction-study-20261012T231628Z'
& .\.venv\Scripts\python.exe -B -m unittest discover -s "$study\tests" -v
Get-Content "$study\EXECUTION_DECISION.json"
Get-Content "$study\PREPARATION_COMPLETION.json"
```

Run `code/evidence_only.py` only in a new output copy: it writes exclusively and intentionally refuses to replace existing summaries. Historical inputs must be available at their recorded repository-relative paths. See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the deferred plan and [IMPLEMENTATION_CHANGES.md](IMPLEMENTATION_CHANGES.md) for material changes.
