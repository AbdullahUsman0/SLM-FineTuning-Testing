# Latest fpy integration check — October 4, 2026

The sibling fpy checkout was fast-forwarded from `ebfea0b` to
`9bc2bd18c495514efa79a4371be1d706b9fc3824` on `main`, from
`https://github.com/int-abd-5/fpy.git`. The new commits add registry-backed
dataset access, HTTP-boundary hardening, registry health/filtering and Render
deployment. They do not change the clarification/reducer logic relative to
`ebfea0b`; that logic is substantially newer than the model's training dependency.

## Recovery regression and local fix

Replaying `results/peft-chat-custom-20261004T223752Z.json` against unmodified
latest fpy reproduced a data-loss path: deterministic source-mode recovery
replaced a six-update extraction with just the recovered `upload` value.
The initial target, unit, frequency, horizon, target column and format were
discarded, and intent remained ambiguous. The reducer alone preserved them;
the loss occurred in the orchestrator's recovery branch.

The local fix merges the grounded extractor updates into the recovered state.
The existing path for recovering a selected answer from a not-forecasting or
unsupported intent remains intact. A new integration regression uses the
actual combined weather/upload sentence and verifies all six extracted facts,
recovered upload mode, persisted state and exactly one extractor call.
The relevant integration and clarification suites passed **53 tests**.

After the fix, offline replay of both saved turns retained all nine facts.
A fresh actual CPU-model call with the latest schema also returned strict
JSON/schema-valid extraction. It still omitted upload mode and filename in
the raw output; the pipeline recovered upload mode and preserved the six raw
facts, then asked for the source filename. See
[latest-fpy-weather-smoke.json](latest-fpy-weather-smoke.json) and
[update-audit.json](update-audit.json).

The fix is applied **locally in fpy**, not committed or pushed to its remote.
The exact source/test diff is retained in [fpy-recovery-fix.patch](fpy-recovery-fix.patch).
For another clean fpy checkout at the pulled revision, review and apply it from
the SLM repository root:

```powershell
$patchPath = (Resolve-Path research-checkpoints/fpy-update-20261004/fpy-recovery-fix.patch).Path
git -C ..\fpy apply --check "$patchPath"
git -C ..\fpy apply "$patchPath"
```

It is already applied on this authoring PC. Do not apply it twice.

## Runtime and remaining question issue

`scripts/chat-v6-local.ps1` now defaults to the sibling fpy checkout. Restart
the Python chat after updating it; the loaded quantized model server can stay
running. `-FpyMode training` selects the untouched archived training dependency
`04d52c015d1e3ecdefe92b87116f209361509b4b`.
New chat transcripts record the actual dependency commit, local-change status,
Python-source hashes, pipeline-recovered slots and post-turn slot snapshots.
Provider-level `recovery_applied` describes the provider, not the application's
separate deterministic recovery. Do not attribute recovered facts to the model.

The separate required `problem_statement` field still causes a target-related
follow-up after `target_description` is populated. Latest fpy therefore does
not eliminate that redundant clarification. Its fallback examples can also
refer to Bitcoin in a weather dialogue; no domain-specific question cleanup
or schema merger was performed in this update.

New dependency versions change prompt descriptions, question priority and
readiness behavior. This is an application integration check, not a frozen
benchmark or evidence that the v6 adapter improved. Old research results,
corpus files, training snapshot and adapter bytes remain identifiable and
unchanged. No final labels or paid API were used.
