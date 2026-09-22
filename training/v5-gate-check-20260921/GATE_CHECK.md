# V5 gate check on Colab — corrected provider (a0020e7), 2026-09-21

Agent-driven re-run of the 1-scenario base vs pilot-LoRA diagnostic with the
corrected provider from commit `a0020e7d5301fb6a93bd5ce0f1c23299103cffda`,
followed by a fresh matched development smoke comparison.

## Critical finding: the audited v5 RUN directory never persisted to Drive

`V5_AUDIT_2026-09-21.md` references
`/content/drive/MyDrive/FYP-model-runs/qwen35-2b-lora-v5-20260921T161000Z`
with a pilot adapter (`adapter_model.safetensors`, 67,332,688 bytes, SHA-256
`603d322fb46f66f613e8a3e1b5e18c5ee53f7a16b2730dda39131f6e24726249`).

On 2026-09-21 (~18:30 UTC) the signed-in Drive account
(abdullahmalick168@gmail.com) was inspected directly:

- `FYP-model-runs/` contains only `preflight-corpus-v3-backup/` and
  `qwen35-2b-lora-v3-repro-20260918/`. There is no v5 run directory.
- Drive searches for `qwen35-2b-lora-v5`, the full run name, and
  `best-adapter` return no v5 artifacts. Drive Trash is empty.

Conclusion: the earlier pilot adapter, its run manifest, and the old INVALID
diagnostic reports are not recoverable from this Drive account. Nothing was
overwritten by this gate check because nothing existed to overwrite; the
preservation requirement is satisfied vacuously and recorded here instead.
The old invalid report hash evidence survives only as text in
`V5_AUDIT_2026-09-21.md` (base and broken-LoRA raw output SHA-256
`a199de7005174c8fad5b5ab6d8010e8c86df8375b46be3cdb236eca599e1da0e`).

## Environment constraint: Drive mount impossible in the agent browser

The gate check was driven through the Qoder in-app browser, which blocks all
popups and new tabs (`window.open` returns null even for trusted clicks).
Colab's `drive.mount` OAuth relay requires a popup whose `opener` is the
notebook page; four consent flows completed server-side
(`authorize-for-drive-credentials-ephem` -> `success:true`) but the kernel
always reported `credential propagation was unsuccessful`. The entire gate
check therefore ran VM-local under `/content/gate-a0020e7/` with NO Drive
persistence. Reports were extracted from the VM via notebook output and are
archived in this directory.

## What was run (VM-local, /content/gate-a0020e7)

- Notebook: https://colab.research.google.com/drive/1G1Q8kfdWPAFK2KIsx0w_jq-SOtWhoug0
  (T4, GPU).
- Stage A: cloned `AbdullahUsman0/SLM-FineTuning-Testing` detached at
  `a0020e7d5301fb6a93bd5ce0f1c23299103cffda` and `int-abd-5/fpy` at
  `04d52c015d1e3ecdefe92b87116f209361509b4b`; installed
  `training/requirements.txt` (transformers 5.15.0, peft 0.20.0, trl 1.9.2)
  and `pip install -e fpy`. torch 2.11.0+cu128, CUDA available.
- Stage B: base weights re-downloaded from the pinned revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc` and hash-verified
  (`aa33250c...`, 4,548,221,488 B). `verify-corpus-v5.py --output` passed.
  Length preflight reproduced from the frozen SFT splits (seed-42 shuffle,
  32 train / 16 validation pilot inputs). One-epoch pilot re-trained with the
  committed settings (LR 5e-5, patience 0).
  NEW pilot adapter: 67,332,688 B, SHA-256
  `f649f4ee6178b15c2383f476b73b8ec13bbb1b2024966d425520f6b3c08ab7fe`.
  This is NOT byte-identical to the lost audited adapter (`603d322f...`);
  it is a regeneration from identical pinned inputs/settings, and GPU
  nondeterminism explains the difference. All gate conclusions below refer
  to this regenerated adapter.
- Stage C (1-scenario diagnostic, first line of
  `corpus-v5/v5-20260919-r1/splits/smoke.jsonl`, float16, max-new-tokens
  3072, device cuda):
  - base: rc 0, 286.6 s; lora: rc 0, 137.1 s.
  - No "missing adapter keys" warning in either run; the corrected provider
    (`local_slm_lab/peft_provider.py`) hard-fails on any missing-key warning,
    so rc 0 is proof of zero missing adapter keys.
  - Architecture identical: adapter `auto_mapping.base_model_class` =
    `Qwen3_5ForConditionalGeneration` = base config `architectures`.
  - Raw output hashes DIFFER: base
    `a199de7005174c8fad5b5ab6d8010e8c86df8375b46be3cdb236eca599e1da0e`
    (identical to the old broken-run hash — the base path is unchanged and
    deterministic), lora
    `cb8a87ee3bca96b9e1e61a1430b56a8a53448515c72a58e48b6be9d693826c3b`.
    3 of 4 records differ; the identical one is an `ask` record where both
    models produced the same question JSON.
  - GATE C1/C2/C3: PASS / PASS / PASS.
- Stage D (24-scenario matched development smoke, same flags, reports
  `smoke-base.json` / `smoke-lora.json`): results recorded below once
  complete.

## Stage D results

### First attempt blocked by an evaluation-harness bug (found and fixed)

The first matched development smoke run (24 scenarios -> 102 provider calls,
`corpus-v5/v5-20260919-r1/splits/smoke.jsonl`) failed for BOTH arms at exactly
the same point: call index 8 (extraction, scenario `v5-20260919-r1/inventory/011`,
turn 1). Each arm had journaled 8 records, then the run aborted with
`{"status": "failed", "error": {"type": "TypeError"}}`.

Reproducing call 8 with a full traceback (real LoRA provider, faithful
`on_record`) pinpointed the cause:

```
File ".../local_slm_lab/v5_eval.py", line 397, in evaluate
    on_record(record)
File ".../scripts/evaluate-v5.py", line 161, in recorded
    calls_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
TypeError: Object of type datetime is not JSON serializable
```

`state_snapshot()` emitted `SlotState.value` verbatim. Datetime-typed slots are
normalized to Python `datetime` objects (`fpy` `normalization.py` lines 182-186),
so any record whose `gold_after`/`predicted_after` contained a datetime slot
value could not be serialized by the journal/report writer. `on_record` is
outside the per-call `try`, so a single such record aborted the entire
evaluation. `inventory/011` is the first smoke scenario whose gold state carries
a datetime slot value, which is why both arms stopped at the same call. This is
a genuine harness defect that would also break the full study evaluation, not a
provider or LoRA problem.

Fix (applied identically to the local repo and the VM checkout):

- `local_slm_lab/v5_eval.py`: added `_snapshot_value()` which coerces
  `datetime`/`date`/`time` to `.isoformat()` (the same ISO-8601 text pydantic
  `mode="json"` and `now()` already use) and used it for the slot `value` field
  in `state_snapshot()`. Non-temporal values are unchanged, and because both
  `gold_after` and `predicted_after` pass through the same function the
  transition-equality comparison is unaffected.
- Patched file SHA-256 (local == VM):
  `7c41e8cefc968dd45b4b661f10f9ef77616a17e5efe95c4e27a488e7913d1a01`.
- Regression: `python -m pytest tests -q` on the VM = **161 passed in 13.70s**.

The corrupt first-attempt partials were preserved (moved to
`/content/gate-a0020e7/corrupt-partials/`), not overwritten.

### Clean re-run (interrupted by Colab VM recycle — Stage D NOT completed)

A fresh base-then-LoRA smoke comparison over all 102 calls was launched
detached on the VM with the patched harness (cuda, float16, max-new-tokens
3072). The base arm ran past the previous failure point without error —
confirming the datetime fix — and had journaled **65 of 102** records
(~20-46 s/call) when Colab disconnected the runtime
("inactivity or reaching its maximum duration") and recycled the VM.

After reconnecting, the new VM had uptime ~622 s and `/content/gate-a0020e7/`
no longer existed. Everything VM-local was lost: the regenerated pilot adapter
(`f649f4ee...`), the in-progress smoke reports/journals, the repo clone, `fpy`,
dependencies, and the verified base weights. The run was free-tier Colab with no
Drive persistence (mount impossible in this browser), so there was no off-VM
copy of the adapter or the smoke partials.

Consequences for the gate:

- Stage C (the actual gate condition — zero missing adapter keys, identical
  architecture, different base vs LoRA output hashes) is **PASS** and is fully
  evidenced by `diag-base.json` and `diag-lora.json` (both `status: complete`),
  which were extracted to this archive before the recycle.
- Stage D (matched development smoke comparison) is **NOT completed**. The
  harness defect that blocked it was found, fixed, and regression-tested
  (161 pass). The fix is committed and pushed to
  `origin/experiment/v5-grounded-20260919` as commit `8490d23`
  (`local_slm_lab/v5_eval.py`, SHA-256 `7c41e8ce...`), but a full clean
  base+LoRA smoke comparison still has to be run.
- Handover (user runs Stage B+D in a normal browser where Drive mounts):
  `notebooks/Qwen_2B_LoRA_v5_Grounded_Colab.ipynb` is ready and needs no
  edits. Its `pinned-source` cell fetches `origin/experiment/v5-grounded-20260919`
  and checks out `FETCH_HEAD` (now `8490d23`, i.e. the fixed harness); its
  `pilot-behavior` cell IS the Stage D matched smoke (base + lora over
  `corpus-v5/v5-20260919-r1/splits/smoke.jsonl`, cuda/float16/3072), writing
  `pilot-base-smoke.json` / `pilot-lora-smoke.json` to Drive and preserving any
  existing report. All Stage D inputs were verified committed at `8490d23`:
  `smoke.jsonl` (24 scenarios), the SFT train/validation splits, `manifest.json`,
  `v5-model-provenance.json`, `requirements.txt`, and every entrypoint
  (`verify-corpus-v5.py`, `train_lora.py`, `evaluate-v5.py`, `compare-v5.py`,
  `v5_eval.py`). Caveat: `source-lock.json` is only written when absent, so if a
  stale lock pinning `a0020e7` exists in the Drive RUN folder it must be deleted
  (or the RUN folder renamed in every cell) to avoid re-checking-out the
  unfixed commit. The gate check established that folder never persisted, so a
  fresh run pins `8490d23`.
- Finishing Stage D requires re-establishing a Colab GPU session and repeating
  Stage A+B (clone pinned repo/`fpy`, install deps, re-download + hash-verify
  base weights, regenerate corpus, re-train the one-epoch pilot adapter) before
  the ~2.5 h base+LoRA smoke can run. Because the free-tier session just hit
  its maximum duration, this likely needs either a fresh session window or
  Colab Pro (an upgrade decision reserved for the user), plus a Drive/persistence
  path so the adapter and reports survive a recycle.

## Honest limitations

- The pilot adapter under test is a regeneration, not the audited artifact.
- Reports exist VM-local and in this repo archive only; nothing was written
  to Drive (mount impossible in this environment).
- No claim is made about task quality from these runs; the gate only
  establishes that LoRA weights now load and change model behavior.
