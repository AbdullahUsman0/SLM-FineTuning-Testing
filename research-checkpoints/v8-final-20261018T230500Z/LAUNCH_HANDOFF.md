# Active authorized v8 training experiment

Read `D:\SLM\active-v8-final.json` first. These paths are on the GPU training PC, not automatically on another PC.

- Repository: `D:\SLM\SLM-FineTuning-Testing`, branch `experiment/v5-grounded-20260919`.
- Starting commit: `bd36ff605d1d7e6c5f79de6e65d60785e0afe49d`. The checkout was clean and fast-forwarded to this exact commit.
- Run: `D:\SLM\FYP-model-runs\qwen35-2b-lora-v8-final-20261018T230500Z`.
- Training Python: `D:\SLM\FYP-environments\v8-final-20261018T230255Z\Scripts\python.exe`.
- Dependency: dedicated clean `runtime\fpy-final-pinned`, detached at `04d52c015d1e3ecdefe92b87116f209361509b4b`. The sibling `D:\SLM\fpy` was preserved; its incidental appearance in the historical trainer's Git inventory does not select it for evaluation.
- GPU: NVIDIA RTX A4000, 16 GB, CUDA 12.8 runtime, torch 2.11.0+cu128. Core GPU package versions match the handoff. A new environment reuses those packages through a read-only .pth reference; missing jsonschema dependencies were added only to the new environment. Preserve the original `.venv` as well as the new environment.

This is the explicitly authorized final **experimental** training scope. It does not change old human-review gates, authorize final-test access or permit paid APIs. Do not read the private prospective 240-conversation cohort. Human review remains incomplete.

The complete v8 dataset and reducer checks passed: 3,600/800 training/development conversations, 17,860/3,980 examples, required domain/language ratios, no cross-split entity or exact-input overlap. All 22 supplied tests passed. Complete tokenizer preflight passed with maxima 3,547/3,531 and zero examples over 4,096. Cached pinned base-model weights and the existing v6 adapter hashes were independently verified.

The supplied detached `training/final_round/run.py` worker owns training and the v8 development diagnostic. It is a fresh rank-16 LoRA on the pinned stock Qwen3.5-2B revision, two epochs, seed42, effective batch16, learning rate5e-5, no truncation, no early stopping, all complete ten-step checkpoints retained. Actual Trainer maximum is 2,234 optimizer steps. Do not alter settings or global_step. The final `best-adapter` means final-step, not behaviorally selected best.

## Inspect actual progress

```powershell
$a = Get-Content -Raw D:\SLM\active-v8-final.json | ConvertFrom-Json
& $a.python -B D:\SLM\SLM-FineTuning-Testing\training\final_round\run.py status --run $a.run
Get-Content "$($a.run)\training-console.log" -Tail 5
Get-Content "$($a.run)\full\events.jsonl" -Tail 2
Get-Content "$($a.control)\followup-status.json"
```

A running status/PID alone is insufficient. Check the real process, advancing logs and `checkpoint-complete.json` artifacts. `first-checkpoint-verification.json`, when present, independently hashes every checkpoint artifact against the run fingerprint and checks the actual Trainer max_steps. It is a historical verification snapshot; later progress lives in the run logs/checkpoints.

Both workers request `ES_CONTINUOUS | ES_SYSTEM_REQUIRED`, without a display-required flag. Windows screen lock and client disconnect are supported. The GPU PC must remain powered on; explicit sleep, shutdown and logout are not prevented. The administrative `powercfg /requests` query was unavailable, so no administrative power-plan changes were made. Successful worker entry/heartbeat confirms the sleep-inhibition API calls succeeded.

## Automatic follow-up

The single active companion is `followup-r4.py --config workflow-config-r4.json`, launched detached through the supplied background helper. Earlier companion revisions/configs remain preserved; only idle companions were replaced during setup. The actual trainer and frozen training/evaluation sources were never restarted or changed.

After the supplied worker completes, the companion:

1. Verifies all complete checkpoints and the final adapter receipt against 2,234 steps.
2. Runs the identical 100-conversation component/rollout diagnostic on the verified v6 adapter, in `v6-same-development-diagnostic` within the same run.
3. Saves exact F1/precision/recall, structural/admission/state/correction/unknown/latency metrics, language and domain-family summaries, all twenty individual domains and automated failure findings. No new final significance test or definitive winner is claimed.
4. Writes the final adapter ZIP and checksum to this PC's Downloads, reads the ZIP members back and verifies their hashes. Base weights, optimizer states, credentials and caches are excluded.
5. Writes compact result/provenance/raw-development archives under `research-checkpoints/v8-final-20261018T230500Z`, scans text and decompressed raw members for credential-shaped strings, and commits/pushes only those records. If the remote advanced or unrelated staged work must be preserved, publication uses a separate clean detached worktree. This behavior passed a local bare-repository integration test. Current real push access passed a non-mutating dry run.

Development results and the final adapter do not exist until those phases complete; do not substitute training loss or token accuracy. Read `followup-status.json` for completion/publication state. Release assets for v8 weights are not automatically published; the adapter ZIP is local for transfer, while compact scientific records go to Git.

## Recovery

Do not launch another trainer while one is alive. If interrupted, inspect the actual failure and unchanged source/config/input hashes. Resume **this same run** through the supplied `run.py start --run <same path>` only from its latest SHA-verified complete checkpoint. A run without any complete checkpoint cannot be called resumed. Preserve partial checkpoints and failures; never reset counters or pretend a fresh start is continuation.

If training/development completed but the companion failed, inspect its log and rerun only the appropriate preserved companion phase; do not needlessly restart completed training or regenerate existing diagnostic journals. A companion failure is preserved in `followup-status.json`; it never starts training from zero. A new source/data/hyperparameter version requires a new experimental version rather than silently editing an active run.
