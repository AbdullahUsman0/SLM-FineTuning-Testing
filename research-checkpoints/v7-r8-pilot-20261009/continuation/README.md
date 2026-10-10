# Automatic continuation after the v7 r8 pilot

The user requested continuation after completion. The hidden Windows continuation process waits for the pilot's verified completion and trainer exit, then performs fresh CUDA BF16 stock/v6/v7 extraction comparisons with the v7 prompt on the same 200 selected validation conversations. It performs 1040 component and 1040 rollout calls per arm: 6240 scored extractions, plus one unscored warmup per arm. No paid API arm is available.

The code, inputs, current v7 annotation policy, shared wire parser, generation settings and call plans are frozen before the first scored calls. Component context matches the actual r8 validation SFT. Rollout uses each model's accepted state; the original declared synthetic repair interventions are applied at their declared turns. Repair cases and rollout excluding repair cases are reported separately. The 20 shared entity groups, containing ten related conversations each, are the bootstrap resampling units. These are development cases with shared templates, not an unseen-user final benchmark.

Ten CPU-only tests passed. They check every gold-oracle component and rollout turn (1040 each), actual provider prompt routing, failed-call penalties, duplicate/limit rejection, preservation of numeric-looking strings, evidence/reducer behavior, matched-context rejection and clustered bootstrap. Supplemental numeric diagnostics distinguish strings/booleans while treating numeric 1 and 1.0 equally. The frozen primary exact tuple metric remains unchanged.

After all arms finish, the process independently reparses all 6240 records, replays rollout contexts, reproduces primary metrics, verifies checkpoint/adapter hashes, calculates paired comparisons and descriptive subgroup diagnostics, and packages raw reports and jobs. It publishes the results ZIP and checksum on the existing experiment branch using an isolated Git worktree, then verifies the actual GitHub download. Base weights, optimizer checkpoints, credentials and caches are excluded. A separate verified experimental final-step pilot adapter ZIP is placed in Downloads for local testing.

No larger training round starts automatically. A next-round decision needs the fresh behavior reports and human failure review; loss alone is insufficient. Interactive V7 runtime admission differs from the shared benchmark parser and is disclosed in the results. No sealed final labels are read or generated and no historical result is changed.

Live files under the run directory:

```text
D:\SLM\FYP-model-runs\qwen35-2b-lora-v7-r8-pilot-20261010T002204Z
continuation-status.json
continuation.console.log
continuation.stderr.log
evaluation/<stock|v6|v7>/status.json
continuation-completion.json  (after verified publication)
```

The process requests system-awake state while it runs. Training failure or infrastructure/integrity failure stops the continuation with a status record; it does not silently restart training. Completed evaluation jobs are retained with hashes for an explicit later resume under the same frozen identity. A continuation-stop.request file in the run directory stops the waiting/evaluation process; it does not stop training.

Commands run from the existing GPU environment with the frozen source bundle and its isolated .review-deps on PYTHONPATH. Code paths and provenance refer to this remote PC. The initial setup commit records the planned pipeline, not completed evaluation results.
