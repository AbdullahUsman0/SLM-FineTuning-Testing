# Local Codex: add the R4 matched development comparison

Safely pull experiment/v5-grounded-20260919 in AbdullahUsman0/SLM-FineTuning-Testing;
preserve local paper edits. Read this publication README, completion.json,
protocol.json, interface-correction.json, slm-reuse-verification.json,
PAPER_UPDATE_TABLE.csv, supplementary-contract-audit.json, failure-findings.json,
billing-usage-analysis.json and offline/replay receipts before updating the actual
existing paper draft. Keep a backup; do not create a replacement manuscript.

Use BOTH R4 rows, never a new hosted row alongside an old strict-scoring SLM row.
OpenAI is fresh; SLM responses were previously generated and were reused only after
verifying identical inputs and native-parser state trajectories. Both were rescored
with the same native admission/actionable-update rules. Report exact precision,
recall/F1, structure/admission, corrections/unknowns, extra actionable slots,
recovery/state success and latency with their denominators. Keep raw extraction
separate from deterministic state recovery and describe the common augmented
adapter and fixed scripted question context. The older R2/R3 conditions remain
historical and must not be erased or called fresh R4 evidence.

This supports a bounded direct matched DEVELOPMENT comparison of configured
systems. It does not establish independent/general model superiority, an
unmodified-production-provider benchmark, real-user adaptive conversation quality,
deployment readiness or independent human annotation. Keep related templates,
previously inspected cases, no confidence intervals, decoder differences and
incomplete human review explicit. Do not infer a new high accuracy from a parser
acceptance rate. Use the measured results regardless of which model scores higher.

The original shared $1 authorization and 440 paid calls remain intact; a separately
recorded user-approved additional $1 raised total authorization to $2. The exported
ledger is evidence, not a new spending account. No further paid calls, training,
sealed labels or automatic expansion are authorized by this completed handoff.

Exact model: gpt-5-mini-2025-08-07. SLM: pinned Qwen/Qwen3.5-2B base and final-step
v8 adapter hashes in protocol.json, with original RTX A4000 CUDA BF16 runtime in
local-runtime.json. Existing adapter release v8-adapter-20261022 remains separate;
this ZIP contains results/source only. Do not assume server D: paths exist locally.

Verify the adjacent ZIP checksum and all ARCHIVE_MANIFEST.json member hashes.
For offline reproduction, restore the exact fpy commit plus verified bundle patch,
use the frozen evaluator/source hashes and offline-tests.json command/environment.
verify_recorded_main.py reproduces the completed raw/state evidence without model
calls in the original run layout. Do not replay completed paid stages or reset
the ledger; source/layout changes on another PC require a documented reproduction.

| Model | Track | Precision | Recall | Exact actionable F1 | Native admitted | Exact pipeline turns | Median seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| v8 | component | 0.976285 | 0.961089 | 0.968627 | 100/101 | 75/101 | 8.082 |
| v8 | rollout | 0.964427 | 0.949416 | 0.956863 | 100/101 | 52/101 | 8.656 |
| openai | component | 0.644518 | 0.754864 | 0.695341 | 101/101 | 54/101 | 2.669 |
| openai | rollout | 0.636364 | 0.762646 | 0.693805 | 101/101 | 6/101 | 2.553 |

R4 usage-rate cost estimate: $0.35448805; cumulative task usage estimate: $0.94693530; cumulative conservative ledger accounting: $1.0469865. Reserved/uncertain $0. Rates include reasoning once; cached input affects the estimate but never dispatch permission. This is not a provider invoice. Total authorization $2; task dispatch ceiling $1.90; this rerun's new-spend ceiling $0.90.

Portable results ZIP: r4-native-comparison-results.zip
SHA-256: 4d4a33bd6ec4ac1bab28b47d37ee324a315177035826119e36a230cd71c143ae
