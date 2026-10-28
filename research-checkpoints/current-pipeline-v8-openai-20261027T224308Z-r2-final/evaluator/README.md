# Current-pipeline v8 / OpenAI evaluator

This is an offline-tested development evaluator, not a completed comparison. The
initial run is `D:\SLM\FYP-evaluation-runs\current-pipeline-v8-openai-20261027T222208Z`.
Read its frozen protocol, label audit, credential receipt and completion receipt
before attempting inference. The initial run is blocked before pilot/main calls.

`prepare.py` selects 20 conversations from the prior public 100-case diagnostic,
one per domain, with 16 English, two Roman Urdu and two mixed conversations, using
a digest order and deterministic constraint backtracking. Two disjoint pilot
conversations are selected separately. Neither old scores nor fresh predictions
enter selection. Labels remain unchanged. Current-schema gold replay is a gate;
pipeline differences with gold extraction are recorded separately as application
findings. Native datetimes serialize to ISO strings; candidates are decoded once.

`runner.py` uses the exact current fpy source plus bundled patch, current extractor
instructions and current input construction. The current user turn is appended
before priority ranking, matching the actual orchestrator. Both systems use the
same fixed script with no declared assistant question; static generated questions
are recorded but not fed to the next turn. This is not an adaptive interview.
Existing fpy integration tests exercise adaptive question/recovery behavior offline.
The actual ElicitationEngine replays model responses for application-state scoring;
raw tuple scoring never credits deterministic recovery or reducer normalization.

`controls.py` provides an OS job lock, a shared SQLite FULL-sync budget ledger,
durable response files and fsynced call journals. A hosted reservation is persisted
before dispatch, raw SDK response/usage before reconciliation, and the same saved
response is reused after interruption. Pending/uncertain requests stop dispatch.
SDK retries are disabled. The internal ceiling is US$0.90 against US$1 authorization.
The shared ledger path in protocol.json must be retained across restarts and new
technical run folders. Never initialize a replacement ledger to renew the budget.
Input reservations use full request UTF-8 byte length plus 16,384 overhead tokens;
output reservations cover all 2,048 output tokens, including reasoning. No cached
input discount is assumed. This is a conservative application guard, not provider
billing enforcement. Observed reservation overruns halt subsequent dispatch.

The frozen local arm is CUDA BF16 Qwen/Qwen3.5-2B at revision
15852e8c16360a2fea060d615a32b45270f8a8fc with the verified final-step v8 adapter,
greedy generation, thinking disabled and 1,024 new tokens. The hosted arm is exactly
gpt-5-mini-2025-08-07, Structured Outputs, minimal reasoning and 2,048 output tokens.
Transport and decoding differ. No silent model substitution, output repair,
retraining, paid judging, paid question generation or final-label access is allowed.

Run offline tests with CURRENT_PIPELINE_RUN pointing to the prepared run and
PYTHONPATH placing that run's isolated offline-deps, fpy/src and fpy root ahead of
other installations. The completed test command is recorded in offline-tests.json.
Preparation and inference commands are in the portable LOCAL_CODEX_HANDOFF.md.
Do not run the hosted stages until label, credential, exact-model availability,
test, source-freeze and pilot affordability gates pass. A reviewed contract change
requires a new version and run; preserve old artifacts and reuse the same ledger.

The initial preparation fixes were made before inference: matching the engine's
current-turn ranking, then serializing native datetimes as ISO text. Earlier
preparation receipts/logs are retained. No model outcomes informed these fixes.
Runtime cost/performance, crash recovery against a live provider and the local
model loader still need pilot validation; passing offline tests does not establish
a successful hosted/GPU comparison.
