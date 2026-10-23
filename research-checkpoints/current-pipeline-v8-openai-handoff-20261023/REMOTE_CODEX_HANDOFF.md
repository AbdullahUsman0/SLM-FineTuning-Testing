# Remote Codex handoff: current-pipeline v8 versus OpenAI

Continue my FYP requirement-extraction research on THIS remote GPU PC. Implement and run a fresh, reproducible comparison of the verified v8 local adapter and the hosted OpenAI baseline under the current fpy pipeline. Execute the work in stages and preserve all prior work. No further training is requested.

## Authorization and boundaries

- The user approved a maximum of **US$1 total OpenAI API spending** for this evaluation. This supersedes the previous $0 hosted restriction only for this bounded task. It includes pilots, main evaluation, failed requests, any explicitly implemented retries and any question-generation requests. It is not $1 per stage or per restart.
- Keep a conservative US$0.10 margin: dispatch no request if reconciled cost plus outstanding/uncertain reservations plus the next request's conservative bound would exceed **US$0.90**. Do not exceed the user's US$1 ceiling or purchase compute/credits. Treat this as an application spending guard, not a guarantee about provider billing enforcement.
- If credentials are absent, prepare everything offline and ask the user to configure the key securely on this machine. Do not reuse API keys pasted into old chats, print secrets, or include keys, .env files, cookies or authorization headers in Git/logs/bundles.
- Do not retrain, modify adapters, change gold after inspecting predictions, access sealed final labels or open the private prospective cohort. No claim of independent human review is authorized. Do not rerun all six extraction strategies or introduce another model-size study within this budget.
- Do not ask the user to reconfirm routine reversible setup. Stop only for a real missing prerequisite, unresolved label/contract decision, unavailable requested model, exhausted budget or other action requiring new authorization.

## 1. Locate repositories and preserve work

Research repository: https://github.com/AbdullahUsman0/SLM-FineTuning-Testing

Branch: `experiment/v5-grounded-20260919`

Required completed-evidence ancestor: `45fd8bd0edfe4996d961d638c5d5d8de63f488bd`.

Expected research checkout is `D:\SLM\SLM-FineTuning-Testing`; locate the real path. Read applicable AGENTS.md. Fetch and fast-forward safely. If a dirty/diverged checkout blocks updating, use a dedicated checkout/worktree; do not reset, discard edits, delete runs or force-push.

Read these research files first:

- `research-checkpoints/v8-final-20261018T230500Z/{README.md,completion.json,comparison.json,summary.csv,ADAPTER_RELEASE_NOTICE.md}`.
- `training/final_round/{EXPERIMENT_PLAN.md,evaluate_development.py,data_tools.py}`.
- `evaluation/current-pipeline-v8-openai-20261023/PLAN.md` if already present, otherwise use the attached bundle's copy.
- The downloaded bundle's `PREPARATION.json`, `API_BUDGET_AUTHORIZATION.json`, `BUNDLE_MANIFEST.json`, `fpy-local-fixes.patch` and `fpy-working-files/`.

The published handoff ZIP is `research-checkpoints/current-pipeline-v8-openai-handoff-20261023/remote-v8-openai-handoff-20261023.zip` on the experiment branch. Verify its adjacent SHA-256 file, extract it into a NEW directory, then verify every member against BUNDLE_MANIFEST.json. The local fpy fixes are INSIDE that ZIP; pulling fpy main alone does not recover them. Use the experiment checkout and the verified extracted bundle together.

## 2. Freeze the current application, including local fixes

fpy repository: https://github.com/int-abd-5/fpy.git

Source snapshot to reproduce: `9bc2bd18c495514efa79a4371be1d706b9fc3824`, plus the four modified files captured in the bundle. The old v8/v6 diagnostic instead used `04d52c015d1e3ecdefe92b87116f209361509b4b`.

Use a new dedicated fpy checkout at the current source snapshot. If the remote fpy has newer work, preserve it separately and document the difference; do not silently substitute another revision during the comparison. Apply the verified patch using `git apply --check` first. If line endings prevent applying it, use the four bundled working-file copies only in this dedicated checkout, after verifying their expected hashes. Do not overwrite an unrelated working fpy.

The four files are:

1. `src/forecasting_assistant/application/orchestrator.py`
2. `src/forecasting_assistant/prompts/llmrei_long.py`
3. `tests/integration/test_elicitation_flow.py`
4. `tests/unit/test_question_policy.py`

They implement known-target acknowledgment, avoid treating a correction as an answer to another selected slot, and retain other grounded updates during selected-slot recovery. Domain/schema, reducer, prompt and provider code also changed between the old pinned revision and current main. Freeze the WHOLE selected fpy revision plus patch, not just one prompt file. Record source hashes and import paths in every run. Do not import a different installed fpy accidentally. Normalize line endings only for explicitly labeled content comparisons; retain exact-byte source hashes too.

## 3. Establish the fixed local model and environment

Use the existing verified v8 final-step adapter if available. Expected weights SHA-256:

`c83e0f1a458125970eb70ff360ebead02618f2f48b5bbb59e2ca91a5b7c4c0df`

Configuration SHA-256:

`a1ce14418c5c22375ccfd773338ef1b09117ec9b9e72db034c71ade886b4664c`

If missing, use `research-checkpoints/v8-final-20261018T230500Z/training-completed-20261023T002215Z/download_adapter.py --output <NEW-directory>` and its committed ADAPTER_RELEASE.json. Release tag: `v8-adapter-20261022`. ZIP SHA-256:

`d2245b988f5b5e6a336c4518f3a161c46e81fc3fe55284735e53bd6ece096653`

Load only on stock `Qwen/Qwen3.5-2B`, explicitly pinned to `15852e8c16360a2fea060d615a32b45270f8a8fc`, using the documented `Qwen3_5ForConditionalGeneration` architecture. The adapter config has no base revision, so explicitly pin it in the loader. Do not stack adapters or substitute quantized/CPU/NPU inference for the CUDA research arm. Historical `best-adapter` naming means fixed final step, not behavioral selection.

Reuse the working RTX A4000 CUDA BF16 environment and HF cache if available. Preserve CUDA PyTorch. Inspect GPU, VRAM, RAM, disk and runtime versions. Make an isolated evaluation environment if necessary. Capture package versions, GPU/driver, model class, base revision, active adapters, dtype and chat-template hash. No local model calls on the main laptop are needed.

## 4. Hosted model and official documentation

The intended hosted snapshot remains **`gpt-5-mini-2025-08-07`**, to retain continuity with the previous hosted study. Do not silently replace it with an alias or newer model. Its official page currently labels it deprecated; verify that the configured account can still use that exact snapshot. A free metadata check is preferable to an unnecessary probe. If unavailable, stop hosted execution and ask the user to choose an alternative snapshot; preserve all offline/local progress.

References inspected when preparing this handoff:

- https://developers.openai.com/api/docs/models/gpt-5-mini
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/reasoning

At preparation, the model page lists input US$0.25, cached input US$0.025 and output US$2 per million tokens. Recheck official pricing before dispatch; these are not a promised run cost. Reasoning tokens consume output allowance and cost. Persist complete SDK responses and usage before scoring; parsed Pydantic dumps alone do not preserve the raw emission. Log refusals, incomplete responses and token-limit failures honestly. Structured Outputs constrains representation, not semantic correctness.

## 5. Build the new evaluator; do not reuse old runs as fresh results

The current-pipeline evaluator does NOT yet exist. Implement it and its offline tests, reusing reliable journaling/scoring components where appropriate. The old `training/final_round/evaluate_development.py` is local-only and pinned to old fpy; the older policy comparison runner also imports an old pinned application. Do not simply point them at the new adapter and call that a current-pipeline comparison.

Freeze one common extraction schema, semantic instructions, current-message payload, label policy and application reducer for both arms. Use current fpy's extractor instructions/context; ensure the local model receives that same content rather than an older V5/V8 builder accidentally. Keep one candidate-value decode and explicit provided/inferred/confirmed semantics. Verify how Pydantic normalizes values. Do not let a representation mismatch or double decoding silently dominate F1.

A small wrapper around the existing low-level local generation method may be reused, but test that it sends the actual frozen current inputs. The hosted arm uses provider Structured Outputs; the local arm uses its recorded greedy decoding. Describe this as a comparison of configured extraction systems, not identical decoding algorithms. Begin with local max_new_tokens 1024 and hosted max_output_tokens 2048 with minimal reasoning, if supported by the exact snapshot. Freeze these settings before main inference and disclose the different budgets. Do not silently increase a capped response's allowance after viewing it.

Measure TWO layers from actual recorded calls:

1. **Raw extraction:** score the proposed updates before deterministic fallback, selected-slot recovery, normalization repair or state assistance. Report structural admission separately. A recovered value is not a model-generated true positive.
2. **Application state:** apply the frozen current pipeline, record which values came from the model versus deterministic recovery, and compare the resulting values/status/confirmation flags with expected state. Report exact state at turns and after corrections with a precise state definition.

Use component (gold prior) and rollout (own accepted prior) tracks. Rollout must never receive gold prior slots except an explicitly declared synthetic intervention. Keep identical scripted messages and declared assistant-question context across systems for the fixed-script study. Test adaptive question wording and selected-slot recovery separately offline; do not mislabel a fixed script as a real-user interview. Additional live question-generation calls are not required in this budget.

## 6. Freeze a small matched development cohort within the budget

Do not create a new training corpus. Deterministically select **20 conversations** from the existing public 100-conversation V8-D diagnostic: one per individual domain, six weather/twelve economics/two combined, with sixteen English/two Roman Urdu/two mixed if feasible. Use a seed/digest rule fixed before any new predictions; resolve selection constraints programmatically and record every selected ID. Selection must not optimize either model's expected score. Report the actual slot support and turn counts; do not claim full coverage of all 79 slots.

This is a NEW current-pipeline development measurement, not an untouched final test. Save it under a new cohort name, for example `current-pipeline-dev20-20261023-r1`. The 100 old cases and their performance have already been inspected. Do not call this subset independent or claim a universal final winner.

Before model calls, validate gold updates and state transitions under the current schema, reducer and label policy. Reject impossible labels and produce an annotation review queue if conventions conflict. Do not automatically rewrite old labels or claim human adjudication. If no defensible common gold/state contract can be frozen, finish the runner and audit, then request adjudication before the main paid comparison.

Choose up to two separate pilot conversations from the remaining 80 cases. Pilot calls and outcomes are excluded from the main paired result; cost is still included. Prefer fixtures covering dense input, cadence distinctions, correction, unknown information and recovery. Run cheap offline fixtures before the paid pilot, and reconcile pilot usage before deciding whether the full frozen 20-case main set is affordable. Do not use pilot outcomes to select easier main cases.

Never automatically expand to the whole 100-case cohort or all six strategies. If the approved budget cannot cover the planned set, report the incomplete coverage honestly. Freeze a smaller design before main outcomes only if the cost preflight requires it; record the change and do not describe it as the original 20-case design. A partial observed subset is not a completed planned experiment.

## 7. Implement and test the spending guard before OpenAI calls

The user authorization is US$1 total. Implement a durable ledger with a single dispatch worker and no concurrent paid requests. Use an exclusive lock. Before each request, persist a conservative reservation for input plus maximum billed output, including reasoning; record the count/upper-bound method, schema/context overhead and current rates. Do not rely on cached-input discounts to make the cap fit.

SDK automatic retries must be disabled (`max_retries=0`). An uncertain timeout may still be charged: retain its reservation and stop for reconciliation rather than blindly repeating it. On a complete response, fsync the raw response/usage and reconcile that reservation before the next dispatch. Budget state must survive restart and span pilot/main/both tracks. Do not initialize a fresh $1 ledger on resume. If the estimator cannot defend a conservative next-request bound, stop rather than claiming the cap is guaranteed.

Reserve at most US$0.90 in total, keeping US$0.10 below the user's US$1 ceiling. Reject the next dispatch if actual reconciled spending plus outstanding/uncertain reservations plus its proposed reservation would exceed that internal threshold. Stop earlier if actual usage exceeds estimates. No batch jobs, additional models, tools, API-based judging or separate API question generation under this authorization.

Offline tests must cover: budget exhaustion, a pending reservation at restart, an uncertain timeout, token usage including reasoning, duplicate journal keys, truncated journal tails, context drift, capped/refused output, malformed/duplicate fields, double-decoding hazards and schema-valid but wrong values. These are tests of scientific and spending controls, not additional paid experiments.

## 8. Execute in recoverable stages

1. Verify the bundle, source snapshots, adapter and environment. Run existing relevant fpy question-policy/elicitation tests plus the new evaluator tests without paid calls.
2. Save case selection, label audit, protocol/config/source hashes, and a full manifest. Record whether labels received independent human review (currently incomplete).
3. Run a local smoke test, then the bounded hosted pilot if credentials/model/ledger/gold gates pass. Count every hosted request against the one shared cap.
4. Freeze the main protocol after technical pilot fixes. If settings changed, do not blend pilot outcomes with main outcomes or resume a mismatching run folder.
5. Run fresh v8 and OpenAI component/rollout calls on the same selected main cases. Do not import either model's old responses. Save raw-model and pipeline-state outcomes separately.
6. Run the bounded supervisor independently of the AnyDesk/client connection. On Windows use a hidden background process (`Start-Process -WindowStyle Hidden`) with a fixed script path, output/error logs, PID/status file and exclusive job lock. Do not claim it was launched before checking a live PID and new journal/status output. A locked client laptop does not stop the remote process, but remote sleep/reboots can; recover through durable journals rather than editing progress counters or replaying paid calls blindly.
7. Freeze implementation changes before the main run. Do not tune prompts, adapter, labels or parser after seeing main failures. New fixes require a new version; never silently continue the same experiment with different conditions.

## 9. Analysis and historical-result audit

Report each track and each model separately:

- Exact non-intent slot/decoded-value/status micro precision, recall and F1; raw-model emissions before recovery and operational admission policy explicitly identified.
- JSON syntax, declared-schema validity, application admission, duplicate slots and evidence failures.
- Extra and forbidden slots, explaining that flags are not a count of all hallucinations or wrong values.
- Correct correction flag/updates, complete state at turns and after corrections, successful unknown omission and prior-state preservation.
- Per-slot support, domain/language breakdowns with denominators, explicit context-intervention cases and exclusions.
- Median extraction latency and request/token usage; transport/runtime/decoding differences disclosed. Record local GPU/runtime details; do not infer a general cost/energy advantage from this study.
- Actual/reserved/uncertain API cost and whether the planned comparison finished within the cap.

Prespecify any uncertainty analysis before main results. With only 20 related synthetic conversations, intervals are exploratory. Keep paired cases together; account for entity/template grouping where appropriate rather than resampling individual turns as independent. If a defensible confidence interval cannot be specified, report descriptive estimates and the limitation, not a fabricated significance claim.

Separately audit the historical OpenAI reports offline: model IDs, cases, labels, acceptance/inner-encoding rules, prior-state construction, truncated/refused answers, completion coverage and metric arithmetic. A pipeline change makes an old study historical; it does not prove its archived arithmetic was wrong. Record actual defects if found. Keep any rescoring of stored outputs labeled as post-hoc rescoring, not fresh inference. Do not replace archived scores with the user-supplied slide figures 60.86%/77.86% without supporting matched reports.

Do not compare the old 31-case OpenAI score or v8's 100-case 96.40% directly with the new 20-case scores. Do not attribute deterministic pipeline recovery to model capability. Do not declare either model superior, equivalent or deployment-ready solely from point estimates.

## 10. Save evidence and return a local-main-PC handoff

Use NEW unique run directories. Preserve the 2,234-step adapter and all older runs. Suggested public result folder: `research-checkpoints/current-pipeline-v8-openai-<UTC-timestamp>/`.

Save `completion.json`, per-arm summaries and raw call journals, protocol/source/case/model hashes, reducer/admission rules, cost ledger, usage, error records, raw-versus-pipeline comparison, descriptive subgroups, offline tests, exact commands and limitations. A completion receipt must distinguish successful full completion, budget-stopped partial results and blocked label review. Include the published current-application commit/patch identity so this PC can reproduce the runtime.

Create a credential-checked portable results ZIP with an internal SHA manifest and adjacent SHA-256 file. Verify archive CRC/member hashes and a downloaded/copy round trip if it is subsequently uploaded. Prepare a publication-ready commit/diff for the experiment branch; publishing new Git commits or release assets is not required by this handoff unless the human separately authorizes it. Do not commit model weights, optimizer state, credentials or private labels to normal Git. If preparing a public result bundle, ensure source patches are appropriate for public release first.

Write `LOCAL_CODEX_HANDOFF.md` with:

1. Actual run/output paths, source revisions/patch hashes, GPU/runtime and adapter hashes.
2. Exact completed/planned cases and calls per arm/track, budget used/reserved and remaining work.
3. Results, raw-model versus pipeline differences, observed failures and honest review/inference limits.
4. Any historical scoring defects demonstrated by audit; no unsupported claim that earlier results were universally wrong.
5. Portable ZIP location/SHA, complete reproduction commands, and explicit instructions for the main-PC Codex to update the existing paper only from completed matched receipts.

Do not fabricate a replacement manuscript if the actual paper source is absent on this GPU PC. The current main-PC paper has six IEEE pages, separate N/M/P/V8-D cohorts and no matched v8-OpenAI result. Return a paper-update table/proposal rather than inserting unsupported new claims. Stop after this bounded evaluation and packaging; no further training or automatic experiment expansion.

## Delivery expectation

Work through the stages, not merely a plan. Report concrete results or the precise gate that prevents execution. Never describe missing cases, incomplete labels, exhausted budget or a failed process as a finished comparison. The approved budget is US$1 for the entire task, with no implicit renewal.
