# Proposed matched v8/OpenAI reevaluation on the current pipeline

Prepared 23 October 2026. This is offline preparation, not a completed benchmark. A later direct human reply authorizes US$1 total for the bounded hosted comparison; see API_BUDGET_AUTHORIZATION.json. No new model training or dataset iteration is proposed.

## Execution location

Run the common supervisor on the remote GPU PC. V8 inference uses that PC's CUDA GPU; the same supervisor sends hosted OpenAI requests and saves both arms' logs. OpenAI computation does not use the local GPU. Preserve the verified v8 final-step adapter and explicitly pinned stock base. Do not stack adapters or replace the CUDA environment.

## Why a new evaluation is needed

The completed V8-D diagnostic used fpy revision 04d52c015d1e3ecdefe92b87116f209361509b4b. This PC's current fpy main is 9bc2bd18c495514efa79a4371be1d706b9fc3824, with four additional locally modified files. Differences affect the extractor prompt, state reducer, orchestration and provider behavior. The working-tree patch and changed-file hashes are recorded alongside this plan; a normal clone of main alone will not include those local changes.

Historical reports remain evidence for their frozen protocols. They are not automatically erroneous because the application changed. Do not overwrite them, relabel them as current, or replace their scores with the separately supplied slide figures 60.86% and 77.86%. A suspected old error requires its own documented audit.

## Freeze before paid execution

1. Agree and preserve the exact fpy source snapshot, including local fixes. Save commit plus patch hash; use a dedicated remote checkout, preserving both PCs' existing work.
2. Select and record one explicitly pinned hosted model snapshot. Reusing the prior hosted snapshot makes historical interpretation easier; availability and actual request settings must be checked before a run. Record SDK versions, schema, instructions, rendered input hashes and generation settings.
3. Freeze common slot meanings, value decoding, evidence, duplicate-slot rules, statuses, unknown handling and scoring. Audit old labels against the current reducer and annotation policy before inference. Do not silently change labels in response to model failures.
4. Use the inspected 100-conversation V8-D set for development regression and integration checks only. It cannot become an untouched final test by rerunning it. A stronger final paper comparison requires a separately reviewed, frozen evaluation cohort and explicit authorization before any sealed labels are accessed. Do not open existing sealed/private labels while preparing this runner.
5. Enforce the user-approved US$1 total ceiling, with an internal dispatch ceiling of US$0.90 including outstanding reservations. The later authorization supersedes the previous $0 hosted rule for this bounded task only. Record a durable shared budget before any request. Store credentials only in the remote environment or its secret store. Never copy or print keys in prompts, patches, results or Git.

## Two questions to measure separately

### Controlled requirement extraction

Use identical cases, schema, instructions and evaluation policy for v8 and the hosted arm. Evaluate component extraction with supplied correct prior state, plus rollout using each arm's own accepted state. Save the actual raw response and score it before application recovery, fallback or repair. Disclose provider-native schema enforcement and reasoning/output-budget differences rather than claiming identical inference implementations.

### Current application behavior

Add integration checks for the current orchestrator: recovery of a selected answer, retention of other extracted updates, corrections aimed at another slot, confirmation handling, known-target question wording and accumulated state. Credit deterministic recovery to the pipeline, not the model. Fixed-script state checks and genuinely adaptive interview quality are different evaluations; do not mix their denominators. Any additional question-generation calls must be budgeted and logged.

## Execution gates and outputs

- First run a small development pilot covering multi-slot input, temporal roles, unknown answers, corrections and recovery. Its purpose is detecting implementation/contract failures. If fixes are required, version them and freeze a new protocol before the main evaluation.
- Resume from a durable call journal keyed by model/track/case/turn and verify context hashes. Do not hide refusals, capped answers, schema failures or infrastructure errors. Limit concurrency, preserve failures, and count any paid retry toward the cap.
- Report precision, recall, exact non-intent F1, JSON/schema validity, admission, extra/forbidden slots, correction updates, complete state after correction, unknown preservation, latency, token usage and hosted cost. Report denominators, domain/language breakdowns and designed-context interventions separately.
- For any new inferential claim, specify paired clustering and confidence intervals before evaluating the main set. Account for shared synthetic templates; per-turn independence is not justified. Do not retrospectively choose a scoring policy, seed or checkpoint to improve the winner.
- Archive source/code/case hashes, manifests, actual prompts, raw responses, usage, failures and comparisons in NEW run folders. Update the paper only from completed and matched receipts.

## Decision and stopping rule

Use this next study to decide between the fixed v8 adapter and a fixed hosted configuration under the agreed application requirements. Prioritize grounded updates and complete state as well as F1. A small uncertain difference should be reported as inconclusive, with privacy, operational cost and latency treated as deployment trade-offs. No further fine-tuning is planned; any later training would be a separate research decision.

## Preparation status

The local changes and source identities are captured. A current-pipeline evaluation adapter/runner still needs implementation and offline tests; the older pinned runners are not suitable unchanged. Remote execution, paid spending, independent adjudication and final-test access have not occurred.

## Budget-adjusted handoff

The proposed main comparison selects 20 conversations deterministically across the 20 domains, plus at most two separate pilot conversations. Both arms and both tracks must use the same frozen main selection. Completion within US$1 is not promised; budget exhaustion must be recorded honestly. No independent final-test claim is supported by this inspected development subset. See REMOTE_CODEX_HANDOFF.md for execution steps.
