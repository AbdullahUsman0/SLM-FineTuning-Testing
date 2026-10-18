# Final bounded training experiment

## Research question

Does one full-data, fixed-schedule Qwen3.5-2B LoRA candidate improve extraction and accumulated requirement state on development cases? This package prepares a candidate for a subsequent independently reviewed matched final comparison. It cannot guarantee a winner or establish an OpenAI comparison without that separate experiment.

## Dataset and policy

`corpus-v8/v8-final-20261018-r1` contains the full public v7 r8 development trajectories, unchanged gold, a versioned instruction extension and 1,000 new targeted synthetic conversations. It preserves entity-group separation. It does not include or open sealed final labels or the private prospective 240-conversation evaluation candidate.

The four new behavior families concern temporal-role isolation, source refresh without access-mode changes, unknown answers without defaults and preservation of unrelated state during corrections. Explicit past-history positives counterbalance omitted-history examples. Some API-source conversations add a separate safe `secret://synthetic-...` identifier turn; these strings do not name real credentials. The pinned redactor accepts a complete reference but scrubs one embedded in a longer sentence, so the entire identifier is supplied as its own turn.

Current-turn exact evidence, one candidate-value decode, the full 79-slot schema and the pinned application's reducer apply. Existing provided/inferred/confirmed conventions are preserved. Missing information does not produce a default. Extraction-level confirmation and stored confirmation are different: the pinned reducer can downgrade a model's confirmed update.

Scenario balance in both splits is 30% weather, 60% economics across twelve subdomains and 10% combined requests. Language labels are 80% English, 10% Roman Urdu and 10% mixed at conversation level; these are generator labels, not independent judgments of language quality.

## Training fixed before outcomes

- Stock Qwen/Qwen3.5-2B at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, plus a fresh LoRA. Do not stack or continue a v6/v7 adapter.
- Two epochs, seed 42, learning rate 5e-5; existing trainer's rank 16, alpha 32, dropout 0.05, completion-only loss and effective batch size 16 on one CUDA GPU.
- BF16 base, no CPU/NPU offload. Maximum allowed example length 4,096; actual full tokenization preflight must pass, and truncation is prohibited.
- Complete checkpoint every ten optimizer steps; retain all. 17,860 examples imply 1,117 optimizer updates per epoch and 2,234 in two epochs on the frozen one-GPU setup. Verify the trainer's actual schedule before starting.
- The retained `best-adapter` name is historical trainer terminology. In this configuration it is a fixed final-step adapter, not behaviorally selected or a proven best model.
- No early stopping or extension after looking at outcomes. Additional seeds require distinct predeclared run folders; do not report the best seed as if it were the only run.

## Development diagnostic

After training, the worker runs 100 fixed validation conversations: five per domain, including one new focused English conversation, three base English conversations and one base Roman Urdu or mixed conversation. Across the set this is 80/10/10 languages and 30/60/10 domain families. Selection is deterministic and independent of predictions. This is a deliberately stratified diagnostic, not a representative or untouched final test.

Both component (gold prior) and rollout (own accepted prior) tracks receive the same current messages. Gold context interventions from r8 are explicit synthetic training/development fixtures, applied at declared turns. Report results excluding their category as well. Gold prior slots and answers are never supplied to a rollout extraction.

Generation is greedy, no thinking, repetition penalty 1.0, one call per turn and at most 1,024 generated tokens. There is no output repair, semantic retry or hidden truncation. Malformed, capped, ungrounded or invalid reducer responses remain failures. Exact non-intent slot/value/status micro F1 is calculated after one JSON candidate decode; numeric or synonym rewrites are not performed for scoring. Readable rejected emissions count as false positives, and all gold updates remain false negatives. Confidence is not a matching endpoint. Complete state, precision/recall, structural validity, evidence admission, correction state, unsupported/forbidden emissions, latency and domain/language summaries are reported separately.

Raw generation records are fsynced once, with context/gold hashes. Resume replays and verifies completed records; an incomplete journal tail is preserved before recovery. A generation/infrastructure exception stops execution instead of fabricating a successful call. There is no paid provider in this runner.

The same diagnostic can be run once on v6 using a separate output folder and the exact same new instructions/policy. Such results are new development measurements; they must not be compared numerically with old cohorts or presented as the deferred final comparison. There is no automatic checkpoint/model promotion.

## Remaining work before a final paper comparison

Independent annotation and disagreement adjudication remain pending. The existing paper's small development audit and the separate prospective final cohort are distinct tasks. Stronger final evidence needs human-authored or independently reviewed withheld conversations, source/template dependence accounted for in uncertainty estimates, a locked model/prompt/parser/scorer and a separately agreed hosted-model snapshot and spending cap. The old `$0` hosted-execution decision remains in force.

Stop after the fixed training and diagnostic. Save technical failures honestly. Do not regenerate data, add epochs, change labels, alter parsing or choose a checkpoint after seeing final-test outcomes. Preserve all earlier paper tables and add new results as their own cohort.
