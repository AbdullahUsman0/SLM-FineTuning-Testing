# Deferred prospective experiment

This protocol is prepared for a possible later study. It is **not registered for final scoring** and carries no current authorization for paid calls. The latest user chose previous results for now. `EXECUTION_DECISION.json` takes precedence over the earlier unapproved `budget-request.json`. Missing execution decisions also fail closed.

## Fixed comparison and gates

Primary comparison: frozen experimental v7 r8 pilot versus `gpt-5-mini-2025-08-07`; secondary comparisons: v6 versus OpenAI and v7 versus v6. Base model: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. Application dependency: pinned fpy `04d52c015d1e3ecdefe92b87116f209361509b4b`. Adapter identities and provenance are in `models-freeze.json`.

Before any final-cohort scoring, require a new explicit scope/budget decision, independent human decisions bound to every candidate conversation, ambiguity adjudication, overlap review, matching rubric/cohort hashes, a real budget-approval receipt, complete source/prompt/policy locks, GPU readiness, and registration. Review is not inferred from a general “approved” message about an earlier dataset. Agent checks do not count as independent human annotation.

The execution switch may be changed to `review_and_budget_gates_enabled` only after a new explicit decision is recorded. That switch alone cannot bypass the independent-review and positive-budget requirements. Do not enable it merely to run local final-cohort arms while paid work remains deferred. A smaller or development-only study requires a separately identified protocol and authorization, not silent changes to this cohort.

## Cohort and dependence

Candidate 2 comprises 240 conversations, 1,180 user turns, 4–6 turns each. Each of weather, inflation, stocks and crypto has 60 conversations: 48 English, 6 Roman Urdu and 6 mixed. This intentionally balanced evaluation distribution differs from historical training. No seeded-state corruption is included in the prospective primary cohort.

It is agent-authored synthetic material with twelve shared macro-template groups, related variants/translations across domains, and 81 repeated normalized turns. The first candidate had excessive repeated phrasing and remains preserved privately; the revised candidate was not selected by model outcomes. Exact/normalized overlap was audited against 34,358 accessible training/development utterances, with no matches. Near-overlap thresholds were token Jaccard at least 0.30 or masked character similarity at least 0.70. No sealed final labels were accessed. Lexical screening cannot certify semantic independence; generic clauses shared between template groups remain a limitation.

Review materials and the deterministic generator are private. Structural validation has passed. Independent review and the final freeze have not occurred. The candidate SHA is `7de52c1c326d1ba1d0b09b8445adf5a48b9894edcb1e922f40ef207821bb09b9`.

## Shared policy and measures

Policy `fresh-exact-evidence-pydantic-once-20261012-1` composes the unchanged historical structural parser with nonblank exact current-message evidence validation and the pinned state reducer. One decoding pass preserves explicitly encoded literal-looking strings. It rejects duplicate keys, nonfinite values, incomplete/limited outputs, refusals, duplicate updates, invalid enums and required/closed-field violations. No repair, defaults, gold-conditioned correction or semantic retries are used.

All arms and optional manual chat use the same admission implementation and full 79-slot schema. Annotation boundaries and status rules are in [the rubric](policy/ANNOTATION_RUBRIC.md). Filename-derived format is inferred; explicitly declared format is provided. Corrections replace supplied values; confirmations affirm already recorded values. Ambiguous cadence is not assigned a guessed role. Unknown information adds no default or `dont_care`; established intent remains established, and unknown-before-goal retains ambiguous intent. State semantics and raw metadata preservation are reported separately.

Primary endpoint: rollout exact non-intent slot/value/status micro F1. Component gives all arms the same gold-replayed prior state. Rollout supplies fixed messages while each arm accumulates its own accepted state. Secondary measures cover slot-name F1, precision/recall, JSON/schema validity, structural and evidence admission, extra/duplicate emissions, correction replacements, unrelated state, complete state/conversations, unknown behavior, original generation latency, tokens and hosted cost. Numeric-canonical and status-free scores remain diagnostics.

Rejected readable tuples count as false positives, gold tuples as false negatives, and rejected calls receive zero true positives. Unreadable failures remain in denominators and are not certified safe. Independent verification replays raw receipts and every state transition, checks complete call inventory, gold/context pairing, hashes and spending reconciliation before publishing results.

Paired uncertainty: 10,000 resamples, seed 42, resampling twelve source/template groups and recomputing micro F1. The primary difference is v7 minus OpenAI; the interval is two-sided. No post-hoc equivalence claim. Domain/language/template findings are exploratory. The effective dependence is not 240 independent conversations.

## Runtime, schedule and spending

Both local arms use CUDA, greedy batch size 1, seed 42, thinking disabled, max 1,024 new tokens, BF16 base weights and PEFT default FP32 LoRA parameters. There is no CPU offload. The grouped schedule alternates which local arm runs first; hosted requests follow each matched local group. Models load serially. Warmups are recorded separately and excluded from quality/timing aggregates. Local and hosted timings are descriptive across different hardware and transports.

The planned full study would contain 7,080 scored calls: 1,180 turns × two tracks × three arms, including 2,360 hosted calls. Hosted transport uses Responses, strict JSON Schema, minimal reasoning and at most 2,048 output tokens including reasoning. The official model catalog currently marks the pinned snapshot deprecated; availability must be checked before any authorized paid request. No automatic model substitution is allowed. [Model catalog](https://developers.openai.com/api/docs/models/gpt-5-mini), [structured-output guide](https://developers.openai.com/api/docs/guides/structured-outputs), [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning).

The earlier estimate was a proposal, not a charge or approval. Current authorized new paid budget is **$0**. If a future study is authorized, the ledger reserves a conservative worst-case amount before every attempt, reconciles recorded usage including cached input and reasoning output, and retains exposure for ambiguous transport failures. SDK automatic retries are disabled. Only transient transport/429/5xx errors can receive one bounded retry; semantic errors cannot. Requests above the frozen 32,768-byte input cap stop without truncation. A resumed call must match run, model, prompt, context and gold identities. Paid execution and snapshot availability have not been validated for this new study.

Stopping rule: complete one registered matched experiment and independent verification, then report findings. Do not train again, expand the cohort in response to outcomes or choose another checkpoint after viewing final results.
