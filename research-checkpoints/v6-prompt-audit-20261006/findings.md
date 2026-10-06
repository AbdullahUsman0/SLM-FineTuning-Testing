# Findings from the v6 label audit and frozen prompt comparison

Both requested steps are complete. All 32 natural conversations and their 96 labels were reviewed by the Codex agent. No clear factual label error was found and no labels were changed. Independent human review remains pending. One combined prompt revision was frozen before new inference, then compared with the original prompt in 384 matched serial BF16 CUDA calls on unchanged v6 adapter weights. Both gold-context component and own-state rollout tracks used the same fixed user scripts.

The original prompt repeated all 192 prior raw responses exactly, with identical gold labels and contexts. All 384 new calls and all eleven published source artifacts were independently checked against their Git blobs and SHA-256 values. See [independent-verification.json](independent-verification.json). The adapter model and configuration hashes also remained unchanged after inference. Four complete lossless reports are under [results](results/README.md), alongside source/runtime pins, per-domain metrics and paired analyses. The frozen prompt text and protocol are under [evaluation/v6-prompt-audit-20261006](../../evaluation/v6-prompt-audit-20261006/README.md).

## Label and failure audit

The original v6 rollout has 49 expected slot updates absent from readable emissions, primarily units, observation frequency and initial horizon. Eleven other gold slots are unassessable because their entire output fails strict JSON parsing. The earlier coarse count of 60 combines these different failure types. One output repeats a target slot. These are failures of structured extraction; absence from a correct slot does not necessarily mean that the model never mentioned the fact anywhere in its response.

Twenty of the 21 differing readable target descriptions retain the complete expected target while appending units or timing requirements. Nineteen of these omit the separate unit update. This identifies separation of facts into their intended fields as a major weakness. The remaining target error returns `basis points` as the target instead of `futures-minus-spot basis`. See [audit-findings.md](audit-findings.md) and [target-description-review.json](target-description-review.json).

Existing application normalization changes original rollout F1 from 0.550 to 0.604. This accepts existing enum capitalization and duration forms, preserves the schema-validity gate and status, and never repairs outputs or changes primary scores. All 32 corrected horizons are right after the common reducer and other prior slots are preserved. Only 3/32 full accumulated states match gold after correction because initial requirements remain missing or wrong. Unknown-information handling fails in four cases sharing one authored wording pattern, with twelve unsupported slot emissions; one further initial emission uses the wrong refresh slot.

## Original versus revised prompt

The candidate appends a completeness/encoding checklist and one generic water-demand illustration to the existing prompt. It asks for separate target/unit/frequency/horizon/refresh updates, exact allowed fields and enum values, one update per slot, JSON-text candidate values, and abstention for unknown information. This was one combined candidate, without further tuning during the run. The experiment cannot isolate which instruction or example causes each effect.

Own-state rollout results, 96 calls per prompt:

| Measure | Original | Revised |
| --- | ---: | ---: |
| Strict slot/value/status F1 | 0.550 | 0.602 |
| Existing application-normalized F1 | 0.604 | 0.670 |
| Slot-name F1, schema-gated | 0.766 | 0.762 |
| Readable-content value F1, descriptive only | 0.564 | 0.716 |
| Strict JSON validity | 94/96 | 94/96 |
| Exact wire-schema validity | 93/96 | 87/96 |
| Observed new unmentioned slot emissions | 13 | 0 |
| Expected updates absent from readable emissions | 49 | 24 |
| Correct correction flag and exact raw typed value | 26/32 | 24/32 |
| Corrected horizon after the common reducer | 32/32 | 32/32 |
| Entire gold state correct after correction | 3/32 | 8/32 |
| Valid unknown turn, no updates, prior state retained | 28/32 | 32/32 |
| Mean extraction latency | 6.604 s | 7.102 s |
| Median extraction latency | 5.300 s | 5.489 s |
| p95 extraction latency | 13.005 s | 15.179 s |

No call reaches the 1,024-token generation limit. The revised prompt cuts readable missing unit updates from 22 to 11, observation-frequency updates from 11 to 5, and horizon updates from 6 to 1. It emits no observed unmentioned slot updates on this cohort. Both prompts still have two strict-JSON failures per track. The revision introduces six outputs with an unsupported update-level `type` field and one that puts unit/frequency/horizon keys inside a single target-description update. These account for its additional schema failures; readable content is not repaired or applied to rescue the operational score. See [contract-failure-review.json](contract-failure-review.json) and [comparison-refined-audit/content-assessment.json](comparison-refined-audit/content-assessment.json).

Domain rollout F1:

| Domain | Original | Revised |
| --- | ---: | ---: |
| Weather | 0.622 | 0.682 |
| Inflation | 0.593 | 0.449 |
| Stocks | 0.420 | 0.769 |
| Crypto | 0.558 | 0.494 |

Gold-context component F1 changes from 0.533 to 0.579 and normalized F1 from 0.604 to 0.670. Its domain direction and schema/unknown tradeoffs agree with rollout. Thus the candidate helps stocks, weather, completeness and abstention, while regressing on inflation, crypto and contract validity. Overall strict rollout F1 rises by 0.051; its paired 10,000-resample conversation-bootstrap 95% interval is [-0.089, 0.189], which spans zero. The corresponding component interval also spans zero. This is insufficient evidence of a reliable overall improvement even within these narrow development cases.

## Decision and limits

Keep the original prompt as the current experimental baseline. The revised candidate is useful evidence about completeness and uncertainty handling, but its domain regressions and increased contract failures prevent recommending it as a general replacement. A focused format-control experiment is a sensible next research step before another training round. New independently reviewed conversations will be needed to establish that any later prompt or dataset change generalizes.

These 32 agent-authored cases were inspected to design the prompt; they are development cases, not an untouched test set. They reuse eight correction and eight uncertainty wording patterns across four domains. Bootstrap intervals describe resampling this cohort and do not cover annotation bias, prompt-selection uncertainty or unseen users. Unknown turns measure abstention/state retention, not generated clarification-question quality. Free-text semantic review remains agent assessment; no human-equivalence score is substituted for the original strict metric. The refined content/contract audits are post hoc descriptions that do not alter gold labels, model outputs, state updates or primary scores.

Fourteen offline regression tests passed before the frozen run. No new training, output repair, inference retry, paid API call or sealed final-label access was used. The default production/manual-chat prompt was not changed by this experiment.
