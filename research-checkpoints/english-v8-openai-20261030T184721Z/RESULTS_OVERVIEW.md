# Completed prospective English comparison

All 820 fresh main calls completed: 205 component and 205 rollout responses
per model, on 44 English synthetic conversations. All 820 saved raw scores,
inputs and application states were replayed; all 205 component input pairs match.
No transport retries were used. The prepared runner's 319 offline tests passed.

These are exact actionable slot/value/status scores for this NEW condition.
Historical R4 used a different cohort/prompt/annotation condition and remains
separate. Do not pool them or infer a controlled before/after difference.

| Model | Track | TP | FP | FN | Precision | Recall | Exact actionable F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| v8 | Component | 94 | 304 | 483 | 23.62% | 16.29% | 19.28% |
| OpenAI | Component | 326 | 542 | 251 | 37.56% | 56.50% | 45.12% |
| v8 | Rollout (primary) | 83 | 316 | 494 | 20.80% | 14.38% | 17.01% |
| OpenAI | Rollout (primary) | 314 | 572 | 263 | 35.44% | 54.42% | 42.93% |

The gold update denominator is 577 per track, not 205 slots. F1 is not generic
accuracy. OpenAI scores higher on this bounded exact endpoint; both systems
have poor exact application-state performance under the frozen ideal contract.
Do not claim universal superiority or deployment readiness.

| Model | Track | JSON valid | Schema/native admitted | Exact final state | Median response seconds |
|---|---|---:|---:|---:|---:|
| v8 | Component | 204/205 | 203/205 | 0/44 | 9.76 |
| OpenAI | Component | 205/205 | 205/205 | 2/44 | 4.17 |
| v8 | Rollout | 204/205 | 203/205 | 0/44 | 11.51 |
| OpenAI | Rollout | 205/205 | 205/205 | 0/44 | 4.01 |

Component final snapshots use gold prior; they are not whole-dialogue successes.
Application-state results include deterministic recovery and confirmation-contract
differences already documented before inference. Raw extraction is separate.

Conversation-cluster paired percentile interval for v8-minus-OpenAI F1 is
component -25.84 percentage points [-29.74, -22.17] and rollout -25.92
[-29.40, -22.46]. The fixed 10,000 resamples/seed20261030 and separate nine-family
sensitivity are stored in the archive. Related synthetic scripts and a single
generator limit these conditional intervals; no population/power claim follows.

The separately planned POSTHOC_NUMERIC_REPRESENTATION_AUDIT.json changes only
numeric JSON equivalence, not primary scoring. It leaves both v8 scores and
OpenAI component unchanged; hosted rollout sensitivity is 44.98% (15 additional
TP). It does not explain the low v8 scores. Keep this explicitly post-hoc.

Human-review completion is USER-ATTESTED in the approval receipt. Reviewer
identities, completed per-turn worksheets and agreement statistics were not
returned to this server; do not invent them. All original AI drafts and changes
are preserved. The raw responses, labels, protocol, code, states and usage are
in the verified archive. See LOCAL_CODEX_HANDOFF.md before editing the paper.

Operational financial records must remain outside manuscript text and tables,
as instructed by the author. No new inference, relabelling or retraining is
needed to reproduce these saved results.
