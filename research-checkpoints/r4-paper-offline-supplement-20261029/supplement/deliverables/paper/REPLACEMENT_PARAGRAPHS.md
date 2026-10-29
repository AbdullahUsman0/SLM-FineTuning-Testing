# Suggested replacement paragraphs for the existing paper

These are proposed insertions, not an edited or replacement manuscript. The latest
six-page paper resides on the main PC and was not available here. Preserve its
cohort history and references; apply only after checking that draft.

## Methods: separate three reliability layers

We distinguish structural validity, actionable requirement extraction and application
state reliability. The completed R4/R4.1 comparison uses 20 previously inspected
synthetic development conversations (101 turns), with one conversation per domain,
six weather/twelve economic/two combined domains and 16 English/two Roman Urdu/two
mixed conversations. Both systems use a common augmented instruction/schema
adapter and identical component inputs; rollout uses each system's accepted state,
subject only to declared synthetic interventions. Native Pydantic candidate parsing
is applied once to both systems. The primary exact slot/value/status multiset F1
excludes intent and non-actionable unmentioned/dont_care entries, which are reported
separately. Exact raw scoring precedes normalization and deterministic recovery.

The hosted arm made 202 fresh main calls and 18 separate pilot calls. The local
arm reused 202 main and 18 pilot responses only after verifying identical input
identities and native-parser state trajectories; both arms were rescored under
the same R4 rules. Local decoding was CUDA BF16 greedy generation with thinking
disabled and 1,024 new tokens; hosted Structured Outputs used the pinned
gpt-5-mini-2025-08-07 snapshot, minimal reasoning and 2,048 output tokens. An R4.1
wrapper continuation bounded an unscored static-question length error without
changing extraction inputs or redispatching completed paid calls. Questions were
not fed into subsequent scripted turns. This evaluates configured systems on fixed
scripts, not an unmodified production prompt or adaptive interview.

## Results: validity is not extraction or state success

All 404 main raw scores, inputs and pipeline states reproduce offline. Local exact
actionable F1 is 0.9686274509803922 component and 0.9568627450980393 rollout; hosted
F1 is 0.6953405017921147 and 0.6938053097345133. Hosted responses are JSON/schema/native
admitted on 101/101 calls per track, yet complete final rollout state is correct in
only 1/20 conversations, compared with 9/20 locally. Component final snapshots use
gold prior and must not be interpreted as successful whole conversations. Report
the configured latency descriptively, not as compute/energy efficiency.

## Error analysis: extraction and recovery can disagree

The local arm emits no actionable updates on all 20 rollout unknown turns, but its
application preserves prior state on only 13; the hosted arm preserves state on
15/20. For the hosted arm, 15 no-actionable outputs and 15 preserved states overlap
on 14 turns, so equal totals do not imply identical behavior. In the oracle replay,
all 20 post-pipeline mismatch calls have correct pre-recovery reducer states despite
gold current extraction and gold prior. Recorded recovery affects output_granularity
in 14 calls, source_reference in four, and target_unit/source_mode in one each.
These trace-backed application failures must not be assigned to raw model extraction.

## Coverage and exploratory uncertainty

Gold contains 257 actionable non-intent updates per track, covering 39 of 78
non-intent slot types. The supplement reports all 78 slots, unsupported-slot and
unknown-ID false positives, per-conversation counts, language/domain denominators
and declared-intervention sensitivity. Supported-slot macro F1 uses the fixed
39-slot positive-support universe and does not replace the primary inclusive
micro metric. The subset does not establish quality across every schema slot.

The post-hoc supplementary plan was saved before calculating new intervals.
Exploratory paired bootstrap resamples whole conversations and recomputes micro
F1 from summed counts, with a paired script-family sensitivity. Distinct entity
IDs are not evidence of independence: 18/20 cases share one template variant.
Intervals are conditional on inspected, related synthetic cases and do not establish
population superiority, equivalence or deployment readiness. No p-values are claimed.

## Scope and remaining validation

Independent human annotation remains incomplete. Gold-first and blinded paired
review forms are prepared, with decisions blank and a coordinator-only identity
mapping; preparation is not annotation. Existing A-F evidence spans different
models, cohorts and policies, rather than a completed two-model-by-six-method
factorial study. Historical model-size runs are not a controlled size experiment.
An independently authored, reviewed prospective cohort and any future inference
need separate approval. These findings support the paper's separation of structure,
actionable extraction and application-state reliability, with bounded development
claims and explicit failure mechanisms.


| System | Track | JSON | Schema / native | Precision | Recall | Exact actionable F1 | Exact pipeline turns |
|---|---|---:|---:|---:|---:|---:|---:|
| v8 | component | 101/101 | 100/101 / 100/101 | 0.9763 | 0.9611 | 0.9686 | 75/101 |
| v8 | rollout | 101/101 | 100/101 / 100/101 | 0.9644 | 0.9494 | 0.9569 | 52/101 |
| openai | component | 101/101 | 101/101 / 101/101 | 0.6445 | 0.7549 | 0.6953 | 54/101 |
| openai | rollout | 101/101 | 101/101 / 101/101 | 0.6364 | 0.7626 | 0.6938 | 6/101 |
