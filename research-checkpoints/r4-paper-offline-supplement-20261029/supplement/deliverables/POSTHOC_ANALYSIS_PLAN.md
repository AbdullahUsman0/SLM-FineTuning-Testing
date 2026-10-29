# POST-HOC supplementary analysis plan

Saved before calculating new intervals. This is not a preregistration: the R4
point estimates and broad failure patterns have already been inspected.

The sealed R4 ZIP, prompts, labels, native parser/scorer, model outputs and
published summaries remain immutable. No inference, paid APIs, training, sealed
labels or shared-ledger access/mutation. All executable analysis strips process
credentials and blocks outbound networking. Tests use only temporary ledgers.

## Fixed baseline and endpoints

Verify the original ZIP SHA/CRC/member hashes. Restore byte-identical application
sources at 9bc2bd18c495514efa79a4371be1d706b9fc3824 plus the four bundled fixes
(including two modified test files), evaluator and R4.1 renderer wrapper. Run the
299-test suite and reproduce all 404 scores, recorded inputs and application
states using saved calls. Environment/path/timezone fixes must not change labels,
application behavior or original evidence. Retain every attempt's logs.

Retain exact original actionable micro precision/recall/F1 by model and track.
Coverage uses the 78 non-intent schema slots, identifying the 39 supported slots.
Supported-slot macro F1 is the unweighted mean across those 39 fixed gold-positive
slots; also report all predictions/FP on the other 39 slots separately. Do not
silently discard unsupported-slot FP from the primary micro score.

## Paired exploratory intervals

Use 10,000 resamples, NumPy PCG64 seed 20261028, percentile 2.5/97.5 intervals.
Resample complete PAIRED conversations, keeping every turn and both model counts
together; recompute micro F1 from summed TP/FP/FN. Report v8, OpenAI and paired
v8-minus-OpenAI differences separately for component and rollout. No turn-level
independence assumption, p-values, significance or population-superiority claim.

The metadata audit found 20 distinct entity/cluster IDs, but 18/20 conversations
share template_variant=4 and two focused cases lack that field. Distinct entity
IDs do not establish independence. Primary intervals are conditional paired-
conversation intervals on these inspected cases, not generalization estimates.

As a prespecified POST-HOC dependence sensitivity, group by
(source_corpus_version or corpus_version, category, template_variant), using an
explicit 'not_recorded' value for missing fields. Resample whole such script
families as paired blocks with replacement and recompute micro F1. Compute this
sensitivity only if there are at least four groups; otherwise report insufficient
groups. Shared generator structure can remain across groups. Grouping is fixed
from metadata, not chosen by observed interval width. No interval validates
independence, equivalence, a universal winner or the population effect.

## Other descriptive analyses

Report per-conversation, domain/language and family counts with denominators.
Reproduce the published exclusion of category explicit_state_repair, and also
exclude entire conversations containing nonempty declared intervention mutations.
Report intervention turns separately. These are subsets of recorded trajectories,
not counterfactual rollouts without interventions.

Taxonomy distinguishes operational misses, extra actionable slots, exact-value
and status discrepancies, candidate cross-slot/temporal-role swaps, unsupported
evidence, duplicate/structural failures, reducer rejection/normalization, recovery
changes and inherited rollout errors. Taxonomy categories may overlap; always
report unique calls separately from individual updates/slot transitions. Exact-
value discrepancies are not automatically semantic errors. Cross-slot matches
are review candidates, not new gold labels.

Reproduce the 20 main oracle component mismatches with gold extraction and trace
whether differences exist before recovery or are introduced after it. Preserve
the unknown-turn result: v8 no actionable updates 20/20, preserved state 13/20;
OpenAI preserved state 15/20. Choose six trace-backed worked examples using fixed
criteria: correct extraction harmed by recovery; v8 unknown-state corruption;
hosted unknown/no-op behavior; exact extraction discrepancy; inherited rollout
error; structural/renderer failure. Include both systems and explicit IDs.

## Human review and future protocol

Prepare blank gold-first forms for all 101 turns, then separate A/B response forms
with conversation-stable randomized mapping (seed 20261028), no reviewer-facing
model identity/score/latency. Keep the mapping outside the reviewer-only kit.
Distinguish supplied component priors from each response's actual rollout prior.
Identity/decision/correction/rationale fields remain blank. An importer validates
forms and exports PROPOSALS only. No automatic label application or claim of
independent human annotation. Prepare a second-reviewer/adjudication form.

The future independent-evaluation protocol and collection sheets are blank.
No participant contact, agent-authored independent data, request generation,
model calls or enrollment. Human/ethics/inference/budget decisions remain pending.

## Reporting

Audit A-F scope from existing public receipts; do not turn a slot-wise one-case
diagnostic, configured XGrammar study or historical model-size runs into a full
two-model-by-six-method factorial study. Main manuscript source is absent here:
return replacement prose/tables and a six-page main/supplement allocation only.
Package verified source evidence, new scripts, logs and deliverables with a
credential check and internal SHA manifest. Do not commit, push or publish.
