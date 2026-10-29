# Taxonomy definitions and counting units

These POST-HOC diagnostics do not replace primary scores or adjudicate semantic
gold. Categories overlap and their counts must not be added to obtain FP/FN.
Each row reports unique affected calls out of 101 separately from update/slot
counts. Call-level indicators have a blank individual-update count. Explicit
zero rows distinguish zero observed events from an omitted measurement.

- Missing current slot: unmatched gold updates with no emitted update for that
  slot. A wrong value on an emitted slot is separate from absence of the slot.
- Extra actionable slot: emitted actionable current-turn slot absent from gold.
  This can be a redundant prior value or an ambiguity, not necessarily a fabricated
  committed fact. Ignored status entries are counted separately and are not errors.
- Exact-value/status discrepancy: readable unmatched tuples sharing a gold slot;
  value-boundary, numeric representation and paraphrase differences need human
  semantic assessment. Categories can overlap when both value and status differ.
- Cross-slot/temporal-role candidate: an unmatched emitted value/status matches
  another gold slot, including observation/issuance/refresh/refit/history/horizon
  roles. This is a review candidate, not an automatically relabeled error.
- Evidence: native canonicalizer rejection and nonverbatim text are distinct.
  Admission-denied readable matches quantify structural gating without granting TP.
- Duplicates: surplus non-intent list entries; structural failures are per-call
  JSON/schema/native parsing indicators. A schema-valid unknown registry slot can
  still be rejected by the application. The hosted component `notes` slot is such
  an example; its FP is retained outside the 78 known-slot table.
- Normalization: a candidate changes representation under the frozen normalizer;
  this does not prove every candidate was committed. Reducer rejection is a saved
  per-call outcome, not proof all surviving updates were grounded.
- Recovery: compare the recorded direct-reducer snapshot with the full engine
  result. Explicit recovery attribution requires the saved recovery slot ID;
  other engine changes are listed separately. Help/harm compares with unchanged
  expected slot state. Complete-state harm and raw-extraction-plus-recovery harm
  are call indicators. Traces provide the detailed mechanism.
- Accumulation: actual rollout prior differs from the supplied gold-reference
  prior after the same declared intervention. This is inherited discrepancy,
  not automatically a new current-turn model error or a no-intervention rollout.
- State comparison uses the published all-non-null-slot value/status/confirmation
  definition, including intent, excluding confidence/evidence/timestamps. The
  exact raw tuple metric uses its original canonical serialization; no new
  semantic normalization or label correction is credited.

All per-call details, prior/pre/post/expected states, source keys and six worked
examples are retained. Structural validity, actionable extraction and application
state reliability remain distinct outcomes.
