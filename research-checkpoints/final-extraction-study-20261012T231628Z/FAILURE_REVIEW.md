# Verified failure review of existing public results

No new final examples were scored. The original policies, raw outputs, scores and intervals remain unchanged. Findings come from the historical pilot ZIP with SHA-256 `a253dc06fe165cbb2c516903e3cd928344ed010407747f51a73a9b494a74bb7a`; its CRC was verified. All six exact metric aggregates were reproduced from stored records under original scoring.

## Slot-role errors and unwanted emissions

Rollout `expected_history_length` extra flags increased from **11 to 28**. Component flags increased from **15 to 30**. These count slot IDs emitted outside the current turn's gold slot set, including readable rejected/repeated emissions; they should not all be described as accepted invented facts.

Examples in [pilot-offline-diagnostics.json](evidence/pilot-offline-diagnostics.json) reproduce sampling being assigned to historical coverage and refresh being assigned to source access mode. For example, “Each observed record covers 1 day” was used as evidence for `expected_history_length`, although it describes `frequency`. “Source ingestion repeats every 5 hours” was assigned to `source_mode`, although it describes `refresh_frequency`. [The ambiguity diagnostic](evidence/ambiguous-cadence-diagnostic.json) separately records cadence emissions on initial turns explicitly leaving sampling versus forecast issuance unresolved, including rejected emissions with invalid statuses. Role errors and unwanted emissions must be interpreted together with admission status.

Mixed-language rollout F1 regressed from 0.860177 to 0.825342; dense lifecycle F1 regressed from 0.887902 to 0.807175. These subgroup results are exploratory and have different denominators. Aggregate improvement does not guarantee robustness on dense or mixed-language turns.

## Exact evidence mismatches

In v7 rollout, **40 admitted calls contained 77 updates with nonmatching evidence**. Of these, 76 matched after whitespace collapsing. The same aggregate counts occurred in component. V6 had 159 such calls/updates per track, all matching after whitespace collapsing. These were admitted by the historical structural policy; this diagnosis does not retroactively reject them or normalize their primary score.

The prospective policy instead requires a nonblank exact substring of the current message, including original case and whitespace. A mismatch rejects the entire call. This gate is shared by all providers and manual chat. It has not yet been evaluated on the new candidate.

## Three v7 invalid JSON responses per track

| Track | Original scenario suffix | Turn | Language | Failure |
|---|---|---:|---|---|
| Component | corporate_finance/0186 | 3 | Roman Urdu | Missing final outer closing brace |
| Component | weather_agriculture/0076 | 4 | Mixed | Missing final outer closing brace |
| Component | precipitation/0076 | 3 | English | Nested output/repetition; 1,024-token limit reached |
| Rollout | corporate_finance/0186 | 3 | Roman Urdu | Missing final outer closing brace |
| Rollout | corporate_finance/0186 | 4 | Roman Urdu | Missing final outer closing brace |
| Rollout | precipitation/0076 | 3 | English | Nested output/repetition; 1,024-token limit reached |

This corrects the supplied shorthand “two Roman Urdu failures per track”: it is accurate for rollout; component has one Roman Urdu and one mixed-language missing-brace failure. [v7-failure-reproduction.json](evidence/v7-failure-reproduction.json) preserves the original raw strings and per-call identifiers. There is no JSON repair in the primary pipeline.

## State handling is harder than recognizing a correction

V7 correctly identified the correction flag and exact replacement tuples in 230/240 rollout correction calls. Complete accumulated state after those corrections was correct in only 115/240. V6's corresponding counts were 212/240 and 0/240. A correct local replacement does not fix errors already accumulated elsewhere.

V7 emitted no updates on all 200 valid unknown-information turns, but prior-state retention succeeded in 199/200. Empty updates alone do not establish complete preservation: top-level intent and intent metadata can still change through the pinned reducer. The prospective scorer reports semantic state and raw metadata preservation separately. A rejection preserves prior slots and appends the real user history without inventing values.

## Readiness checks are not quality claims

The separate six-fixture CUDA smoke per adapter verified active LoRA loading, pinned base revision, no CPU offload and shared precision. It retained failures without retry/repair. Its small convenience fixture set is not a representative estimate and is not mixed into either historical table or a new final score. One implementation-only failed launch occurred before inference because an adapter path was passed as a string; the verified loader now passes `Path`. No model outcome selected another checkpoint.

Recommended present action: use existing results with explicit limitations; retain v6 and v7 as experimental candidates. Do not promote v7 merely because training loss or aggregate F1 improved, and do not initiate more training while the matched comparison remains unresolved.
