# Audit of the 32 natural forecasting conversations

All 32 scenarios (eight per domain) and all 96 annotated turns were reviewed by the Codex agent. No clear factual label error was found and no labels were changed. This is an agent review after inspecting model outputs; independent human review remains pending. The original cases are now development data, not an untouched test set.

The annotations distinguish target descriptions from units, observation frequency from forecast horizon and refresh frequency, explicit target/time columns, and explicit upload/file-format requirements. The 32 horizon corrections replace the original horizon; unknown-information turns provide no new values. Review notes for every scenario are in [label-review.json](../../evaluation/v6-prompt-audit-20261006/label-review.json).

For the original v6 own-state rollout:

| Finding | Count or score | Interpretation |
| --- | ---: | --- |
| Original strict slot/value/status F1 | 0.550 | Unchanged schema-gated operational score |
| Supplementary application-normalized F1 | 0.604 | Existing enum/duration normalization only; no repairs or synonym matching |
| Expected slots absent from readable emissions | 49 | All occur in initial turns; primarily units, observation spacing and horizon |
| Gold slot contents unassessable because strict JSON failed | 11 | Across two initial outputs; do not call all of these literal model omissions |
| Duplicate slot emission | 1 | One initial output repeats the target-description slot; invalid wire contract |
| Other emitted values/types differing from gold | 27 | Includes fused or shortened target descriptions; not independently adjudicated semantic errors |
| Normalization-only differences | 9 | Six corrected horizons and three initial enum values |
| New unmentioned slot emissions | 13 | Twelve in four unknown turns, plus one incorrect refresh slot in an initial turn |
| Corrected horizon after the common application reducer | 32 / 32 | All corrections work locally after normalizing accepted duration forms |
| Corrected horizon with other prior slots preserved | 32 / 32 | Does not imply that earlier initial requirements were complete |
| Entire accumulated state correct after correction | 3 / 32 | Initial errors persist even when the corrected horizon is right |
| Valid unknown turn with no updates and prior-state retention | 28 / 32 | Four unknown turns invent values |

The earlier coarse count of 60 absent slots includes the 11 gold slots whose outputs could not be assessed because of duplicate JSON keys. The refined audit separates these failure types. In readable output, the missing slots are target_unit (22), frequency (11), forecast_horizon (6), source_mode (6), file_format (2), and prediction_frequency (2). Raw content in a schema-invalid response remains unusable by the application even if some facts are readable.

Twenty-one of the 27 differing initial values concern target descriptions. Agent review of every one found that twenty retain the complete expected target while appending units or timing requirements; nineteen of those omit the separate unit update. An example is `digital-asset market capitalization in millions of USD`, with no separate `millions of USD` unit. The remaining case, crypto/06, returns `basis points` as the target instead of `futures-minus-spot basis`. This points to failure to separate requirements into their intended fields. The full expected target being present within a longer description does not supply a missing structured unit. Review details are in [target-description-review.json](target-description-review.json). This semantic inspection remains agent review, and no scores or labels were changed.

All twelve unsupported emissions in unknown turns occur with the same authored utterance pattern: asking for questions about timezone, seasonal periods and prediction interval levels because their values are unknown. These four cases also have different domains. They reveal a concrete vulnerability to this wording; they do not establish its prevalence among real users. Three unknown slots are supplied in each case, including invented interval levels and seasonality values. The remaining unexpected emission is `refresh_frequency` where the stated requirement is a forecast refresh interval (`prediction_frequency`).

The original strict label/metric definitions are preserved. Supplementary normalized scoring applies the existing pinned `normalize_value` function to both gold and observed values and keeps the schema-validity gate and status. The refined content assessment is a post hoc diagnostic added during the frozen prompt experiment; it never repairs outputs, updates dialogue state or changes F1. Its source report hashes and code hash are in [baseline-refined-audit/content-assessment.json](baseline-refined-audit/content-assessment.json).

The 32 cases reuse eight correction and eight uncertainty wording patterns across domains, and the unknown turns test abstention/state retention rather than generated clarification questions. Results are narrow development diagnostics. No training, paid API calls or sealed final-label access occurred. One combined prompt revision was frozen for a 384-call matched comparison; no changes to that candidate prompt are made in response to interim results.
