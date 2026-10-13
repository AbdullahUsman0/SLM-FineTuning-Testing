# Annotation rubric: fresh requirement-extraction evaluation

This is an evaluation candidate, never a training dataset. Agent drafting, structural validation and independent human adjudication are distinct. A pending row is not approval. Review every conversation and its labels, including whether utterances sound natural in the stated language. Cases use synthetic entities.

Retain the full pinned 79-slot schema. Extract current-turn requirements only. Slot IDs, values and statuses are scored exactly, excluding the intent slot from primary F1. Confidence is required but is not a gold matching endpoint. Gold candidate values are JSON-encoded strings; parser decoding occurs once.

## Target phrase boundaries

Use the exact, contiguous meaningful target noun phrase for both problem_statement and target_description. Include measurement, adjustment, geography or asset modifiers when they are part of the stated target phrase. Exclude request verbs, politeness, project preambles, source clauses and horizon clauses. Preserve spelling, whitespace and language; do not paraphrase.

- Accepted: “overnight air temperature” in “Please forecast overnight air temperature for six days.”
- Rejected: “Please forecast overnight air temperature for six days” (verb and horizon included).
- Rejected: “temperature” (meaningful metric modifier omitted).
- Accepted: “dividend-adjusted closing price of the synthetic LUNO share”.
- A bare column identifier such as “2026” gives target_column only, not a domain target description.

## Filename and explicit format

A clear supported .csv/.json/.parquet/.xlsx extension supplies file_format=inferred, case insensitive, ignoring URL queries. A declared format supplies provided, including an upgrade from inferred to provided. A filename supplies no source_mode, columns, contents, privacy or authentication. Explicit upload/API/database/catalog words may independently supply access mode. Unknown or extensionless names supply no format.

## New values, corrections and confirmations

New and replacement values use provided. A replacement of an existing value has correction_detected=true. Explicit affirmation of an already recorded value emits only the affirmed slot with confirmed; omit unrelated unchanged fields. An inferred-to-provided format declaration is a status upgrade, not a replacement of a different value. The pinned application reducer may store extraction-level confirmed as provided; extraction labels and stored application state are different measurements.

## Time roles

Observation spacing -> frequency; historical coverage -> expected_history_length; forecast issue schedule -> prediction_frequency; input refresh -> refresh_frequency; model refitting -> retraining_frequency. Durations use periods and singular unit. Trading/calendar semantics are only added to calendar_type when stated. Do not infer business_days_only merely from trading days. An unresolved “daily” supplies none of the competing cadence roles. Do not use an observation interval as history length.

## Unknown information and intent

Unknown information supplies no slot value, default, dont_care or invented requirement. A follow-up retains established top-level intent unless explicitly changed. If prior intent is ambiguous and an unknown turn supplies no forecasting goal, top-level intent remains ambiguous. No explicit intent slot update is required for either case. The pinned reducer can update intent-slot provenance even on an empty-update create_forecast turn; report semantic preservation and raw metadata preservation separately.

## Lists, identifiers and language

Preserve array ordering and nested keys. A quoted identifier that looks like 2026, true, null or JSON remains a string after one decoding pass. Equivalent English/Roman Urdu/mixed meanings follow the same semantic rule; exact free-text wording may differ across languages. Technical identifiers in Roman Urdu do not by themselves make the whole utterance mixed language. Independent reviewers must assess this distinction and translation fidelity.

## Evidence and review decisions

Evidence must be a nonblank exact, case-sensitive, whitespace-sensitive substring of the current message. Whole-current-message evidence is structurally acceptable but should be reviewed for whether it grounds the claimed requirement. Prior messages or rewritten evidence are rejected in the primary study. Whitespace-collapsed matching is diagnostic only.

For each review-sheet row set decision to approve or revise; supply reviewer identity, UTC timestamp and comments. Revisions require a replacement-label file and a new candidate hash, followed by revalidation and renewed review binding. Resolve every ambiguity and overlap decision; do not claim independent adjudication if decisions came only from the authoring agent. Label uncertainty should be resolved before the final freeze, never after viewing model outcomes.
