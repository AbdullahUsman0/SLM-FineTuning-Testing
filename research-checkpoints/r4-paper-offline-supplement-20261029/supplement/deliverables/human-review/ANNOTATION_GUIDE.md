# Human review guide: gold first, then blinded responses

These are previously inspected synthetic DEVELOPMENT conversations. This kit is
prepared, not independently annotated. All reviewer fields are deliberately blank.
Use a pseudonymous reviewer identity. A coordinator must verify human identity,
independence and adjudication; software cannot certify those properties.

## Release order and blinding

1. Give reviewers only `01-gold-first-review-kit.zip`. Review each message,
   supplied prior, intervention and expected update/state before seeing model
   predictions where practical. Record approve/propose_change/unclear and a reason.
2. A second human independently reviews `second_reviewer_gold.csv`. Do not copy
   the first reviewer's decisions into it. Resolve disagreements using the blank
   adjudication form; preserve original and proposed labels separately.
3. Release `02-blinded-response-review-kit.zip` after the gold pass is locked.
   A/B identity is fixed across all turns and tracks within a conversation, but
   may swap across conversations. Only the coordinator has the mapping.
4. Do not distribute the full supplement, source evidence, coordinator directory
   or mapping to blinded reviewers. Raw response styles may still provide clues;
   this is label masking, not proof that model identity cannot be inferred.

Conversation IDs C01-C20 mask original case/version identifiers. Reviewer forms
contain no external model scores or latency. Confidence fields inside raw JSON
are emitted content, not evaluation scores. Do not alter immutable input columns.
The two tracks can have different inputs: component supplies the reference gold
prior; rollout supplies each system's ACTUAL accepted prior after any declared
intervention. Judge against the displayed input, not the other system's history.
Assistant-question context is explicitly absent in this fixed-script study.

## Annotation policy

- Extract facts stated in the CURRENT user message. Prior state and assistant
  suggestions provide context, not new current-message evidence. Omit unchanged
  requirements unless the user explicitly confirms the same value.
- Unknown/missing/undecided information supplies no positive value. Do not guess
  defaults, source access, history length, columns, privacy or authentication.
  Explicit indifference is different from unknown: `dont_care` may express it,
  although the current application reducer ignores that status. Record a policy
  concern rather than silently rewriting it as an unknown value.
- `provided`: explicit current fact. `inferred`: an allowed, clearly evidenced
  inference. `confirmed`: explicit confirmation of the SAME recorded value.
  `ambiguous`: unresolved alternatives. `conflicting`: unresolved contradiction.
  `invalid`: application-invalid candidate. `unmentioned`: no supplied value;
  ordinary missing information should normally be omitted from update gold.
- Explicit replacement of a previous value uses `provided` and
  correction_detected=true. Update only the changed fields. Distinguish a
  correction from an unresolved contradiction and from confirming a value.
- A meaningful described forecasting target supports both problem_statement
  and target_description with the same exact meaningful target phrase. A column
  name alone does not establish domain meaning. Review exact span boundaries
  separately from semantic acceptability; a paraphrase can fail exact matching
  without being a fabricated fact. Do not silently change the frozen metric.
- Cadence roles are distinct: observations -> frequency; forecast issuance ->
  prediction_frequency; source ingestion -> refresh_frequency; model refitting ->
  retraining_frequency. Future coverage -> forecast_horizon; explicit past coverage
  -> expected_history_length. An unresolved cadence-role ambiguity supplies
  neither role until clarified. Refreshing a source does not change source_mode.
- A clear .csv/.json/.parquet/.xlsx filename extension permits file_format with
  status inferred, case-insensitively and ignoring URL query text. An explicit
  format declaration uses provided and can upgrade an inferred value. Do not
  infer anything else from the filename. Unsupported/absent extensions give no
  format. Do not infer file contents or columns.
- Evidence should be a nonempty verbatim CURRENT-message substring, preserving
  wording/whitespace. The application has a separate evidence canonicalizer;
  distinguish its acceptance from literal evidence fidelity.
- Duplicate slot entries are not multiple independent requirements. Identify
  repeated versus conflicting duplicates. The native parser can admit duplicates;
  the raw multiset metric still penalizes surplus actionable predictions.
- Candidate values use the published JSON-text envelope. The native decoder
  decodes once and retains ordinary strings that are not inner JSON. Literal
  column names such as `true` need careful string-versus-boolean review. The
  approved application state for one such column is invalid; propose disagreements
  separately rather than changing the existing raw gold or state automatically.

## What to judge separately

1. JSON/schema and proposed update content.
2. Current actionable extraction: missing/extra slot, exact value/status,
   possible wrong role, unsupported evidence, duplicates.
3. Native reducer rejection/normalization.
4. Recovery-introduced changes and final state, including confirmation flags.

An ignored unknown marker is not necessarily an invented actionable fact. A high
extraction score does not imply a fully correct conversation. A correct extraction
can be harmed by recovery; a recovered value is not model extraction credit.

## Filling and importing forms

Gold decisions: approve, propose_change, unclear. Pair-side decisions: correct,
partly_correct, incorrect, unclear. Pair preference: A, B, tie, unclear. Give a
rationale for each completed decision. Adjudication: retain_original,
propose_change, unresolved, with two reviewer identities and an adjudicator.

For a proposed correction, use a JSON object with one or more of
gold_extraction, expected_state_after or context_state, optionally notes.
gold_extraction uses the original five fields; each update has slot_id,
candidate_value (JSON text), status, confidence and evidence_text. State entries
contain value/status/confirmed_by_user. The importer checks shape, slot names,
status/range/evidence and immutable rows. It exports PROPOSALS, never applies them.

Coordinator command (use a NEW output directory):

    python import_reviews.py --kit <human-review> --template gold_first_review.csv --form <completed.csv> --output <new-proposals-directory>

Use --allow-partial only when intentionally importing a sheet with blank pending
rows. Missing/duplicate/tampered rows or malformed completed decisions are errors.
Even a complete form does not automatically set independent human review complete.
