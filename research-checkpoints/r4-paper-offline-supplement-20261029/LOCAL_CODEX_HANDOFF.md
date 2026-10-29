# Paste-ready handoff for local Codex: pull evidence and update the existing paper

Continue our FYP using the completed R4/R4.1 comparison and its offline supplement.
Do the useful paper work and checks; do not merely repeat a plan.

Repository: https://github.com/AbdullahUsman0/SLM-FineTuning-Testing
Branch: experiment/v5-grounded-20260919
Required baseline ancestor: 6c51ee2c32e16d3237fce78523a07e04cdb29667
Current supplement: research-checkpoints/r4-paper-offline-supplement-20261029/

## 1. Pull safely and verify

Preserve all local paper edits, environments, adapters, results and ledgers. Fetch
and fast-forward, or use a separate checkout if existing edits/divergence conflict:

    git fetch origin experiment/v5-grounded-20260919
    git pull --ff-only origin experiment/v5-grounded-20260919

Read THIS root handoff and README first, then PUBLICATION_VERIFICATION.json.
The latter records the immutable evidence commit and download verification.
Verify the adjacent r4-paper-offline-supplement-20261028-r2.zip:

    SHA-256 a09f33e7c36bab427991e0ed30a5916a926e99035d1c661e4cf926425a4ed3f7

Extract into a NEW directory. Check CRC, every PACKAGE_MANIFEST.json member hash
and exact member coverage (manifest plus its listed files). The full package has
246 members. The nested source-evidence/r4-native-comparison-results.zip remains
unchanged at SHA-256 4d4a33bd6ec4ac1bab28b47d37ee324a315177035826119e36a230cd71c143ae;
it has its own ARCHIVE_MANIFEST.json and 529 members. The same material is available
unpacked under supplement/ for browsing.

The internal local-only/unpublished flags are preserved pre-publication receipts.
The user subsequently authorized publishing these completed artifacts on Oct 29.
This does not authorize new paid work, training, label changes or publishing future
review decisions/manuscript revisions without a further user instruction.

## 2. Important results to retain exactly

| Model / track | Precision | Recall | Exact actionable F1 |
|---|---:|---:|---:|
| v8 component | 0.9762845849802372 | 0.9610894941634242 | 0.9686274509803922 |
| OpenAI component | 0.6445182724252492 | 0.754863813229572 | 0.6953405017921147 |
| v8 rollout | 0.9644268774703557 | 0.9494163424124513 | 0.9568627450980393 |
| OpenAI rollout | 0.6363636363636364 | 0.7626459143968871 | 0.6938053097345133 |

- Twenty matched synthetic development conversations, 101 turns/model/track.
- Hosted main 202 fresh calls, plus 18 separate fresh pilot calls. Local main 202
  and pilot 18 responses were reused only after identical-input/trajectory proof.
  Both systems were rescored under common R4 native/actionable rules. Do not call
  both arms fresh or combine a new hosted row with an older SLM scoring condition.
- Complete final rollout state: v8 9/20; OpenAI 1/20. Component final snapshots
  use gold prior and are not successful whole conversations.
- Local no-actionable output on unknown rollout turns: 20/20; prior preserved
  only 13/20. OpenAI preserves prior on 15/20. Raw abstention is not state safety.
- All 20 oracle mismatch calls are correct before recovery and wrong afterward,
  with explicit recovery IDs: output_granularity 14, source_reference four,
  target_unit one and source_mode one. See six trace-backed examples. Do not
  attribute these application changes to model hallucination without the trace.
- Gold support is 39 of 78 non-intent slots; all 78 and unsupported/unknown-ID FP
  are reported. One hosted component response uses unregistered slot notes despite
  passing JSON/schema/native parsing; the application rejects it.
- Original 299 tests and all 404 scores/inputs/states reproduce, including a fresh
  shipped-layout check. Ten additional supplement controls pass. Windows needed a
  shorter pytest temporary path in the deep portable copy; application/test source
  stayed unchanged. tzdata and the two modified bundled test files are documented.

Exploratory paired conversation intervals (10,000 resamples; seed 20261028) for
v8-minus-hosted F1 are component +0.2733 [0.2200, 0.3323] and rollout +0.2631
[0.1901, 0.3338]. A ten-script-family sensitivity is also provided. The plan is
POST-HOC, not preregistered. Eighteen cases share a template variant. These are
conditional estimates on inspected synthetic cases, not population-independence,
significance, equivalence or universal-superiority proof. Keep the original no-CI
receipt unchanged and report new intervals only as a separate supplement.

## 3. Read the working materials

Inside the complete ZIP or unpacked supplement/:

- ANALYSIS_COMPLETION.json and deliverables/README.md.
- deliverables/POSTHOC_ANALYSIS_PLAN.md and BASELINE_REPRODUCTION.json.
- deliverables/analysis/: all-78-slot, conversation, subgroup/intervention tables;
  taxonomy with unique-call versus update/slot counts; oracle traces and six examples.
- deliverables/fairness-and-methods/: fairness audit, A-F coverage and source receipts.
- deliverables/paper/: replacement paragraphs, tables and six-page allocation.
- deliverables/human-review/: blank forms, annotation guide, importer and coordinator key.
- deliverables/future-independent-evaluation/: proposal and blank collection forms only.
- code/, logs/, reproduction/ and the original sealed source evidence.

## 4. Next: human review in the correct order

All reviewer fields are blank. Arrange real independent humans and an adjudicator;
do not fill their decisions as Codex or label agent review independent annotation.
Give only the gold-first reviewer ZIP initially. Lock that pass, obtain a separate
second review/adjudication, then release the A/B-response reviewer ZIP. A/B mapping
is stable per conversation and must be withheld during review. Do not distribute
the full supplement, coordinator folder, source outcomes or model scores.

The complete bundle and mapping are now public, so do not promise enforced blinding.
Record reviewer prior exposure to the repository/results. Procedural masking and
human independence require real coordination, not an automated flag.

Use the importer on completed forms with a NEW output directory. Example:

    python supplement/deliverables/human-review/import_reviews.py --kit supplement/deliverables/human-review --template gold_first_review.csv --form <completed.csv> --output <NEW-proposals-directory>

It checks immutable rows, duplicates, known slots, evidence/status/confidence and
exports PROPOSALS ONLY. It never edits original labels or certifies independence.
Keep proposed corrections separate; adjudicated label changes require a new
approved/versioned analysis, not overwriting R4 gold or summaries.

## 5. Update the ACTUAL existing main-PC paper

Back up the latest draft, which may contain unpublished main-PC edits. Use the
prepared prose/tables to strengthen it; do not invent a replacement manuscript or
overwrite the latest draft with an older server copy. Emphasize the separation of
structure, actionable extraction and application-state reliability.

Keep N/M/P/V8-D/R4 cohort history, exact F1 definitions, fixed-script/common-adapter
scope, fresh/reused provenance, parser/decoder differences and incomplete human
review explicit. Preserve the original integration failure as historical evidence.
Use a bounded matched-development claim, not generic accuracy, real-user interview
success, deployment readiness or general SLM superiority. Use the suggested
six-page main/supplement split to avoid bloating the existing draft.

The A-F matrix is not a completed two-model-by-six-method factorial study. Do not
turn the one-conversation slot-wise diagnostic into a full comparison, configured
XGrammar findings into universal results, or historical 0.8B/2B rounds into a
controlled model-size experiment. Historical primary A/B/D/full-E receipts are
absent on this server; locate the exact PUBLIC receipts on the main PC if available
and verify them offline. Do not inspect private/sealed prospective labels.

Complete the draft's appropriate build/document checks and report actual edited
paths, changed tables/claims, validation results and unresolved human decisions.
Do not mark human review or the whole research complete.

## 6. Boundaries for further work

No new model generation, paid APIs, retraining, new adapters, sealed-label access,
participant contact or spending-ledger changes. Old spending approvals do not
authorize a new experiment. The future independent-study protocol is only a
proposal: humans must approve cohort/language/writer/reviewer independence,
privacy/institutional requirements, endpoints, resources and any inference budget
before collection or execution. No independent requests were generated here.
