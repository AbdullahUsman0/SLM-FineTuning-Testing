# V7 r8 remote verification and pilot readiness

The remote GPU PC pulled `6bd184364580c4e4108d27b2efb81ee01a27a400`, independently verified the r8 archive and all 131 payload files, and completed development-data verification and selected sentence review. No training has started. The six annotation rules are agreed; the actual human sample and overlap decisions remain pending.

Source bundle SHA-256: `9f8424895c4f7591b65a5f1aaa7e87104f8035671c272177cd8b6a5bb8bba22b`.
Corpus manifest SHA-256: `4e59ba3eafa689325f94b108719da02b955e5c4f5de5571f208053e4c905da72`.
Prompt: `v7-user-approved-extraction-1`.
Base: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
Extracted source: `D:\SLM\v7-approved-policy-20261009-r8-review`.

## Verification and limits of this review

- Frozen verification reproduced 3,400 development conversations and 17,680 SFT rows: 2,800/600 conversations and 14,560/3,120 rows in training/validation.
- Independent checks examined all 17,680 turns and 46,150 updates for JSON-encoded candidate values, unique slots, agreement with fact records, exact current-message evidence and relevant status invariants. No targeted failures were found. The supplied targeted audit also reproduced zero flags in its known defect families across all 524 sample entries.
- Codex read illustrative sentences and labels covering all ten behavior categories, additional Roman Urdu/mixed conversations, filename controls, lifecycle clauses and all 14 initial-message overlap pairs. `selected-sentence-review.json` records the actual scope. This is an agent review and targeted pre-screen, not an independent semantic accuracy estimate or 524 individual human approvals.
- All seven r8 policy tests passed locally. The three unchanged historical gate tests skipped because this portable package omits their r4/r7 fixture directories. Three additional negative checks against the actual r8 materializer passed: pending reviews, pending ledger even with completion flags, and a mismatched policy hash all prevented materialization. Do not describe the three skipped tests as local passes.
- The GPU environment's cached tokenizer reproduced the transferred receipt exactly, including keyed lengths. Training/validation maxima are 3,406/3,390, with zero above 3,584 and no truncation. No model weights were loaded for this check.
- The original human ledger remains byte-identical, with all 524 decisions pending. No approval was signed or human reviewer identity invented.

## Findings from the actual examples

The checked filename examples label the source reference `provided` and the supported filename-derived format `inferred`, while omitting access mode until stated. Later explicit declarations upgrade format to `provided` without a correction flag. Extensionless and `.unknown` examples leave format unset until the explicit declaration. Uppercase supported suffixes are present. URL query and `.csv.gz` exclusions are covered by the policy tests, rather than demonstrated as positive SFT inference coverage.

Explicit same-value confirmations use `confirmed`; subsequent replacements use `provided` with correction true. The independent check found 680 confirmation updates with matching prior values. The 340 unresolved-cadence initial turns fill neither cadence field; the later clarification supplies both stated schedules. All 3,400 unknown-omission turns have no updates. These checks establish consistency with the convention, not model behavior.

Filename-inference coverage is sparse and uneven:

| Language | Training inferences | Validation inferences | Coverage detail |
| --- | ---: | ---: | --- |
| English | 60 | 12 | All four formats in both splits |
| Roman Urdu | 6 | 1 | All four in training; only parquet in validation |
| Mixed | 2 | 0 | Only JSON in training; none in validation |

There are 81 inferences in total, but the statement that all four formats and all three languages occur in the corpus does not establish coverage of every language/format combination. Add independently worded Roman Urdu and mixed validation cases before drawing multilingual filename conclusions. Decide explicitly whether to retain r8 for a limited pilot or create a new immutable revision to improve this coverage; do not change r8 in place.

The 14 queued nearest pairs all belong to `ambiguous_then_clarified`. They repeat the unresolved-cadence clause and target forms with different entity groups and horizons. Their Jaccard scores range from 0.6000 to 0.6667. The original overlap summary reports 20 threshold pairs among 1,680,000 initial-message comparisons; the queue contains nearest representatives, not every pair. The corresponding actual messages are in `overlap-review.md` and `overlap-detail.json`.

A separate exact-string check found **744 of 3,120 validation current messages (23.85%)** also in training: 600 unknown-omission, 59 explicit-nonforecasting, 60 explicit-boolean and 25 lifecycle-bundle turns. There are **zero exact complete SFT message-list duplicates** and **zero entity-group collisions** across training/validation. The repeated current messages have different prior contexts. This is a template-overlap limitation, not proof of an invalid split or zero semantic leakage. Synthetic validation is useful for controlled behavior checks; an independently authored conversation cohort is still needed.

Several bundled overview paragraphs still point to r4/r7 review paths or historical counts. Follow `START_HERE_V7_APPROVED_POLICY.md` and the policy/rubric inside r8. Those documentation inconsistencies did not change the verified r8 records.

## Concrete pilot plan, pending review

`pilot-plan.json` freezes one complete ten-conversation entity group per domain per split, using a deterministic hash selection. Each split contains 200 conversations / 1,040 extraction rows, with 160 English, 20 Roman Urdu and 20 mixed conversations. Domain families are 60 weather, 120 economics and 20 combined, preserving 30/60/10. Source groups remain disjoint.

Proposed pilot: fresh stock base plus a new LoRA adapter; rank 16, alpha 32, dropout 0.05, one epoch, learning rate 5e-5, batch 1, accumulation 16, maximum length 3,584, no truncation, seed 42, complete resumable checkpoints every ten optimizer steps and no early stopping. Expected optimizer steps: 65. This is a plan; no training-input directory or adapter has been created.

Before starting: complete genuine human decisions, resolve the disclosed filename-language coverage choice, bind approval to the corpus/policy/rubric/decision hashes and local tokenizer receipt, then use the unchanged guarded materializer. Subset only materialized approved inputs by the frozen IDs. Freeze a v7 extraction evaluator and behavioral acceptance criteria before measurements. Give stock, v6 and the pilot the same v7 prompt, cases, parser and generation settings; measure component and rollout behavior, keep failed outputs in the denominator and use conversation-cluster intervals. Loss alone cannot select the adapter.

The gate in `scripts/materialize-v7-training.py` requires substantive decisions for all sample IDs and both `review_complete=true` and `overlap_review_complete=true`. Its explicit error is: **“Human corpus and overlap reviews are still pending.”** Policy agreement and agent checks cannot satisfy that requirement. The report does not request a second approval of the six conventions.

## Human review files and reproduction

`human-review/` contains byte-identical copies of the 20 readable domain files, the 524-row pending decision ledger, review manifest and approval template, plus r8's current policy and rubric. Review the actual sentences, unwanted/missing updates, statuses, evidence and prior-state interventions. Review every pair in `overlap-review.md`. Work in a new copy for human edits so the verified portable source stays intact. Record the actual reviewer, UTC date, decision and substantive rationale; completion flags alone are insufficient.

Run the source package's verifier and tests from its extraction root with the pinned `requirements-v7-data.txt` dependencies. Remote data-check dependencies were installed in `.review-deps`, leaving the existing GPU venv unchanged. No paid model calls or final-label reads/generation were performed. Final split assignments were treated only as metadata. No historical results or production prompt defaults were changed.

To reproduce the additional audit, copy `independent-review.py` into a new child directory of the verified r8 extraction root and run it with that root's pinned dependencies on `PYTHONPATH`. It writes new diagnostic files, refuses overwrites and never starts the trainer. The tokenizer receipt is in `local-tokenizer-preflight.json`; the supplied known-defect audit's outputs are in `targeted-audit/`.

The published review ZIP contains these diagnostics, selected sentence-review scope, human-readable pending review files and the pilot plan. It contains no model weights, optimizer checkpoints, credentials, caches, installed dependencies or sealed labels.
