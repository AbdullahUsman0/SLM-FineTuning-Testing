# V7 dataset preparation and teacher report

**Current candidate: v7-20261008-r4. Human review pending; not trained.**

Read [the requirement and training plan](V7_DATASET_PLAN_2026-10-08.md).
The thirteen visually reviewed pages in [the editable teacher report](DATASETS_AND_SLOT_FILLING_2026-10-08.md)
and [printable PDF](Forecasting_datasets_v1_v6_and_slot_filling_2026-10-08.pdf)
explain v1-v6, A-F/J, examples in English and Roman Urdu, the older OpenAI method
study and the fresh four-model comparison. Cohorts and parser policies remain
separate; unmeasured C and limited D results are identified.

[Download the standalone development bundle](v7-development-bundle-20261008-r4.zip)
and [its checksum](v7-development-bundle-20261008-r4.sha256). Extract into a new
folder and read START_HERE_V7.md. It includes the original pinned fpy source,
frozen parser, code and review materials; it does not change live FYP code.
All paths below refer to the extracted bundle. No sealed labels, credentials,
weights or production database are included. No training or API calls were made.

| Item | Count |
| --- | ---: |
| New source conversations | 4,000 |
| Weather / economics / combined | 1,200 / 2,400 / 400 |
| English / Roman Urdu / mixed conversations | 3,200 / 400 / 400 |
| Training / validation / final conversations | 2,800 / 600 / 600 |
| Training extraction SFT examples | 14,560 |
| Validation extraction SFT examples | 3,120 |
| Positive non-intent non-secret slots in each development split | 77 |
| Stratified review sample | 524 conversations |
| Flagged validation overlap representatives | 35 |

The schema still contains 79 slots. Intent is supervised at the top level,
without a redundant intent-slot update. Authentication references have no positive
supervision because they are redacted in the prompt. Every required non-intent
slot has positive development coverage. The 80/10/10 language balance is exact
at scenario level; differing trajectory lengths give slight example-level
differences, recorded in statistics.json.

The final split is stored separately under gitignored
`training-runs/v7-sealed/v7-20261008-r4/`. Its 3,120 SFT examples are not public
training inputs. The public candidate and review files contain development
records only. No model, credentials or production database is included.

Machine checks validate unique slot IDs, exact evidence, source fact identity,
typed values, pinned schema/parser, chronological state transitions, frozen group
splits and regenerated SFT. The exact tokenizer measured a maximum of 3,203
tokens in training and 3,156 in validation, with **zero above 3,584** and no
truncation. These checks do not certify naturalness or semantic annotation quality.

The 1,680,000 initial-message train-validation Jaccard pairs include 75 pairs at
or above .6, yielding 35 nearest-neighbor representatives for human review.
Shared clauses and semantic patterns remain; zero semantic leakage is not claimed.
The old diagnostic cohorts are protected against exact normalized utterance
copying, while their known failures informed this new development candidate.

## What changed

- Initial requests supervise all named target, unit, cadence, horizon and source
  facts; the target phrase supports both problem_statement and target_description.
- Observation cadence, prediction generation, source refresh, retraining and
  output resolution are expressed separately.
- Replacement and confirmation have distinct extraction status labels.
- Unknown timezone, seasonal cycles, interval levels and access references give
  no updates. Later explicit facts resolve them without inventing defaults.
- Seeded prior-state errors are repaired only by explicit new user evidence.
- Filename-only references precede explicit access and format details.
- CPI levels and binary events have coherent nonnegative value plans. Explicit
  no-seasonality and true/false privacy cases counter one-sided boolean learning.
- All SFT is extraction-only; no clarification questions or bad JSON are targets.

## Review before training

Start with `reviews/v7-20261008-r4/temperature.md`, then review the twenty domain
files and record genuine decisions in `decisions.csv`. No review is marked
approved. The separate review-manifest identifies the sample and corpus hash.
An approval template remains false/pending and is not a training authorization.

Use the guarded materializer described in the plan after actual review and
overlap adjudication. Rejected labels require a new immutable corpus revision.
Keep the frozen v5/v6 adapters and all historical results unchanged.

V7 performance, independent human review, the new evaluation protocol and GPU
training are pending. The existing 31-case runner must not be silently repointed
to this extraction-only corpus. Unresolved conflict, dont_care, known-value
withdrawal and STT/adversarial regression coverage remain separate work.
