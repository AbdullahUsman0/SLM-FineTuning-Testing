# V7 dataset requirements and preparation

## Decision

The candidate is **v7-20261008-r4**, 4,000 new controlled synthetic source
conversations. It retains the 79-slot schema and exact v5 prompt interface but
uses extraction-only SFT. It is not trained, independently human reviewed or a
demonstrated improvement over v6. The agreed scenario mixture is 30% weather,
60% economics and 10% combined; language is 80% English, 10% Roman Urdu and 10%
mixed. Example-level proportions are reported separately because trajectories
have different lengths.

R1 was an internal positive-intent construction trial. R2 added negative intent,
privacy boolean balance and unknown wording. R3 adds the observed seasonality and
interval unknown failures, explicit resolution, and source-reference-only
contrasts. R4 corrects inherited domain-sign assumptions for index levels and binary events,
and adds explicit no-seasonality controls. Earlier internal candidates are superseded, not additional trained
datasets. Their saved files are preserved locally; use only r3 with the current
generator source lock.

## Observed defects and requirements

The 31-case fresh study is development evidence. V6's rollout F1=0.4986,
precision=0.6312 and recall=0.4120. The paired OpenAI interval includes zero.
No system achieved an exact complete conversation.

| Observed v6 exact errors | V7 requirement | Verification and later measurement |
| --- | --- | --- |
| problem_statement 31 missing/mismatched tuples | Explicit target supports both problem and target noun phrase | Fact identity and human phrase-boundary review; exact and supplemental semantic analysis |
| units 26, target descriptions 23 missing/mismatched | Unit in each initial request; domain target contrasts | Per-slot recall and domain/language breakdown |
| horizon 18, frequency 14 missing/mismatched | Multiple cadence roles, replacements and clarification | Replacement-only delta, unchanged-state preservation and exact status |
| 4 extra timezone, 4 seasonal-period and 4 interval tuples | Unknown controls naming those unset slots; later explicit values | No-update rate and false positives by unknown family |
| 3 rejected first turns | Unique IDs and consistently encoded gold | JSON/schema/admission measured separately from semantics |
| Initial mistakes persist through correction | Transparent seeded context and user restatement | Recovery with evidence versus unsupported invention; full-state rollout accuracy |
| Mode/format errors and inference risk | Source-only initial requests and later explicit access details | No filename-to-column/mode/private-fact inference |

Counts are from the unchanged human-review queue, reproduced in
training/v7-preparation-20261008/error-requirements-source.json. A wrong tuple may
contribute both a missing and extra item. They are not all pure omissions or
fabricated real-world facts. Human adjudication of exact-label boundaries remains
pending. This plan targets failure themes without copying those evaluated cases
into training.

## Composition

Twenty subdomains remain: temperature, precipitation, wind, humidity/pressure,
solar/cloud, weather extremes; inflation/prices, output/growth, labor/employment,
rates/fixed income, equities/funds, foreign exchange, commodities, crypto spot,
crypto derivatives, housing, trade/fiscal, corporate finance; weather-energy and
weather-agriculture. Each has 200 scenarios. Fictional scopes and example.org-like
reserved `.example` endpoints describe inputs, not live financial/weather data.

Ten scenario behaviors are balanced at 400 sources each:

1. Complete core extraction followed by identifiers, purpose and a correction.
2. Observation, prediction, source-refresh, retraining and output-resolution contrast.
3. Single and two-slot correction chains.
4. Unset unknowns followed by explicit timezone, seasonality and interval values.
5. Confirmation compared with replacement status.
6. Incorrect/omitted prior state repaired only by explicit new evidence.
7. Dense lifecycle bundles with full positive non-secret slot coverage.
8. Requested domain target versus an explicitly rejected alternative.
9. Quoted/numeric column identifiers and explicit positive/negative booleans.
10. Ambiguous cadence followed by an explicit resolution.

One rotated non-forecasting control appears per ten-scenario source family.
Five to seven turns are stored chronologically; this is not three paraphrases
counted as an independent conversation. Facts precede outputs, and every gold
update has a value, status, unique ID and current-turn evidence span. The same
source/contrast group and language variants cannot cross splits.

## Split and leakage policy

The frozen 70/15/15 split has 2,800 training, 600 validation and 600 final source
conversations. A ten-case entity/contrast group stays together. Outer wording
variants are reserved by split; technical clauses and semantic structures still
repeat. Initial-message train-validation Jaccard is exhaustively measured, and
flagged validation representatives are queued for review. This is not an
exhaustive semantic-overlap audit or proof of unseen natural-language quality.

The older 31/32 evaluated conversations are checked for exact normalized
utterance copying. No old model output supplies gold. Final labels are generated
once in the ignored training-runs/v7-sealed/v7-20261008-r4 path. The verifier may
hash those files but never opens them as labels. They are procedurally separate,
not encrypted, and must not be regenerated or read for development.

## Prepare and review

Use the original pinned fpy snapshot and exact parser dependencies from the
existing study: pydantic 2.13.5 and jsonschema 4.26.0. Do not install the latest
live sibling fpy as a substitute. ComparisonPolicy verifies all 31 original
Python source hashes and the frozen policy.

```powershell
python scripts/prepare-openai-study-fpy.py --fpy-repo ../fpy
python scripts/verify-corpus-v7.py --output corpus-v7/v7-20261008-r4
```

Read corpus-v7/ANNOTATION_RUBRIC.md, then the domain files under
corpus-v7/reviews/v7-20261008-r4/. Write genuine decisions into decisions.csv,
with reviewer, UTC time and rationale. The sample is stratified; say explicitly
whether approval covers this sample or a broader actual review. Flagged overlaps
and unclear multilingual/technical labels require review, too.

The saved tokenizer preflight uses Qwen3.5-2B revision
15852e8c16360a2fea060d615a32b45270f8a8fc, transformers 5.15.0 and
enable_thinking=False. It measures every development example without weights or
truncation. The final saved result, rather than a guessed word count, determines
the sequence limit.

```powershell
python scripts/preflight-v7-lengths.py --corpus corpus-v7/v7-20261008-r4 `
  --output training/v7-preparation-20261008/length-preflight-r4.json --max-length 3584
```

The preflight defaults to cached tokenizer files. On a new GPU machine use
--allow-tokenizer-download to fetch only the tokenizer at the pinned revision.
Use a new output path when repeating a preflight; existing results are immutable.

After review, write approval with actual corpus_manifest_sha256,
decision_ledger_sha256 and rubric_sha256, review_complete=true,
overlap_review_complete=true, reviewer, reviewed_at_utc and review_scope.
No approval is created by the dataset generator. This records a human decision;
it does not cryptographically authenticate one or imply every generated label
was individually reviewed.

```powershell
python scripts/materialize-v7-training.py --corpus corpus-v7/v7-20261008-r4 `
  --review corpus-v7/reviews/v7-20261008-r4 `
  --approval training/v7-preparation-20261008/human-review-approval.json `
  --preflight training/v7-preparation-20261008/length-preflight-r4.json `
  --output training-runs/v7-inputs/v7-20261008-r4
```

## Training and evaluation after approval

First make a small, source-group-preserving training pilot and measure behavior
against stock/v6 on a frozen validation smoke cohort. Do not select a run merely
because loss is small. For the later full experiment, retain stock Qwen3.5-2B,
pinned revision, LoRA rank 16/alpha 32, dropout .05, batch 1, accumulation 16,
learning rate 5e-5, fixed two epochs and complete checkpoints every ten steps.
The larger dataset changes compute and sample budget; report that confound.

Illustrative full-train command after reviewed materialization and pilot acceptance:

```powershell
python training/train_lora.py --model Qwen/Qwen3.5-2B `
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc `
  --train training-runs/v7-inputs/v7-20261008-r4/train.jsonl `
  --validation training-runs/v7-inputs/v7-20261008-r4/validation.jsonl `
  --output training-runs/qwen35-2b-v7-NEW-UNIQUE-RUN `
  --epochs 2 --learning-rate 5e-5 --max-length 3584 `
  --save-steps 10 --early-stopping-patience 0 --resume-from-checkpoint auto
```

Keep adapter/runtime/input hashes, complete model/optimizer/scheduler/RNG saves,
loss curves and raw reports. Behavioral checkpoint selection must use validation
only, with its rule recorded; the final folder name best-adapter is not evidence
that it is best. Transfer whole verified checkpoints, not only LoRA weights, to
resume after an interruption.

A new v7 extraction-only evaluator must be frozen before measurements. The old
policy runner is intentionally tied to its 31-case cohort; do not repoint its
files or adapt the question-containing v5 runner silently. Pair v6/v7/F on the
same fresh development contexts, then use a separate independently annotated
natural cohort. Score admission, grounding, exact tuple precision/recall/F1,
status-free diagnostics, correction behavior, forbidden updates, per-slot/domain/
language recall, accumulated state and complete conversations. Use scenario-group
bootstrap, not independent-turn intervals. Keep invalid calls in the denominator.

For causal claims, compare equal training budgets and ablate language mix,
extra data and state-repair supervision. Once prompt/parser/adapter choices are
frozen, authorize final evaluation separately. Neither training nor paid API
execution is started by this preparation.

## Remaining limits

Controlled synthetic rendering is reproducible but cannot replace independent
human language. Multilingual examples retain technical English terms and need
fluent review. Rare complex objects are explicit technical requirements rather
than casual speech. No positive credential references are included. Intent is
top-level supervision; no redundant intent slot update is generated. Withdrawal,
unresolved conflicts, dont_care and adversarial/STT corruption require separate
regression cases rather than invented labels in this candidate. A new trained
adapter's actual improvement, final performance and training cost are unmeasured.
