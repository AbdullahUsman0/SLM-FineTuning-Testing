# V7: six annotation conventions agreed by the user

The user agreed points 1, 2 and 3 after clarification, and had already agreed
points 4, 5 and 6. The agreement is recorded in
`corpus-v7/policies/v7-user-approved-annotation-1.json`.

**Current development candidate: v7-20261009-r8-dev.** All six annotation
conventions are settled. Individual example and overlap reviews remain pending;
no training approval or claim of human-reviewed dataset accuracy is recorded.

## Accepted rules

| Rule | Agreed behavior | Example |
| --- | --- | --- |
| Problem and target | An explicitly described forecasting target supplies both fields with the same exact meaningful noun phrase | “Forecast maximum temperature for Lahore” → both contain “maximum temperature for Lahore” |
| Filename format | Infer a supported extension; explicit format statements use provided | “Source is weather.csv” → source reference provided, file_format csv inferred; access mode remains unset |
| Confirmation | New/replacement values are provided; explicit affirmation is confirmed | “I confirm the recorded seven-day horizon” → confirmed; “change it to fourteen days” → provided, correction true |
| Unresolved cadence | Do not guess the role | “Every hour, but sampling or generation is undecided” → neither cadence filled |
| Unknown information | No value, fabricated default or indifference label | “Timezone nahi pata” → no timezone update |
| Languages | Preserve the same meaning in English, Roman Urdu and mixed wording | “Aglay 7 din ka forecast chahiye” → seven-day horizon |

**Roman Urdu:** Aap ne ab chhay rules agree kar diye hain. Filename se format
nikalenge, lekin usay inferred kahenge. Agar user format saaf bata de, to woh
provided hoga. Pehle se recorded value ki saaf tasdeeq confirmed hogi; nayi value
se replacement correction hogi.

## Implemented changes

- Created a new immutable r8 development revision; r7 and previous reports remain
  preserved. The approved policy and rubric are included and hashed inside r8.
- Added **81 development examples** with supported filename-derived formats,
  including all four supported formats and all three language groups. The later
  explicit statement upgrades inferred to provided without changing the value
  or falsely labeling that provenance change as a correction.
- Added extensionless and unsupported-extension controls. Extension matching is
  case-insensitive and uses URL paths, not query strings. The supported suffixes
  are `.csv`, `.json`, `.parquet`, `.xlsx`; `.csv.gz` is outside this rule.
- Added `v7-user-approved-extraction-1` instructions. This is a new prompt version;
  the v5 instructions and historical benchmark/parser files are unchanged.
- Added a v7 Transformers extraction provider using the same pinned generation
  path and wire contract, and added `--prompt-version v7` to `chat-peft.py` for
  future r8 adapter testing. Provider prompt routing was tested with a mock;
  no model weights were loaded and no inference quality was measured.
- Bound the training materializer to r8's own rubric and annotation-policy hash.
  Policy agreement cannot release pending sample decisions.

The pinned reducer still normally stores an extracted confirmed update as
provided. Extractor status and saved application state remain distinct. No new
confirmation-lock controller was implemented or claimed by this policy change.

## Verification

R8 has **2,800 training / 600 validation conversations** and
**14,560 / 3,120 extraction SFT rows**. The 30/60/10 domain and 80/10/10 language
mixtures, frozen source groups and 79-slot schema are retained. Both development
splits have positive coverage for all 77 non-intent, non-secret slots.

All **seven policy tests** passed, including supported/unsupported suffixes,
current-message grounding, inferred-to-provided upgrades, model prompt routing
and rejection of policy-only approval as dataset approval. The three existing
review-gate checks also passed after the materializer update.

Full verification reproduced 3,400 development conversations and 17,680 SFT
rows, source/file hashes, exact evidence, typed values, chronological state,
group separation and SFT generation. The expanded agent pre-screen found no
instances of its targeted defect families. This is not a human accuracy estimate.

Pinned tokenizer maxima: **3,406 training / 3,390 validation tokens**, with zero
above 3,584 and no truncation. The current overlap queue still has 14 nearest
representatives from 20 threshold pairs; shared synthetic language is disclosed.

Corpus manifest SHA-256:
`4e59ba3eafa689325f94b108719da02b955e5c4f5de5571f208053e4c905da72`.

## What remains

1. Review actual r8 sample labels and overlap messages. Record genuine decisions
   in `corpus-v7/reviews/v7-20261009-r8-dev/decisions.csv`. The 524 rows are pending.
2. Bind a completed human review to the corpus, annotation policy, rubric,
   decision ledger and tokenizer receipt; then prepare inputs and run a pilot.
3. In future model comparisons, give every model the same v7 prompt and the same
   approved annotation convention. Older scores used their original policies and
   prompts and remain historical results, not fresh r8 comparison scores.

No final labels were accessed or generated. The 600 final assignments are
metadata only, and older sealed labels are not a matching r8 benchmark. No
training or paid API calls were started. Independently authored evaluation and
the previously declared conflict/withdrawal/STT gaps remain future work.
