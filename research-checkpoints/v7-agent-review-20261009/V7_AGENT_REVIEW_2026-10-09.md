# V7 dataset review and corrections — 9 October 2026

## Decision

**Use `v7-20261009-r7-dev` for the next human review. Do not train r4 or the
internal r5/r6 development trials.** This is a corrected development candidate,
not a new trained model and not an independently human-approved dataset.

The original `v7-20261008-r4` files, manifest, human decision ledger and sealed
final remain preserved. No final labels were read or regenerated in this review.
R7 has training and validation records only; the 600 reserved final assignments
are metadata, not 600 generated r7 conversations. R4's sealed final must not be
silently reused as an r7 benchmark.

**Roman Urdu:** Maine dataset ke rules, labels aur examples check karke jo
maslay mile woh nayi copy mein theek kiye hain. Yeh abhi model ki training ya
insani approval nahi hai. Purani files mehfooz hain; final test ko nahi khola.

## What was reviewed

- A targeted programmatic audit of all **3,400 development conversations** and
  **17,680 extraction turns** against the annotation contract.
- A **524-conversation stratified sample ledger**. This records agent checks;
  it does not claim that 524 independent human conversations were manually read.
- Close inspection of representative English, Roman Urdu and mixed behavior
  clauses, domain target forms, lifecycle values and generator logic.
- All **35 original nearest-neighbor overlap summaries**, with their initial
  target, horizon, language, source group and shared clause structure compared.
  The audit records full paired messages for subsequent human review. This is
  not an audit of every one of the 75 threshold pairs or all semantic overlaps.

The sample was chosen by domain, language and behavior. Flag rates below are
engineering findings, **not an unbiased estimate of dataset accuracy**.

## Findings in the original r4 candidate

| Finding | Observed scope | Correction |
| --- | --- | --- |
| Unchanged fields repeated as new `provided` updates | 125 updates in 82 conversations | Exclude already introduced target/time columns from later lifecycle bundles |
| Success criterion refers to another fictional location | 62 conversations | Bind goal, decision and success criterion to the same scenario scope |
| Rejected alternative appears inside the positive target value | 600 target/problem tuples across 300 conversations | Keep the requested measure as the positive value; use a distinct alternative in explicit contrast cases |
| Horizon number is predictable from domain and behavior | All 200 domain × behavior cells have a constant numeric horizon; 165 also have a constant complete duration object | Vary the horizon across source groups; vary target forms within each behavior |
| Approval/provenance booleans only supervise `false` | 62 approval and 63 provenance updates; no `true` | Balance explicit true/false values across source groups |
| Confirmation relies on “Yes” / “Haan” alone | 680 confirmation turns in 340 conversations | Explicitly confirm the previously recorded horizon |
| “Every hour” followed by a different schedule without explicitly resolving the mention | 340 conversations | State that the earlier phrase was not agreed, then supply the two agreed schedules |
| Awkward plural Roman Urdu durations | 768 flagged turns in 426 conversations | Use forms such as `3 ghantay`, `2 mahinay`, `4 haftay` |
| Synthetic overlap | 35 flagged representatives, all shared templates; none has an identical initial message | Disclose dependence; retain only as synthetic development data |

The issue flags affect **1,318 development conversations** in total, with
overlap between issue families. Of the sample's 524 rows, **304 were flagged for
revision** and **220 had no issue detected by those checks**. The latter were
not approved by a human. Wording concerns and semantic contradictions are not
the same severity and should not be combined into a “wrong-label percentage.”

### Example 1: repeating a known field

The user first says `The target column is called tmean_f.` A later lifecycle
bundle repeats `The target field name is tmean_f.` The old gold emits a new
`provided` update despite unchanged prior state. The corrected lifecycle bundle
introduces other fields instead. Explicit confirmations are a separate category.

**Roman Urdu:** Agar target column pehle se `tmean_f` hai, to wohi baat dobara
kehne par usay nayi requirement nahi banana chahiye. Tasdeeq aur nayi information
ko alag rakhna hai.

### Example 2: wrong place in the success criterion

The initial request is for `Merlow temperature site 1000`, while the old
success criterion refers to `Birch synthetic temperature scope 2`. Both strings
pass structural validation, but they describe different locations. R7 uses the
same scenario scope throughout and aligns the MAE threshold with
`acceptable_error` when both fields are stated.

**Roman Urdu:** Forecast Merlow ke liye tha, lekin success ka rule Birch ke liye
likha hua tha. JSON sahi tha, magar baat aapas mein match nahi karti thi.

### Example 3: a shortcut in the horizon

In r4, the formula includes `7 × group + index`, while
`index = 10 × group + behavior`. Modulo 17, the group contribution cancels.
Therefore a model can learn a fixed number per domain/behavior rather than
read the user's horizon. R7 makes that number vary across groups.

**Roman Urdu:** Har aik qisam ki request ka horizon number wohi reh raha tha.
Model jumla samajhne ke bajaye pattern yaad kar sakta tha. Ab number badalta hai,
is liye usay user ki baat se value leni hogi.

### Example 4: positive target and rejected target

For `headline CPI index level, not its growth rate`, the positive requested
measure is the index level. The old value includes the rejection clause inside
the target itself. R7's positive value contains the requested measure and scope;
contrast cases explicitly name a different rejected measure.

**Roman Urdu:** User CPI ka index level chahta hai, inflation rate nahi. Jo
cheez nahi chahiye usay positive target ki value mein nahi bharna.

## Corrections checked before releasing this candidate

R5 was an internal trial. Rotating the targets exposed **91 conversations**
that requested and rejected the same target. The initial agent auditor did not
detect that interaction; the expanded audit did. R6 corrected the contrast
selection. R7 additionally binds historical weather feature names to the
rotated weather–economics target and uses informative binary-target MAE
thresholds between 0 and 1. These are corpus revisions, **not additional model
fine-tuning rounds**. Internal files are retained for provenance.

For example, a solar-generation request with irradiance covariates uses
`observed_irradiance` in its explicitly listed historical features, rather than
an inherited heating-degree-day feature chosen using the old target index.
Historical weather measurements are not labeled as available future weather.

The early auditor also accidentally counted empty sample entries as affected
scenarios. Its original receipts were retained; the corrected audit calculates
affected scenarios from actual findings. Use the r4 `audit-v2` receipt and the
current r7 receipt, not the superseded initial summary.

## Current development candidate

| Item | R7 development candidate |
| --- | ---: |
| Planned total source assignments | 4,000 |
| Rendered training / validation conversations | 2,800 / 600 |
| Training / validation extraction SFT rows | 14,560 / 3,120 |
| Reserved, unrendered final assignments | 600 |
| Schema slots | 79 |
| Positive non-intent, non-secret slot coverage in each development split | 77 |
| Human review sample | 524 conversations, all human decisions pending |
| Current overlap queue | 14 nearest-neighbor representatives, 20 threshold pairs |

The agreed **30% weather / 60% economics / 10% combined** and
**80% English / 10% Roman Urdu / 10% mixed** proportions are retained exactly
at source level within both development splits. Turn-level language proportions
can differ because conversations have different lengths. The Roman Urdu group
retains technical English vocabulary; the tag is not proof of pure or fluent Urdu.

The same 79-slot schema, pinned parser, model revision and `v5-all-schema-1`
prompt are retained. No earlier cohort, model report or measured score changed.
No model was run, trained or shown to improve during this review.

R7 has no occurrences of the expanded auditor's seven specific issue families,
and no constant-horizon domain/behavior cell. “No detected issue” means only
that these checks passed; it does not certify every possible semantic error.
The lower overlap count is **not proof of independent language or zero leakage**.
Both splits still use the same controlled generator and shared clauses.

## Verification receipts

- Full development verification checks frozen file/source hashes, exact current
  evidence, typed values, source-to-gold pairing, chronological reducer replay,
  group separation, no exact/normalized duplicate full inputs and exact SFT
  regeneration. Additional agent regression checks reject the observed defects.
- `tests/test_v7_agent_review.py` contains 12 targeted regression checks.
- `tests/test_v7_review_gate.py` checks pending human decisions and the old
  candidate's review hold. Agent pre-screen results cannot release training.
  All **15 targeted tests passed** (12 semantic regression checks and 3 gate checks).
- The pinned tokenizer preflight measures every development SFT row without
  loading model weights or truncating examples. See the bound receipt under
  `training/v7-review-20261009/length-preflight-r7-dev.json` for measured lengths.
  Measured maxima are **3,205 training tokens** and **3,189 validation tokens**,
  with zero examples above 3,584 and no truncation.

Current corpus manifest SHA-256:
`9fd0d10a357df7845afe83b2462c6a2299b7afc2b8e650277e9b53b41ab8cf7f`.

## Decisions a human still needs to make

| Policy | Proposed convention | Concrete example |
| --- | --- | --- |
| Problem and target boundaries | An explicit target in a forecasting request fills both with the same exact target noun phrase | `Forecast maximum temperature for Site A` → both contain `maximum temperature for Site A` |
| Filename-only inference | A suffix alone does not fill file format or access mode in this conservative training contract | `Source is weather.csv` → source reference only; `I will upload CSV data` explicitly supplies mode and format |
| Confirmation exception | Explicit affirmation of the recorded value can emit `confirmed`; unchanged facts otherwise are not new `provided` updates | `I confirm the recorded 7-day horizon` → confirmed horizon |
| Unresolved cadence | Do not guess which cadence an unresolved phrase refers to | `Every hour, but sampling or generation is undecided` → neither cadence guessed |
| Unknown information | Unknown/undecided gives no update or fabricated default | `I do not know the timezone` → no timezone update |
| Multilingual meaning | Review target boundaries and inflection with a fluent speaker | `Aglay 7 din ka forecast chahiye; timezone nahi pata` → horizon only from those phrases |

**Roman Urdu:** In rules par insani review abhi chahiye. Khaas taur par file
name se format nikalna hai ya nahi, confirmation ka status kya ho, aur target ka
exact lafzi hissa kya hai — yeh sab training aur testing mein aik jaisa hona chahiye.

## Next steps

1. Read the current domain review files under
   `corpus-v7/reviews/v7-20261009-r7-dev/`. Record actual human decisions and
   reasons in that revision's `decisions.csv`; do not copy r4 approvals or
   convert the agent's 524 “no issue detected” rows into human approvals.
2. Review the 14 current overlap representatives and acknowledge the shared
   synthetic language. The paired messages are in the agent overlap ledger.
3. Adjudicate the six policy choices above, especially confirmation versus the
   unchanged-state instruction. If wording/prompt/labels must change, make
   another immutable revision and a new bound preflight rather than editing
   r7 files. Freeze the actual approved policy before training.
4. After real review, bind approval to corpus, rubric, decision ledger and
   preflight hashes. Use the guarded materializer to prepare development inputs.
   No approval file has been completed by this agent.
5. Run a small training pilot on the remote GPU before the full run. Evaluate
   stock, v6 and the pilot using the same extraction interface and report exact
   F1, per-slot recall, structural validity, correction preservation and unknown
   false positives separately. Keep previous result cohorts unchanged.
6. Add independently authored, human-adjudicated natural conversations for
   external validity. Later provision a matching final split through a separate
   controlled procedure; do not open or retarget the existing r4 final labels.

Known gaps remain: no clarification-question SFT, known-value withdrawal,
unresolved conflict, explicit indifference, STT noise or adversarial fixtures in
this candidate. The non-forecasting extractor controls do not implement a new
dialogue reset controller; the pinned reducer may retain earlier slot values.

## Research-safe conclusion

We identified and corrected specific supervision and generator defects before
v7 training. Machine checks and agent inspection support proceeding to human
adjudication, **not a claim of human-reviewed accuracy or better model quality**.
The previous v5/v6/OpenAI performance findings remain unchanged.

**Roman Urdu:** Dataset pehle se behtar tayyar ho gaya hai, lekin abhi yeh nahi
keh sakte ke model ki performance barh gayi. Insani review, training aur fair
evaluation ke baad hi woh nateeja niklega.
