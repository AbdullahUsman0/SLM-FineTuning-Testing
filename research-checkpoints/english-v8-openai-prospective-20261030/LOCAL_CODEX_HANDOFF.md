# Remote GPU Codex handoff: English-only prospective v8/OpenAI comparison

Continue the existing FYP project in `AbdullahUsman0/SLM-FineTuning-Testing`, branch `experiment/v5-grounded-20260919`. This publication is a PREPARATION capsule. No new comparison results, independent human labels, spending approval or training are supplied. Do not confuse it with the completed historical R4 study.

## Authorization and immediate objective

The author requested publishing these inputs for the remote GPU server. Complete offline setup, detailed semantic checking of the AI proposals, a reproducible executable evaluation runner, a source/model freeze proposal and a realistic API cost estimate. Use judgment to resolve routine technical questions. Do not return to Grok or train a new adapter.

Before candidate inference, resolve the label and execution gates below. Existing approval for old R4 paid calls does not authorize this new workload. New API spending remains $0 until the human explicitly approves a new bounded budget. Do not run either candidate arm while changing its labels or prompts against observed responses. Do not turn machine checks into claims of independent human review.

## 1. Safe checkout and verified intake

Preserve all local edits, model caches, adapters, training runs and old evaluation evidence. Fetch and use `--ff-only` if safe. If the worktree diverges or has conflicting changes, use a separate checkout; never reset/clean away local work.

Read the files in `research-checkpoints/english-v8-openai-prospective-20261030/`: README.md, PUBLICATION_MANIFEST.json and this handoff. Verify their recorded file hashes.

Run the standard-library verification/extraction script, choosing a NEW output directory:

```powershell
python research-checkpoints/english-v8-openai-prospective-20261030/verify_extract.py --output runtime/english-followup-r1
```

ZIP: `english-protocol-r1.zip`

SHA-256: `310f27c64afba23342310fad9742b90032f4fc63a8bc4ae3ae79b5b84a539b95`

The script checks ZIP SHA-256, CRC, internal member hashes and safe new extraction. It must reject existing destinations. Read the extracted README.md, protocol.json, READINESS.json, intake-audit.json, INPUT_PROTOCOL_SNAPSHOT.json, case-decision-register.jsonl and AI_DRAFT_VALIDATION.json before proceeding.

## 2. Scope fixed before model predictions

- Primary: **44 English single-target conversations, 205 turns**: 14 weather, 30 economics.
- Separate challenge: **3 English multi-target conversations, 18 turns**, EXT-010/020/050. Current singular target/source fields cannot represent their joint requests. Do not simplify their text, pretend one target was intended or pool them into primary F1. The current task is preparing/reporting their scope limits, not inventing a new multi-target benchmark.
- Twelve Roman Urdu/mixed records are excluded at the author's request before inference. Original historical multilingual R4/V8-D results remain historical and must not be relabelled English-only.
- EXT-060 was truncated in the received export; it is quarantined, not completed by guessing.
- Original case IDs and user messages are preserved. Do not renumber or create extra cases to make the balance look right. The earlier 30/60/10 target is not achieved in the primary set; report actual counts.
- Grok-generated synthetic scripts, with Codex input-only curation. Exact Grok model/settings are unavailable. Do not call them independently human-authored, authentic-user, sealed-final, zero-leakage or guaranteed unseen data.
- No normalized exact matches found against the declared 48,947 public v5-v8 training/development turns. This is a limited exact-match screen. All candidates still name files and narrative-template dependence remains a limitation.

## 3. Semantic annotation comes before outputs

`AI_DRAFT_ANNOTATIONS.jsonl` contains AI-authored candidate extractions and ideal-state drafts for all 205 primary turns. `draft-annotation-review.csv` is populated for checking. These proposals are NOT ground truth; the original blank worksheet is also retained. The 205 Pydantic checks establish shape/evidence properties, not semantic correctness or complete slot coverage.

Inspect every turn and previous context. Check missed slots, extra values, precise free-text boundaries, explicit confirmations, replacements, exact current-message evidence and preservation of unrelated state. Review both FP and FN possibilities; do not merely validate already-present proposals. Correct genuine annotation mistakes before outputs exist. Keep every original proposal, before/after label diff, reason and actual checker identity. Codex checking is AI checking, not human annotation. Mark human review honestly and leave a reviewable corrected packet if no human checker is available. Ali Hassan Khan's previous R4 review does not cover these new cases.

Follow the prospective protocol, which settles representation decisions:

- Keep the 79-slot schema. Target/source singular fields are not silently converted to lists.
- Unknown information and vague horizons supply no invented actionable values; retain unresolved constraints separately. Never map unknown to dont_care.
- Observation, forecast-generation, data-fetch and refitting cadences are different slots. In EXT-004 daily observations remain daily when the requested forecast output becomes weekly.
- A filename supports source_reference and inferred suffix format; it does not alone establish upload mode, unit, column, data contents or timezone.
- Year-only, season, harvest, mid-year and unanchored relative history do not get fabricated precise timestamps. Record them as unresolved under datetime-only slots. Event schedules and clock details must not become invented numeric durations/timezones.
- Do not infer IANA timezones from locations or UTC-like column names. Forecast delivery UTC is not necessarily the source timestamp timezone.
- Use a fixed normalization and target-boundary convention. Treat the input files/market descriptions as hypothetical scenario facts; do not browse to repair them to real-world market frequencies.
- Preserve negative constraints without converting rejected targets into positive slots. Report constraints that the schema cannot represent separately from actionable F1.

The primary endpoint is exact actionable slot/value/status F1. Do not claim it measures every requirement the user expressed. If policy must change after this offline check, version/diff it before inference and supply identical policy information to both models.

When genuine semantic review is complete, hash the final labels, ideal-state rules, ambiguity/adjudication record and case eligibility. Keep reviewer attestations separate from machine receipts. Do not fake a signed human approval to clear a gate. If no human review is available, prepare everything and report that gate; any later explicitly authorized AI-labelled exploratory run must be labelled accordingly, not silently promoted to the planned reviewed comparison.

## 4. Reuse historical implementation carefully, freeze a NEW condition

Read the existing R4 publication and offline supplement:

- `research-checkpoints/current-pipeline-v8-openai-r4-20261028/LOCAL_CODEX_HANDOFF.md`
- `research-checkpoints/r4-paper-offline-supplement-20261029/LOCAL_CODEX_HANDOFF.md`

Verify their historical archives before using their reproduction code. Their known current-pipeline/evaluator sources are preferable to independently reinventing scoring. Historical R4 used verified reused local responses; this prospective run requires **fresh responses for both arms**. Do not reuse any old R4/v8-development outputs.

Build a new resumable runner against the actual common augmented prompt, schema, native parsing/actionable policy and fixed user scripts. The packet is not a ready-to-run GPU launcher; runner integration is part of this handoff. Validate it on separate artificial control fixtures, not by tuning against candidate outputs. Do not add an adaptive interview, synthetic gold-state repairs, automatic recovery fixes, model retries or per-case prompt adjustments without a newly declared condition.

Compare raw extraction before deterministic recovery. Track admission, recovery and resulting application state separately. Keep the ideal reference-state reducer independent of the application recovery code being evaluated.

Freeze actual repository/dependency commits, local patches and file hashes; common prompt/renderer/schema/scorer; semantic labels; generation budgets; parsing/normalization/recovery; failures/refusals/transport retry rules; hardware and library versions. Input/protocol snapshot alone is not this executable/gold freeze.

## 5. Locate models, preserve their identities

- Base: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Final-step v8 adapter weights SHA-256: `c83e0f1a458125970eb70ff360ebead02618f2f48b5bbb59e2ca91a5b7c4c0df`.
- Config SHA-256: `a1ce14418c5c22375ccfd773338ef1b09117ec9b9e72db034c71ade886b4664c`.
- Hosted candidate: `gpt-5-mini-2025-08-07`. Verify its availability before the source/model freeze; if unavailable, declare a different model/condition rather than substituting silently.

Find the existing verified adapter and base cache on this server, report actual paths and verify hashes. Do not assume the main PC's or a previous GPU PC's paths exist. This capsule contains no weights. Read the training publication at `research-checkpoints/v8-final-20261018T230500Z/training-completed-20261023T002215Z/LOCAL_CODEX_HANDOFF.md` if a model download is needed. Do not retrain or silently quantize/change the adapter.

Use the compatible frozen GPU runtime and record differences from historical A4000 BF16. OpenAI is an API workload; the GPU is for local v8 only. Secrets stay in environment/secure local storage; never print or commit keys, authorization headers or cookies. Do not test keys via a paid candidate call before budget and readiness gates.

## 6. Planned execution after gates

Both tracks use the same 44 conversations/205 turns:

| Arm | Component with gold prior | Rollout with own accepted prior | Main calls |
|---|---:|---:|---:|
| v8 on GPU | 205 | 205 | 410 |
| OpenAI API | 205 | 205 | 410 |

Thus 820 main calls in total, **410 paid hosted calls**. Pilot/transport retries, if approved, are separate and must be included in the budget estimate. Do not quietly multiply calls for slot-wise extraction or six-strategy comparisons; that is outside this handoff.

The future fresh-call budget must be estimated using measured prompt lengths and current official rates for the selected endpoint/model, including output/reasoning allowances and retry reservations. No new dollar ceiling has been approved. Do not reset or treat the old R4 ledger's unused balance as authority for this new task. Present the estimate and wait for explicit human budget approval before paid dispatch.

After readiness and budget gates, execute each conversation sequentially and keep component/rollout states separate. Alternate arm order where practical. Store exact inputs, response/model metadata, usage, raw outputs, parsed updates, before/after recovery states, timings and errors after every call. Save durable call IDs and hashes. Resume only matching completed records; reject changed inputs, source or labels. Do not cherry-pick successful reruns. Report transport failures/caps/refusals honestly.

Run in a persistent process on the remote server with stdout/stderr and progress files. Locking the controlling laptop does not stop work on the remote server, provided the server remains powered, awake and connected. Do not promise it survives server sleep, restart or lost power. Use a hidden background process on Windows; do not change system power policy without authorization. No anti-idle or quota evasion tricks.

## 7. Analysis and publication

Primary: rollout exact actionable micro F1. Secondary: component P/R/F1, JSON/schema/native admission, unsupported/duplicate updates, complete final-state equality, per-turn equality, correction and unknown handling, median/p95 latency, tokens and cost. Save TP/FP/FN counts and explicit denominators; do not call F1 generic accuracy or training loss extraction accuracy.

Paired bootstrap: 10,000 resamples, seed 20261030, conversations as primary clusters, recompute micro counts per resample. Keep repeated narrative families/contributor dependence explicit and use a defensible preassigned family sensitivity if feasible. No significance guarantees, equivalence claim or winner selection from isolated point estimates. The 44-conversation scope has no power guarantee.

Keep this English-only prospective condition separate from N/M/P/V8-D and historical R4/R4.1. Do not compare unmatched cohorts as if paired. No paper update may report new F1 until both fresh arms/tracks pass completion and matching checks. New synthetic generator data does not establish human-authored or real-user generalization.

Publish only a bounded, reviewed artifact directory on this branch: frozen protocol/gold/source receipts, raw responses/states with secrets removed, summaries, failure findings, paired analysis, environment, API ledger, offline replay tests, archive manifest and ZIP SHA-256. Verify remote download and every archive member. Preserve models/runs and all historical reports. Do not push `.env`, private credentials or human identities without authorization.

## Return to the author

Initially return verified extraction path, label-check findings/diffs, actual model paths/hashes, tested runner, freeze/readiness status and bounded API estimate. State whether the run is blocked on genuine label review or spending approval. Do useful offline work first; do not simply ask the author to solve technical issues.

After any separately authorized completed experiment, return commit, verified ZIP link/hash, all main metrics, paired intervals, provenance, actual spend and explicit limitations. Do not say a paper is 8/10 or ready for deployment simply because preparation completed. The paper is on the main PC; this publication does not overwrite or fabricate a replacement manuscript.
