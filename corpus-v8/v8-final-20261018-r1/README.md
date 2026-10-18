# Final experimental requirement extraction dataset

Version: `v8-final-20261018-r1`. This is a full-data training candidate, not another completed result or a final test.

| Split | Conversations | Extraction examples | Source |
| --- | ---: | ---: | --- |
| Train | 3,600 | 17,860 | Full 2,800 r8 training conversations plus 800 new focused conversations |
| Development validation | 800 | 3,980 | Full 600 r8 validation conversations plus 200 new focused conversations |

Both splits preserve 30% weather, 60% economics across subdomains and 10% combined requests. Conversation language labels are 80% English, 10% Roman Urdu and 10% mixed. There are twenty domain profiles. Language quality has not been independently judged.

The v7 r8 pilot trained only 200 conversations. This package retains the complete public r8 development trajectories and gold labels, adds a versioned instruction extension, and supplies contrasts for the observed failures:

- Sampling, forecast issuance and source refresh are separate roles.
- A refresh interval does not change API/upload access mode.
- Historical coverage is filled when past coverage is explicit, and omitted when only a cadence or future horizon is stated.
- Unknown information and rejected assumptions provide no value.
- Corrections change only their specified fields and preserve earlier requirements.
- Safe external-reference identifiers use invented `secret://synthetic-...` values; these are not passwords, keys or usable credentials. The pinned redactor accepts a complete safe reference turn, but redacts the same URI embedded in a longer sentence.

All 79 slots are in every prompt. All 78 non-intent slots have positive update supervision somewhere in the development corpus. Intent is supervised at the top level. Positive coverage is not a guarantee of adequate per-slot diversity or accuracy. See `validation-report.json`, `statistics.json` and the fact/evidence ledgers in `splits/`.

## What was checked

The frozen compressed inputs and their uncompressed hashes are recorded in `manifest.json`. The verifier checks SFT-to-conversation alignment, closed response structure, current-message exact evidence, duplicate/forbidden supervision, all-schema prompt coverage, entity/input separation and full gold reducer replay with the pinned fpy source. Ratios are checked per split. The lexical overlap report exposes remaining shared wording and near-template pairs; no zero-leakage claim is made.

The authoring agent performed these machine checks. Independent human annotation and overlap adjudication remain pending. The user's explicit experimental training request is recorded separately in `training/final_round/TRAINING_AUTHORIZATION.json`; it does not assert that people reviewed these labels. The earlier human-review gates and approval files were not altered or forged.

`review-sample.jsonl.gz` contains development conversations for inspection. It is neither a completed review ledger nor a final test. Existing r8 synthetic context interventions are preserved and marked. They must be reported separately in behavioral evaluation.

## Use

Read `training/final_round/README.md` and `EXPERIMENT_PLAN.md`. The remote GPU agent can prepare, train, resume and run a local development diagnostic without importing untracked v7 authoring code. The builder can recover its selected public parent inputs from the tracked r8 ZIP if the loose parent files are absent.

Training is extraction-only; the model is not learning numerical forecasts or the complete interview question policy. No final labels, private prospective evaluation candidates, base weights or adapters are included. Do not insert this corpus's development scores into an old paper comparison table as if its cases and policies were unchanged.
