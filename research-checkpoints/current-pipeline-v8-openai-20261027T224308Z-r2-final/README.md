# Current-pipeline v8 versus OpenAI development evaluation

Status: complete_matched_development. Independent human review remains incomplete; this is not a final test or a universal winner claim.

Current fpy: 9bc2bd18c495514efa79a4371be1d706b9fc3824 plus patch 63f5d34a7ac553a9c84f0323f4a5f6955bd6a0948483fb23d69a0627377d8fb9. The same 20 conversation IDs selected before inference were retained. The user approved marking target_column="true" invalid in the application state for labor_employment/0178; supplied prior/expected states carry that status forward. Raw extraction gold is unchanged. The changes and approval are archived.

| System | Track | Raw exact F1 | Precision | Recall | JSON | Schema | Raw contract | Current reducer | Exact pipeline state, turns | Recovery calls | Median seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openai | component | 0.152284 | 0.134731 | 0.175097 | 101/101 | 101/101 | 56/101 | 48/101 | 37/101 | 29 | 2.590 |
| openai | rollout | 0.122112 | 0.106017 | 0.143969 | 101/101 | 101/101 | 59/101 | 49/101 | 0/101 | 17 | 2.491 |
| v8 | component | 0.000000 | 0.000000 | 0.000000 | 87/101 | 25/101 | 25/101 | 25/101 | 15/101 | 50 | 5.980 |
| v8 | rollout | 0.000000 | 0.000000 | 0.000000 | 101/101 | 7/101 | 7/101 | 7/101 | 0/101 | 43 | 5.615 |

Shared task API spending: actual $0.250278, reserved/uncertain $0.000000. User ceiling $1; internal dispatch ceiling $0.90. Usage cost is an application estimate, not an invoice. No budget renewal on key changes or restart.

Raw exact scoring uses non-intent slot + decoded candidate value + status, one wire decode, duplicate-preserving multiset matching. Invalid/capped/refused contracts receive zero true positives. This raw score precedes recovery and reducer normalization. Reducer admission means no exception, and does not mean all updates were grounded/accepted or that the resulting state is correct. The current reducer can discard individual unsupported evidence and retain other updates; trace and before-recovery state disclose this. Pipeline exact state compares all non-null slot values, statuses and confirmation flags, including intent, excluding timestamps/evidence/confidence. Counts are turns, not successful whole conversations.

Deterministic recovery is attributed to the pipeline only. main-summary.json includes correction/unknown, extra/forbidden, language/domain and seeded-intervention exclusion results. Scored records include raw responses, prior/reduced/pipeline states and trace provenance. Extra-slot flags do not count every hallucination or wrong value. per-slot-results.json records exact support and confusion.

Both systems received current fpy semantic instructions and fixed scripted user messages. Component inputs and labels were checked identical; rollout uses each system's own accepted state. No declared assistant question is fed to either system; static generated questions are recorded and removed before the next fixed turn. Current fpy still ranks a pending slot. This is not a real adaptive user interview. The oracle audit identifies deterministic recovery failures even with gold extraction; it is not a model score.

Local decoding: active final-step v8 LoRA on pinned stock Qwen3.5-2B, CUDA BF16, greedy, thinking disabled, max 1024 new tokens. Hosted: exact gpt-5-mini-2025-08-07, native Structured Outputs, minimal reasoning, max 2048 output tokens, retries disabled. Native schema enforcement and transport differ. Unscored local warmup fixtures and runtime manifests are archived separately. No claim of energy/compute efficiency follows from cross-system latency.

The 20 cases are an inspected synthetic subset of the earlier public V8-D set, with shared template/entity structure; 16 English, two Roman Urdu, two mixed, one per 20 domains. Support is not all 79 slots. No significance/equivalence analysis is claimed. Do not compare scores numerically with old 31-case OpenAI or 100-case v8 tables. The offline historical arithmetic audit reproduced the hosted scores; unavailable historical parser implementation limits a full admission re-audit. No new calls were used for that audit.

No retraining, private prospective cohort, sealed final labels, paid judges, paid question generation or automatic study expansion was used. Do not replace the existing paper or fabricate a manuscript absent on this server. Update it only from completed matched receipts and keep earlier cohorts separate.


The final verified packet adds FINDINGS_AND_LIMITATIONS.md, PAPER_UPDATE_TABLE.csv and PAPER_UPDATE_PROPOSAL.md. Cache-adjusted usage-rate estimate: $0.23867135; conservative ledger accounting: $0.25027775. Both include pilots and every hosted generation request, with zero reserved/uncertain requests. These are usage estimates, not an invoice. The original export and primary results are preserved.
