# Proposed paper update; not a replacement manuscript

Add a separate current-pipeline development cohort table, keeping existing N/M/P/V8-D and historical OpenAI results separate.

| System | Track | Precision | Recall | Raw exact F1 | JSON | Schema | Raw contract | Exact pipeline state, turns | Median seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v8 | component | 0.0000 | 0.0000 | 0.0000 | 87/101 | 25/101 | 25/101 | 15/101 | 5.980 |
| v8 | rollout | 0.0000 | 0.0000 | 0.0000 | 101/101 | 7/101 | 7/101 | 0/101 | 5.615 |
| openai | component | 0.1347 | 0.1751 | 0.1523 | 101/101 | 101/101 | 56/101 | 37/101 | 2.590 |
| openai | rollout | 0.1060 | 0.1440 | 0.1221 | 101/101 | 101/101 | 59/101 | 0/101 | 2.491 |

Methods: 20 frozen conversations, 101 calls per model/track, plus two separate pilot conversations (nine calls per model/track), one candidate per turn, exact non-intent slot/value/status scoring before recovery, current fpy commit 9bc2bd18c495514efa79a4371be1d706b9fc3824 plus bundled fixes and the common strict admission adapter. Local RTX A4000 CUDA BF16 and the pinned final-step v8 adapter; hosted exact gpt-5-mini-2025-08-07 with native Structured Outputs/minimal reasoning. Disclosure: local cap 1024, hosted cap 2048; decoding/transport differ and arms ran serially.

One user-adjudicated state contract change was frozen BEFORE inference: target_column="true" remains a provided raw extraction but is invalid under the unchanged current validator, and that status is carried through expected/supplied prior states. Case IDs and raw extraction gold are unchanged; all eight changed state-status fields are inventoried. Do not describe this as completed independent human annotation.

Results/discussion: contract compatibility and value/status representations dominate the low scores; deterministic recovery is not raw model credit, and neither rollout reaches an exactly complete final state. See FINDINGS_AND_LIMITATIONS.md and the detailed CSV/JSON. Avoid claims about general model capability or a universal winner.

Total hosted usage-rate estimate including cache rates: $0.23867135; conservative guard-accounted cost without cache discounts: $0.25027775. Reserved/uncertain $0; provider invoice unverified. All 220 hosted requests, including pilot and failed semantic outputs, are included.

Validation: 91 evaluator/application tests and three offline server-authentication tests; source/adapter/case hashes verified; raw SDK responses, usage, context hashes, per-slot/domain/language results and recovery traces archived. Human review remains incomplete; this is inspected synthetic development and a fixed script, not an adaptive interview or untouched final test.
