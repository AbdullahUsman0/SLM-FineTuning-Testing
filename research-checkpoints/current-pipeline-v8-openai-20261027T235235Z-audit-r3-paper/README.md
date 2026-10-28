# Corrective r3 development comparison and paper handoff

The original native-interface zero run is preserved as an integration-failure diagnostic. Its prompt omitted the trained update-entry contract and typed slot metadata; only OpenAI received a machine-readable output schema. The pilot admitted only two of nine v8 component responses, yet the first runner lacked a successful positive-extraction gate before main inference. This was an evaluation setup/gating mistake, not proof that v8 lacks semantic extraction capability.

The corrective common adapter retains native current application context, exposes the SAME complete response schema and typed definitions to BOTH models, and explicitly supplies the unchanged annotation/value contract. Local greedy decoding versus hosted native Structured Outputs still differs and is disclosed. Fresh local positive controls and the full offline suite passed before the corrective main run. No output repair, label-dependent case selection, scorer relaxation, adapter change or retraining was used.

This is a post-hoc corrective DEVELOPMENT revision informed by failed integration, not an independent confirmatory test. The same 20 main conversations and raw gold are retained. The previously user-approved invalid state for column `true` is unchanged. Raw model exact slot/value/status scores precede deterministic recovery/normalization; current application state is reported separately. Fixed-script question context remains None for both arms, and static questions are not fed into later turns. This does not measure adaptive interview quality.

All automated current-application/evaluator checks (303) passed with API credentials removed and outbound networking blocked except loopback. The original 996-call archived v8 replay reproduced F1 0.963953 in both tracks. The known-answer scorer produced 257 TP and zero FP/FN on all 101 main turns. Fresh legacy-interface GPU control on nine disjoint pilot turns reached F1 0.809524; the explicit-contract probe reached F1 1.0, with all nine admitted and 21 TP. These small controls are not main comparison results.

The 20-case main has 101 turns and 257 gold non-intent updates per model/track, 39 positive slot types out of 79, and 16 English/two Roman Urdu/two mixed conversations. Cases were previously inspected public synthetic development and share generation/template structure. No significance/equivalence or universal-superiority conclusion is established. Independent human annotation remains incomplete, and sealed/private final labels were not accessed. Keep the older 31-case OpenAI, 100-case V8-D, failed native r2 and corrective r3 results as separate conditions.

Application recovery can introduce changes even with gold model extraction: the oracle audit records 20 mismatching turns under the declared fixed-script fixtures. Recovery is not raw model credit. Exact state projects non-null values/status/confirmation flags including intent, excluding evidence/confidence/timestamps. Component state uses gold prior; a correct final component snapshot is not a completed successful conversation. Inspect rollout state and the supplementary contract audit for full-conversation conclusions. Extra/forbidden flags do not enumerate every invented fact or wrong value, especially unreadable rejected structures.

No further model calls are authorized by this handoff. The authoritative task budget is shared across old and corrected revisions; changing credentials/revision does not renew it. The exported SQLite snapshot is analysis evidence, not a fresh dispatch ledger on another PC. No model weights, optimizer state, credentials, .env files or caches are included. The same verified v8 weights remain in the existing v8-adapter-20261022 release.

All 404 main raw responses were rescored offline, and all 404 current-pipeline states and component/rollout inputs replayed exactly from the saved journals. Outbound network access was blocked except loopback; no additional model calls or ledger mutations were used. See recorded-main-replay-verification.json and its saved script/log.


| System | Track | P | R | Exact F1 | JSON | Schema | Raw admitted | Pipeline exact turns | Median s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v8 | component | 0.9763 | 0.9611 | 0.9686 | 101/101 | 100/101 | 100/101 | 75/101 | 8.082 |
| v8 | rollout | 0.9644 | 0.9494 | 0.9569 | 101/101 | 100/101 | 100/101 | 52/101 | 8.656 |
| openai | component | 0.4373 | 0.6654 | 0.5278 | 101/101 | 101/101 | 84/101 | 40/101 | 2.583 |
| openai | rollout | 0.4051 | 0.6148 | 0.4884 | 101/101 | 101/101 | 79/101 | 8/101 | 2.479 |

Whole-task usage-rate estimate: $0.59244725. Whole-task conservative ledger accounting: $0.64702325. Corrective revision alone: $0.35377590 usage estimate. All previous spending is included; reserved/uncertain $0; user cap $1, internal ceiling $0.90. Usage-based estimates are not a provider invoice.
