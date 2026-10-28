# Start here: audited and corrected v8/OpenAI comparison

**Latest rerun:** See [the R4/R4.1 native-admission comparison](../current-pipeline-v8-openai-r4-20261028/README.md) and its handoff. It uses fresh hosted responses and trajectory-verified reused SLM responses, with the same native/actionable metrics applied to both. The results below remain the historical R3 strict-contract condition.

**Paper-use qualification:** Read [PAPER_VALIDITY_ADDENDUM.md](PAPER_VALIDITY_ADDENDUM.md) before using these results. The rows below are reproducible strict-contract raw-extraction development scores. They must not be presented as validated updated-production-pipeline accuracy or general model superiority. Native parser fidelity and no-op/unknown metric differences were confirmed in the subsequent offline audit.

Read LOCAL_CODEX_HANDOFF.md and ROOT_CAUSE_AUDIT.md first. Corrective result folder: ../current-pipeline-v8-openai-20261027T235235Z-audit-r3-paper/ . The prior failed native-interface run is retained at ../current-pipeline-v8-openai-20261027T224308Z-r2-final/ ; its zero score is not a reliable model-capability headline. The corrective main results are below; both conditions remain public for transparency.

| System | Track | P | R | Exact F1 | JSON | Schema | Raw admitted | Pipeline exact turns | Median s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v8 | component | 0.9763 | 0.9611 | 0.9686 | 101/101 | 100/101 | 100/101 | 75/101 | 8.082 |
| v8 | rollout | 0.9644 | 0.9494 | 0.9569 | 101/101 | 100/101 | 100/101 | 52/101 | 8.656 |
| openai | component | 0.4373 | 0.6654 | 0.5278 | 101/101 | 101/101 | 84/101 | 40/101 | 2.583 |
| openai | rollout | 0.4051 | 0.6148 | 0.4884 | 101/101 | 101/101 | 79/101 | 8/101 | 2.479 |

Whole-task usage-rate estimate: $0.59244725. Whole-task conservative ledger accounting: $0.64702325. Corrective revision alone: $0.35377590 usage estimate. All previous spending is included; reserved/uncertain $0; user cap $1, internal ceiling $0.90. Usage-based estimates are not a provider invoice.

Independent human review remains incomplete.

Use this publication folder and the corrective result README as the current status. The frozen evaluator README retains preparation-era text for byte-level reproducibility; its initial blocked status is historical. Completion receipts and the ZIP were sealed before upload. PUBLICATION_VERIFICATION.json records the subsequent GitHub download verification.
