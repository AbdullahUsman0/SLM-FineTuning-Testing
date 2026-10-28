# Proposed additions to the existing paper

| System | Track | P | R | Exact F1 | JSON | Schema | Raw admitted | Pipeline exact turns | Median s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v8 | component | 0.9763 | 0.9611 | 0.9686 | 101/101 | 100/101 | 100/101 | 75/101 | 8.082 |
| v8 | rollout | 0.9644 | 0.9494 | 0.9569 | 101/101 | 100/101 | 100/101 | 52/101 | 8.656 |
| openai | component | 0.4373 | 0.6654 | 0.5278 | 101/101 | 101/101 | 84/101 | 40/101 | 2.583 |
| openai | rollout | 0.4051 | 0.6148 | 0.4884 | 101/101 | 101/101 | 79/101 | 8/101 | 2.479 |

Add a separate corrective-development cohort, with the methods and limitations in README.md. Disclose the failed native-interface configuration and the diagnostic controls; do not erase it or replace historical scores. Use PAPER_UPDATE_TABLE.csv for complete corrections/unknown/recovery/state/latency counts and billing-usage-analysis.json for per-revision/cumulative cost. Do not fabricate or replace the paper source, claim independent human review, treat component final snapshots as whole-conversation success, or claim a universal winner from these point estimates.


Whole-task usage-rate estimate: $0.59244725. Whole-task conservative ledger accounting: $0.64702325. Corrective revision alone: $0.35377590 usage estimate. All previous spending is included; reserved/uncertain $0; user cap $1, internal ceiling $0.90. Usage-based estimates are not a provider invoice.
