Add a separate R4 configured-system matched development cohort using the methods and limitations in README.md. Retain older cohorts and the integration-failure audit. Human review remains incomplete.

| Model | Track | Precision | Recall | Exact actionable F1 | Native admitted | Exact pipeline turns | Median seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| v8 | component | 0.976285 | 0.961089 | 0.968627 | 100/101 | 75/101 | 8.082 |
| v8 | rollout | 0.964427 | 0.949416 | 0.956863 | 100/101 | 52/101 | 8.656 |
| openai | component | 0.644518 | 0.754864 | 0.695341 | 101/101 | 54/101 | 2.669 |
| openai | rollout | 0.636364 | 0.762646 | 0.693805 | 101/101 | 6/101 | 2.553 |

R4 usage-rate cost estimate: $0.35448805; cumulative task usage estimate: $0.94693530; cumulative conservative ledger accounting: $1.0469865. Reserved/uncertain $0. Rates include reasoning once; cached input affects the estimate but never dispatch permission. This is not a provider invoice. Total authorization $2; task dispatch ceiling $1.90; this rerun's new-spend ceiling $0.90.
