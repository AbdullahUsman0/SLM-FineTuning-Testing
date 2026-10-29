| System | Track | JSON | Schema / native | Precision | Recall | Exact actionable F1 | Exact pipeline turns |
|---|---|---:|---:|---:|---:|---:|---:|
| v8 | component | 101/101 | 100/101 / 100/101 | 0.9763 | 0.9611 | 0.9686 | 75/101 |
| v8 | rollout | 101/101 | 100/101 / 100/101 | 0.9644 | 0.9494 | 0.9569 | 52/101 |
| openai | component | 101/101 | 101/101 / 101/101 | 0.6445 | 0.7549 | 0.6953 | 54/101 |
| openai | rollout | 101/101 | 101/101 / 101/101 | 0.6364 | 0.7626 | 0.6938 | 6/101 |

Complete final rollout state: local 9/20; hosted 1/20. Do not report component final snapshots as complete conversations.
