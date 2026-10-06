# V6 failure audit and original versus revised prompt comparison

All 384 new scored calls are complete: two prompts, two context tracks, the same 32 conversations and the same unchanged final-step v6 weights. No training, repairs, retries, paid APIs or sealed final labels. Original and revised prompt order alternated per scenario in one serial BF16 CUDA session. Complete raw predictions, contexts, timings and gold labels are preserved in four compressed reports.

All 32 conversation labels and 96 turns were reviewed by the Codex agent, with no clear factual label error or label changes. This is not independent human review. The original cohort was already inspected to design the candidate prompt, so this experiment measures development-set response to one fixed prompt change, not generalization. Input review notes and the frozen protocol are under evaluation/v6-prompt-audit-20261006.

Strict F1 preserves the original schema-gated exact slot/value/status definition. Supplementary normalized F1 applies only the existing pinned application normalize_value to both gold and observed values, retains the schema-validity gate and status, and does not repair output. It distinguishes accepted enum capitalization and duration text from omissions and wrong values. Free-text synonyms and missing target qualifiers are not automatically forgiven.

| Track | Prompt | Strict F1 | Normalized F1 | JSON / schema valid | New unmentioned slots | Corrected horizon / 32 | Fully correct state after correction / 32 | Unknown valid/no updates / 32 | Median / p95 (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| natural_component | original | 0.533 | 0.604 | 0.979 / 0.969 | 13 | 32 | 32 | 28 | 5.465 / 13.629 |
| natural_component | revised | 0.579 | 0.670 | 0.979 / 0.906 | 0 | 32 | 32 | 32 | 5.435 / 15.378 |
| natural_rollout | original | 0.550 | 0.604 | 0.979 / 0.969 | 13 | 32 | 3 | 28 | 5.300 / 13.005 |
| natural_rollout | revised | 0.602 | 0.670 | 0.979 / 0.906 | 0 | 32 | 8 | 32 | 5.489 / 15.179 |

Local corrected-horizon success and preservation of other prior slots are separate from matching the entire gold accumulated state. A correct correction can coexist with missing initial facts. Unknown tests measure abstention and prior-state retention, not generated clarification-question quality. Unmentioned slots are annotation-relative flags, not independently adjudicated semantic hallucinations.

See summary.csv for weather, inflation, stocks and crypto; analysis.json for full counts, normalized diagnostics, paired latency and 10,000-resample conversation bootstrap intervals; slot-error-audit.csv for every expected/emitted slot. Confidence intervals reflect resampling these development conversations, not unseen users or prompt-selection uncertainty. No fine-tuning decision is automated.
