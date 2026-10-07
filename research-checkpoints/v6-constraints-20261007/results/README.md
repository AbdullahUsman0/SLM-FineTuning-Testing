# V6 decoding comparison: original versus constrained

All 384 new scored calls are complete: two frozen decoding arms using the identical original prompt, two context tracks, the same 32 conversations and the same unchanged final-step v6 weights. No training, repairs, retries, paid APIs or sealed final labels. Decoder order alternated per scenario in one serial BF16 CUDA session. Complete raw predictions, contexts, timings and gold labels are preserved in four compressed reports.

Grammar uses the existing exported structural schema with fixed object-field order, one fresh XGrammar matcher per request, and torch_native masking on CUDA. Inner JSON strings, duplicate slot updates and semantic correctness still require the unchanged application validator. Compilation time is separate from warmed extraction timings.

All 32 conversation labels and 96 turns were reviewed by the Codex agent, with no clear factual label error or label changes. This is not independent human review. The original cohort was already inspected to design the earlier prompts, so this experiment measures development-set response to one fixed decoding change, not generalization. Input review notes and the frozen protocol are under evaluation/v6-constraints-20261007.

Strict F1 preserves the original schema-gated exact slot/value/status definition. Supplementary normalized F1 applies only the existing pinned application normalize_value to both gold and observed values, retains the schema-validity gate and status, and does not repair output. It distinguishes accepted enum capitalization and duration text from omissions and wrong values. Free-text synonyms and missing target qualifiers are not automatically forgiven.

| Track | Decoder | Strict F1 | Normalized F1 | JSON / schema valid | New unmentioned slots | Corrected horizon / 32 | Fully correct state after correction / 32 | Unknown valid/no updates / 32 | Median / p95 (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| natural_component | original | 0.533 | 0.604 | 0.979 / 0.969 | 13 | 32 | 32 | 28 | 5.676 / 13.915 |
| natural_component | constrained | 0.343 | 0.416 | 1.000 / 0.812 | 23 | 12 | 8 | 31 | 7.561 / 13.560 |
| natural_rollout | original | 0.550 | 0.604 | 0.979 / 0.969 | 13 | 32 | 3 | 28 | 5.794 / 13.334 |
| natural_rollout | constrained | 0.384 | 0.418 | 1.000 / 0.802 | 22 | 12 | 0 | 30 | 7.974 / 13.421 |

Local corrected-horizon success and preservation of other prior slots are separate from matching the entire gold accumulated state. A correct correction can coexist with missing initial facts. Unknown tests measure abstention and prior-state retention, not generated clarification-question quality. Unmentioned slots are annotation-relative flags, not independently adjudicated semantic hallucinations.

See summary.csv for weather, inflation, stocks and crypto; analysis.json for full counts, normalized diagnostics, paired latency and 10,000-resample conversation bootstrap intervals; slot-error-audit.csv for every expected/emitted slot. Confidence intervals reflect resampling these development conversations, not unseen users or prompt-selection uncertainty. No fine-tuning decision is automated.
