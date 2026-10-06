# Focused v6 schema reminder: reject the candidate

The single frozen format-only prompt addition failed. Retain the original prompt as the baseline; adopt neither this schema reminder nor the earlier revised prompt as a proven replacement. No training or production default change was made. The result supports testing generation-time enforcement of the existing output schema next, with the same strict validator and a matched semantic evaluation. Such enforcement has not been installed or evaluated here and would not by itself prove correct slot values.

## What was tested

384 new scored extractions: revised control versus `schema_fixed`, each on 32 three-turn conversations in weather, inflation, stocks and crypto, under both gold-prior component contexts and each arm’s own accumulated rollout state. The 1,165-character addition specifies allowed keys, separate slot updates and no duplicates; its prefix is byte-identical to the prior revised prompt. This was one fixed candidate, frozen before inference. No examples with case answers were added. Labels, weights, validator and greedy decoding remained unchanged. Arms alternated order by conversation after equal warmups in one BF16 CUDA session.

Base: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; final-step v6 adapter step 726. Seed 42, batch 1, 1,024 new-token limit, thinking disabled, RTX A4000. Run code pinned at `161f961`; raw reports published at `5b73d8d1a120b7a10b67011550eee33e3897e0f3`. The original prompt below is a historical reference from the prior study, not a third same-session arm.

## Results

Strict F1 measures exact slot name, value and status and retains the schema gate. Normalized F1 applies the pinned application normalizer to both sides, retains the same gate and status, and does not repair outputs or forgive arbitrary free-text differences. Each row has 96 calls; correction and unknown tests have 32 cases.

| Track | Prompt | Strict F1 | Normalized F1 | Strict JSON / schema valid | Raw typed corrections | Local normalized horizon correct | Full state after correction | Unknown valid, no updates | New unmentioned slots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| component | Original (historical) | 0.533 | 0.604 | 94/96 / 93/96 | 23/32 | 32/32 | 32/32 | 28/32 | 13 |
| component | Revised (current control) | 0.579 | 0.670 | 94/96 / 87/96 | 20/32 | 32/32 | 32/32 | 32/32 | 0 |
| component | Schema reminder (candidate) | 0.154 | 0.241 | 88/96 / 67/96 | 18/32 | 32/32 | 32/32 | 32/32 | 0 |
| rollout | Original (historical) | 0.550 | 0.604 | 94/96 / 93/96 | 26/32 | 32/32 | 3/32 | 28/32 | 13 |
| rollout | Revised (current control) | 0.602 | 0.670 | 94/96 / 87/96 | 24/32 | 32/32 | 8/32 | 32/32 | 0 |
| rollout | Schema reminder (candidate) | 0.228 | 0.241 | 88/96 / 67/96 | 30/32 | 32/32 | 0/32 | 32/32 | 0 |

Component full-state correction starts with gold prior state, so its 32/32 score does not demonstrate reliable capture of initial facts. In rollout, the candidate correctly changed the local horizon in all cases and preserved all other prior slots, but **0/32** resulting states matched the complete gold requirements. Prior slots can be missing because the first response was rejected. Unknown abstention likewise does not establish a complete state or good generated clarification questions.

The candidate missed the frozen success criteria: schema validity was 67/96 on each track, below both the original reference target of 93/96 and revised control’s 87/96. Normalized F1 fell from 0.670 to 0.241; rollout full-state accuracy fell from 8/32 to 0/32. It retained 32/32 valid unknown abstentions and zero annotation-relative new unmentioned slots, which is insufficient to offset these failures.

| Rollout domain | Original (historical) F1 | Revised control F1 | Schema reminder F1 | Revised schema valid | Reminder schema valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| weather | 0.622 | 0.682 | 0.182 | 21/24 | 16/24 |
| inflation | 0.593 | 0.449 | 0.238 | 20/24 | 17/24 |
| stocks | 0.420 | 0.769 | 0.211 | 24/24 | 16/24 |
| crypto | 0.558 | 0.494 | 0.289 | 22/24 | 18/24 |

The primary F1 difference, candidate minus revised control, is -0.424 in component (95% conversation-bootstrap interval [-0.554, -0.289]) and -0.373 in rollout ([-0.495, -0.245]). Both intervals exclude zero within this development cohort. These 10,000-resample intervals do not measure unseen-user generalization or prompt-selection uncertainty.

## Failure mechanism

Every invalid response occurred on the initial turn. On each track, the revised control had nine failures: two duplicated JSON object keys, six updates with an unsupported `type` field, and one merged update with extra `forecast_horizon`, `frequency` and `target_unit` keys. The reminder had 29 failures: sixteen responses added `type`, five added `update_type`, and eight duplicated object keys (seven `evidence_text`, one `candidate_value`). The format instructions did not prevent the behavior they described. None of the 384 calls hit the generation limit.

“Strict JSON” includes rejecting duplicate object keys. These strings can be accepted by a permissive parser that discards an earlier key; they remain invalid under the frozen contract. The supplementary readable-content audit does not repair or apply them. It finds 24 readable slot omissions and 11 expected-slot entries unassessable due to duplicate-key JSON in the control, versus 15 readable omissions and 45 unassessable entries in the candidate. The smaller readable omission count is not evidence of improvement because substantially more candidate content is unassessable. Different free-text values remain unadjudicated semantic differences, rather than automatically counted as concept errors.

## Timing and verification

| Current-session arm | Component median / p95 seconds | Rollout median / p95 seconds |
| --- | ---: | ---: |
| revised | 5.464 / 14.630 | 5.384 / 14.692 |
| schema_fixed | 5.551 / 15.349 | 5.389 / 16.140 |

The candidate’s paired mean latency increased by 0.434 seconds in component and 0.508 seconds in rollout. Medians alone conceal this tail increase. No same-session speed claim is made against the historical original prompt. Timings measure local synchronized extraction, not a complete live conversation or external API calls.

Independent verification checked all 11 hashed artifacts against disk, size and published Git blobs; all four gzip reports decode losslessly to the raw reports and contain 96 unique matched records. Gold labels match across arms and historical references; gold component contexts match across current arms. **All 192 revised-control raw outputs, contexts, labels and validity outcomes exactly repeated the prior study.** Adapter model/config SHA-256 values independently matched the frozen protocol after inference. Optional `model_calls` and `retry_count` record fields are null; the pinned runner has one extraction per turn and no repair/retry loop.

The frozen protocol accidentally retained an old descriptive order string. Its arms field and executed alternating order were correct. The original bytes were retained, with the correction documented in [protocol-errata.json](../../evaluation/v6-schema-fix-20261006/protocol-errata.json).

All cases were agent-authored and reviewed by the agent; independent human review is pending. These are the same development conversations already inspected to design the prompts. No unseen-case claim, sealed final-label access, paid API use or further training occurred. Keep these failed results as evidence and finish human case/error review before drawing broader model-quality or training conclusions.

Raw outputs, metrics, timings and exact run code are in [results/README.md](results/README.md), [results/summary.csv](results/summary.csv) and [results/analysis.json](results/analysis.json). Independent integrity checks and every structurally invalid call are in [review/independent-verification.json](review/independent-verification.json), [review/failure-review.json](review/failure-review.json) and [review/duplicate-key-review.json](review/duplicate-key-review.json). The readable-content classification is in [review/content-assessment.csv](review/content-assessment.csv). Review scripts are preserved as provenance; they contain local run paths and are not standalone portable inference commands.
