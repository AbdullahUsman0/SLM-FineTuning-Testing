# V6 generation-time schema enforcement: completed findings

The constrained candidate failed one or more frozen development-screen criteria. Keep the original decoding baseline; do not adopt this candidate or infer a need for more training from structural validity alone.
No production default or adapter weights were changed. This study tests one fixed decoding configuration, not a new fine-tuning round.

## Matched design

384 new extractions: original versus constrained decoding on 32 three-turn validation conversations (eight each weather, inflation, stocks and crypto), using gold prior states in the component track and each arm’s own accumulated state in rollout. Both arms used the byte-identical original prompt, same final-step v6 adapter, fixed script, unchanged labels, strict application validator and greedy generation settings. Order alternated by conversation in one serial BF16 CUDA session after equal unscored warmups.

Run code was pinned at `2e25b15`; the base is `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; v6 final step 726. Seed 42, batch 1, 1,024 new-token limit, thinking disabled, four torch threads, RTX A4000. `xgrammar==0.2.7` and `apache-tvm-ffi==0.1.14.post1` were added; existing PyTorch, Transformers and other model libraries were unchanged. All scored inference used the local offline cache.

The decoder compiles the existing exported structural JSON Schema, constructs one fresh matcher per generation and masks the full 248,320-token model vocabulary. It uses the actual model generation stop ID 248044. XGrammar’s CPU matcher creates the bitmask and its explicit `torch_native` backend masks CUDA logits; model weights remain on CUDA. Grammar setup/compilation time is separately recorded in execution evidence. There is no parsing repair, retry or per-case hint.

Fixed field order is an explicit additional restriction: arbitrary-order mode accepted duplicate object keys in unscored contract tests. The fixed-order configuration passed those tests before protocol freeze. This restriction can affect generation, so the comparison isolates this complete decoder configuration rather than attributing every change to extra-key rejection alone. The grammar treats `candidate_value` as a JSON string; correctness of the JSON encoded inside that string, uniqueness of slot IDs and semantic grounding remain application checks.

## Results

Strict F1 retains schema gating and exact slot names, values and status. Supplementary normalized F1 applies only the existing pinned application normalizer to both sides, retains the same gate and status, and performs no output repair or arbitrary semantic paraphrase matching. “Schema valid” below means acceptance by the entire unchanged application wire contract, including inner JSON-text validation. It is stronger than outer JSON grammar compliance.

| Track | Decoder | Strict F1 | Normalized F1 | Strict JSON / application schema valid | Raw typed corrections | Local normalized horizon correct | Full state after correction | Unknown valid, no updates | New unmentioned slots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| component | original | 0.533 | 0.604 | 94/96 / 93/96 | 23/32 | 32/32 | 32/32 | 28/32 | 13 |
| component | constrained | 0.343 | 0.416 | 96/96 / 78/96 | 6/32 | 12/32 | 8/32 | 31/32 | 23 |
| rollout | original | 0.550 | 0.604 | 94/96 / 93/96 | 26/32 | 32/32 | 3/32 | 28/32 | 13 |
| rollout | constrained | 0.384 | 0.418 | 96/96 / 77/96 | 8/32 | 12/32 | 0/32 | 30/32 | 22 |

A locally correct horizon correction can coexist with incomplete initial requirements. Component correction starts with gold prior state, so its full-state success cannot establish reliable initial extraction. Unknown tests measure valid abstention and prior-state retention, not generated clarification-question quality. New unmentioned slots are annotation-relative flags requiring semantic human review.

| Rollout domain | Original strict F1 | Constrained strict F1 | Original application schema valid | Constrained application schema valid |
| --- | ---: | ---: | ---: | ---: |
| weather | 0.622 | 0.282 | 23/24 | 18/24 |
| inflation | 0.593 | 0.568 | 23/24 | 21/24 |
| stocks | 0.420 | 0.315 | 23/24 | 17/24 |
| crypto | 0.558 | 0.370 | 24/24 | 21/24 |

| Track | Candidate minus original F1 | 95% conversation-bootstrap interval | Paired mean latency difference, seconds |
| --- | ---: | --- | ---: |
| component | -0.189 | [-0.297, -0.080] | +0.549 |
| rollout | -0.166 | [-0.263, -0.070] | +0.616 |

Intervals use 10,000 resamples of the 32 authored conversation clusters. They describe uncertainty within this development cohort; they do not account for unseen users or configuration-selection uncertainty.

## Failures and independent grammar checks

- `original-natural_component`: 3 application-invalid calls; observed categories: `{"duplicate_slot_ids": 1, "strict_json_failure": 2}`. Categories may overlap.
- `constrained-natural_component`: 18 application-invalid calls; observed categories: `{"candidate_value_not_valid_json_text": 18}`. Categories may overlap.
- `original-natural_rollout`: 3 application-invalid calls; observed categories: `{"duplicate_slot_ids": 1, "strict_json_failure": 2}`. Categories may overlap.
- `constrained-natural_rollout`: 19 application-invalid calls; observed categories: `{"candidate_value_not_valid_json_text": 19}`. Categories may overlap.

Independent character-level checks against the same frozen grammar, using no model weights or repair:

- `natural_component`: `{"complete_grammar_valid": 96}`.
- `natural_rollout`: `{"complete_grammar_valid": 96}`.

Every failed call remains in the scored denominator. See failure-review.json for the complete raw trace of each invalid response. Content-assessment.csv separates readable slot omissions from strict JSON failures and normalization-only differences. A grammar-valid object can still omit stated facts, invent facts or contain invalid inner JSON; only the unchanged application gate determines whether it is applied.

All 18 constrained component failures and 19 constrained rollout failures were invalid JSON text inside candidate_value, despite valid outer objects. Component failures comprised 14 initial turns and four corrections; rollout failures comprised 14 initial turns and five corrections. Strict JSON validity therefore reached 96/96 on each track, while application validity declined.

The readable-content audit finds 55 initial slot omissions in each constrained track, including 24 missing target units, ten initial horizons, nine observation frequencies and four timestamp columns. These counts do not certify schema-invalid emissions as usable. Rollout also has 16 new-unmentioned-slot flags on correction turns and six on unknown turns; some corrections add target columns, units or dates rather than only changing the horizon. Local corrected-horizon success fell from 32/32 to 12/32, and horizon correction with all other prior slots preserved fell from 32/32 to 8/32. Semantic error flags remain subject to human adjudication.

## Response time

| Track | Decoder | Mean / median / p95 seconds | Generation-limit calls |
| --- | --- | ---: | ---: |
| component | original | 6.894 / 5.676 / 13.915 | 0 |
| component | constrained | 7.443 / 7.561 / 13.560 | 0 |
| rollout | original | 6.849 / 5.794 / 13.334 | 0 |
| rollout | constrained | 7.465 / 7.974 / 13.421 | 0 |

Timing includes warmed local extraction and synchronized CUDA execution, including matcher/mask overhead. It excludes one-time model loading and compilation and is not complete live-conversation or external-data API latency. There is no cross-session speed claim.

## Verification and decision

All 11 hashed result artifacts independently matched disk SHA-256, size and published Git blobs at `b53e915c578be88c1d204de2e9d7c85196de8d4f`. Four gzip files decode losslessly to the original raw reports, with 96 unique matched records each. Current-arm gold labels and component contexts match; historical labels also match. Original-control repeat checks against the previous study: `{"natural_component": {"context": 96, "expected": 96, "prediction_valid": 96, "raw_output": 96}, "natural_rollout": {"context": 96, "expected": 96, "prediction_valid": 96, "raw_output": 96}}`. All calls record one generation and zero retries. Adapter SHA-256 values were independently verified unchanged after inference.

The shared evaluator sanitizes away the optional per-token mask-step trace extension; the per-call backend and model-call counts are retained. The pinned decoder code and independent output-language checks are preserved, without claiming a token-by-token execution trace. No raw output, label or primary score was repaired or changed.

The first supplementary analysis stopped because normalization of four invented forecast_start date emissions returned Python datetime values that the audit did not serialize. A separate analysis checkout at 2272807 uses the existing state-snapshot ISO serializer. All 24 relevant tests passed, and supplementary metrics on all 768 records from the two previous prompt studies remained unchanged. Original inference at 2e25b15 and the incomplete first analysis were retained; no generation was repeated. The raw reports and strict metric implementation are unchanged. See [review/analysis-recovery.json](review/analysis-recovery.json); reproduction preserves both frozen inference-era and executed analysis code.

The frozen criterion outcomes are recorded in [review/criterion-assessment.json](review/criterion-assessment.json). Keep the original control as the reference. If further decoding work is pursued, distinguish enforcing the inner JSON-text contract from improving the extraction of facts, and evaluate both with the same application gate. This experiment does not establish that another fine-tuning round is necessary.

These are agent-authored development cases already inspected in earlier studies. Independent human adjudication and unseen conversations remain pending. No training, paid APIs or sealed final-label access occurred. Default interactive inference is unchanged.

See [results/README.md](results/README.md), [results/summary.csv](results/summary.csv) and [results/analysis.json](results/analysis.json) for all metrics, cases and raw outputs. Checks and errors are in [review/independent-verification.json](review/independent-verification.json), [review/failure-review.json](review/failure-review.json) and [review/content-assessment.csv](review/content-assessment.csv). The frozen [protocol](../../evaluation/v6-constraints-20261007/protocol.json) records exact settings and limitations. Scripts under review retain machine-specific local paths as provenance, not portable inference instructions.

Primary references for the backend: [XGrammar installation](https://xgrammar.mlc.ai/docs/latest/start/installation.html), [engine integration](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html). Exact installed 0.2.7 package code and the locally passing tests govern this run.
