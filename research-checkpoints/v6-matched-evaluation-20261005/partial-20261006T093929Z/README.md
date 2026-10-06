# Partial matched stock 2B / v5 / v6 evaluation

Snapshot UTC: 20261006T093929Z. The full evaluation is still running. This is not the completed study or a decision to train again.
The worker had saved 1,245 of 2,469 calls at snapshot start; this archive contains 1,239 matched calls from complete scenarios only. In-flight calls and unmatched scenario fragments are excluded.

Both natural tracks are complete: 32 conversations, eight each for weather, inflation, stocks and crypto, with an initial requirement, horizon correction and unknown-information turn. The frozen synthetic validation cohort is incomplete; only the intersection of completed scenario IDs across all three models is compared. Coverage and exact IDs are in analysis.json. Its ordered subset can be biased. Full-study confidence intervals and conclusions are deferred.

Natural component calls use gold prior state. Natural rollout uses each model's own state and the same fixed user script. These are controlled diagnostics, pending independent human annotation review, rather than live-user trials. Source corpus human review and fuzzy-overlap review remain outstanding.

Exact primary F1 requires valid wire schema plus exact slot name, JSON-decoded value and status. Slot-name and slot/value F1 are also reported. Readable-content F1 is a supplementary post hoc diagnostic for valid JSON with schema failures; it never repairs responses or enables state updates. Thus stock operational F1 can be zero despite partially readable content. JSON syntax and schema validity are separate.

New unmentioned slots flag possible inventions against the annotations; repeated unchanged prior facts and wrong values for stated slots are counted separately. Correction metrics distinguish raw flag/value accuracy from common-reducer transition accuracy. Invalid outputs cannot certify safe unknown handling. Unknown-state retention includes intent and slots.

All arms use the same pinned base revision, prompts, BF16 CUDA GPU, greedy generation, 1,024-token limit and serial execution. Native stock was checked against disabled-adapter stock before scoring; warmup calls are excluded. Response times exclude loading and adapter switching. Generation limits and full timing are in analysis.json. No new training, paid APIs or sealed final labels.

| Track | Arm | Exact F1 | Readable value F1 | JSON valid | Schema valid | Extract median / p95 (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| natural_component | base | 0.000 | 0.398 | 0.906 | 0.000 | 8.479 / 34.522 |
| natural_component | v5 | 0.439 | 0.457 | 1.000 | 0.948 | 5.788 / 13.142 |
| natural_component | v6 | 0.533 | 0.546 | 0.979 | 0.969 | 5.587 / 13.312 |
| natural_rollout | base | 0.000 | 0.350 | 0.990 | 0.104 | 7.853 / 16.845 |
| natural_rollout | v5 | 0.445 | 0.463 | 1.000 | 0.948 | 6.020 / 13.741 |
| natural_rollout | v6 | 0.550 | 0.564 | 0.979 | 0.969 | 5.716 / 13.621 |
| frozen_component | base | 0.000 | 0.426 | 0.851 | 0.024 | 15.653 / 35.396 |
| frozen_component | v5 | 0.556 | 0.826 | 1.000 | 0.804 | 16.045 / 36.999 |
| frozen_component | v6 | 0.998 | 0.998 | 1.000 | 1.000 | 16.290 / 38.800 |

Study summary and per-domain comparisons: summary.csv. Raw expected labels beside responses: errors-and-boundary-cases.csv. Nine lossless compressed reports preserve contexts, labels, raw outputs, validation errors and latency. Source/code/model pins: run-manifest.json and reproduction/. Artifact integrity: artifact-hashes.json.

The complete evaluation will be published separately under ../results after all 2,469 calls and integrity checks complete. Training loss is not evidence of conversational quality.
