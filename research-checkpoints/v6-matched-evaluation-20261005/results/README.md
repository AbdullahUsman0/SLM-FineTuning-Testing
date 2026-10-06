# Matched stock 2B / v5 / v6 evaluation

All 2,469 scored calls completed on the same RTX A4000, BF16 CUDA, pinned Qwen3.5-2B revision, greedy decoding and 1,024-token cap. Stock adapters were disabled and checked against native stock output. Arms rotated per scenario; serial latency follows per-arm warmup. No JSON repairs, retries, training, paid APIs or sealed final labels.

The frozen component cohort contains all 150 v6 validation scenarios in the requested domains (90 weather, 15 inflation, 15 stocks, 30 crypto). Its language is synthetic and template-based. The 32 natural conversations (eight per domain) were authored before inference, with initial facts, an explicit horizon correction, and unknown information. They are exploratory diagnostics, pending independent human review. Natural component calls use gold prior state; rollout calls use the model’s own state with a fixed user script. These are not live-user trials.

Primary F1 requires exact slot, JSON-decoded value and status. CSV also reports slot-name and slot/value F1 without the status requirement. Free-text values follow the exact wording required by the prompt. Intent is scored separately. Unexpected new slot emissions flag possible invented requirements against the annotations, not independently adjudicated semantic hallucinations. They are separated from repeated prior facts, wrong values of stated slots and unsupported evidence. Invalid outputs fail extraction and cannot certify unknown-information handling. JSON syntax validity and field/type schema validity are separate from canonical slot-ID correctness; the table uses extraction calls only, while question timing/validity are separate in analysis.json. Timing excludes model load. Generation-limit hits are disclosed in analysis.json.

Supplementary readable-content scores inspect structurally readable slot IDs and decoded values in valid JSON even when the response fails the wire schema. They were added after initial calls to distinguish formatting failures from content omissions. These descriptive scores do not repair responses or make them usable by the pipeline; primary scores and rollout behavior are unchanged.

| Track | Model | Exact F1 | JSON valid | Schema valid | New unmentioned slots | Corrections exact | Unknown valid/no updates | Extract median / p95 (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| natural_component | base | 0.000 | 0.906 | 0.000 | 114 | 0.000 | 0.000 | 8.479 / 34.522 |
| natural_component | v5 | 0.439 | 1.000 | 0.948 | 21 | 0.031 | 0.750 | 5.788 / 13.142 |
| natural_component | v6 | 0.533 | 0.979 | 0.969 | 13 | 0.719 | 0.875 | 5.587 / 13.312 |
| natural_rollout | base | 0.000 | 0.990 | 0.104 | 143 | 0.000 | 0.125 | 7.853 / 16.845 |
| natural_rollout | v5 | 0.445 | 1.000 | 0.948 | 21 | 0.062 | 0.750 | 6.020 / 13.741 |
| natural_rollout | v6 | 0.550 | 0.979 | 0.969 | 13 | 0.812 | 0.875 | 5.716 / 13.621 |
| frozen_component | base | 0.000 | 0.802 | 0.015 | 943 | 0.000 | — | 17.372 / 38.618 |
| frozen_component | v5 | 0.502 | 0.998 | 0.742 | 265 | 0.588 | — | 17.389 / 38.071 |
| frozen_component | v6 | 0.999 | 1.000 | 1.000 | 0 | 1.000 | — | 16.563 / 40.778 |

See summary.csv for every domain, analysis.json for full metrics and paired 10,000-resample cluster bootstrap intervals, and errors-and-boundary-cases.csv for raw predictions beside expected labels. The nine compressed reports preserve every model response and its context losslessly. Unequal domain sizes require per-domain interpretation; aggregate F1 is micro-weighted. Bootstrap intervals address sampling variability in this cohort, not annotation bias or real-user generalization.

Correction metrics separate the flag, raw typed values, and the result of the common pinned state reducer. Unknown-state retention includes established intent as well as slot values/statuses. No application recovery or output repair is used. The source v6 corpus also has pending human review and unresolved fuzzy-overlap flags; high synthetic validation scores do not establish natural-language transfer.

No new fine-tuning round has been started. Review the natural conversation failures and independent annotations before choosing whether to change the dataset, prompts, or training. Teacher-forced training loss is not used as evidence of behavioral success.
