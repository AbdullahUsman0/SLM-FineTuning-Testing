# Fresh reviewed-policy GPU comparison

All three local GPU arms completed: stock 2B, v5 and v6, with 31 reviewed development conversations / 93 calls per component and rollout track. Total: 558 scored extractions. The exact transferred ZIP was verified and executed without altering its frozen parser, prompts, labels or sources. No training, paid APIs, sealed final labels or historical response-cache reuse.

This is a new experiment under policy `closed-schema-pydantic-once-20261006-1` (`558581a35a0abc592c59d6f30bc7a2a3c0542368bfee668ac3018800b702189e`). Ordinary candidate text and JSON-encoded candidate text are admitted according to the same pinned parser. One ambiguous weather conversation is excluded and later gold contexts changed after review. These scores cannot replace or be directly compared with the earlier 32-case historical studies.

| Track | Arm | Exact slot/value/status F1 | JSON / declared schema / admitted | Unmentioned slots | Raw correction flag+values | Full state after correction | Unknown valid/no updates | Mean / median / p95 seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| component | base | 0.0000 | 88/93 / 14/93 / 14/93 | 83 | 0/31 | 0/31 | 0/31 | 10.189 / 8.121 / 31.307 |
| component | v5 | 0.4213 | 93/93 / 93/93 / 93/93 | 18 | 1/31 | 31/31 | 23/31 | 6.695 / 5.480 / 12.480 |
| component | v6 | 0.4818 | 91/93 / 91/93 / 90/93 | 13 | 22/31 | 31/31 | 27/31 | 6.834 / 5.502 / 13.405 |
| rollout | base | 0.0000 | 91/93 / 34/93 / 34/93 | 104 | 0/31 | 0/31 | 0/31 | 8.253 / 7.581 / 15.160 |
| rollout | v5 | 0.4270 | 93/93 / 93/93 / 93/93 | 18 | 2/31 | 0/31 | 23/31 | 6.597 / 5.378 / 12.009 |
| rollout | v6 | 0.4986 | 91/93 / 91/93 / 90/93 | 13 | 25/31 | 0/31 | 27/31 | 6.550 / 5.262 / 12.795 |

Raw typed corrections and matching the complete accumulated state are different measurements. Component uses reviewed gold prior state; rollout uses each arm’s own accepted prior extraction. Unknown abstention alone does not establish state completeness. Unmentioned-slot flags are relative to reviewed labels and remain subject to human semantic adjudication. summary.csv also separates slot-name F1 from slot-value F1 without status; domain-summary.csv gives every weather/inflation/stocks/crypto subgroup. These supplemental summaries were added during inference, do not alter the frozen primary scorer, and are descriptive diagnostics.

| Pair | Track | F1 difference, right minus left | 95% scenario-bootstrap interval |
| --- | --- | ---: | --- |
| base → v6 | component | +0.4818 | [+0.4157, +0.5480] |
| base → v6 | rollout | +0.4986 | [+0.4334, +0.5650] |
| v5 → v6 | component | +0.0604 | [-0.0166, +0.1412] |
| v5 → v6 | rollout | +0.0716 | [-0.0060, +0.1523] |
| base → v5 | component | +0.4213 | [+0.3678, +0.4749] |
| base → v5 | rollout | +0.4270 | [+0.3725, +0.4807] |

Intervals use 10,000 paired resamples of the 31 source-conversation clusters. They are exploratory, with no correction for multiple comparisons, and do not establish general superiority or unseen-user quality.

The runs were serial, v6 then stock then v5, rather than interleaved in one model session. All used the same RTX A4000, BF16, pinned base revision, offline cache, greedy decoding, four torch threads and one unscored warmup. Cross-run response-time differences can include hardware/session drift and changed output length; do not interpret them as a controlled latency advantage. Latency summaries retain the frozen scorer conventions, including nearest-rank p95. They measure synchronized extraction, exclude loading/warmup, and do not represent an entire live conversation or external data fetching.

The hosted OpenAI arm was not run on this PC and its raw reports are not in this archive. The main PC must use its fresh completed reports with compare-policy-comparison.py to verify policy, gold and component-context matching before making any GPU-versus-OpenAI comparison. Do not compare the historical v6 0.5503 score directly with the fresh hosted score.

This is agent-reviewed posthoc development material. Independent human adjudication and unseen-case validation remain pending. No model or production default was changed.

The results ZIP includes completion.json, run-manifest.json, runtime.json, both raw reports and all 62 completed jobs per arm. It excludes weights, training checkpoints, credentials, caches and dependency packages. See summary.csv, pairwise comparison JSON files and independent-verification.json for the supporting measurements and checks.

A rejected wire response can contain readable facts; low operational F1 must not be interpreted as absence of semantic knowledge. Failed/unparseable outputs are not certified free of invented requirements.
