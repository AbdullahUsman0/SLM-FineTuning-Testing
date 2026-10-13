# Evidence available for the paper

## Task and conclusion

We evaluated extraction of user requirements for tabular time-series forecasting into a 79-slot schema. The models do not predict weather, inflation, equity prices or cryptocurrency prices in this experiment. Exact non-intent slot/value/status micro F1 measures extraction agreement with annotated requirements, not forecasting accuracy.

The current evidence does **not** establish that the SLM or OpenAI wins. The earlier matched v6–OpenAI intervals include zero, while the later v7 pilot improves over v6 on a different cohort and still fails a behavioral promotion gate. No fresh v7–OpenAI comparison has been run. At the user's request, the current phase uses existing verified artifacts and spends $0 on new paid API calls.

## Historical matched comparison: 31 development conversations

The same 31 conversations cover weather, inflation, stocks and crypto. Each model made 93 extraction calls per track. Component evaluation supplies the correct prior state; rollout accumulates each model's own accepted state. OpenAI Structured F used `gpt-5-mini-2025-08-07`. Local arms used Qwen3.5-2B plus the designated adapters. All scores below use the original frozen comparison policy, unchanged.

| Model | Component exact F1 | Rollout exact F1 | Rollout precision | Rollout recall | Rollout valid JSON | Rollout admitted | Median rollout time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock Qwen3.5-2B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 91/93 | 34/93 | 7.581 s |
| Generalist v5 | 0.4213 | 0.4270 | 0.5429 | 0.3519 | 93/93 | 93/93 | 5.378 s |
| Domain-focused v6 | 0.4818 | 0.4986 | 0.6312 | 0.4120 | 91/93 | 90/93 | 5.262 s |
| OpenAI Structured F | 0.4844 | 0.5086 | 0.4332 | 0.6157 | 93/93 | 93/93 | 2.688 s |

Paired v6-minus-OpenAI exact F1 differences:

| Track | Difference | Paired percentile 95% interval |
|---|---:|---:|
| Component | -0.002611 | [-0.091205, +0.089546] |
| Rollout | -0.010005 | [-0.083586, +0.063873] |

Intervals use 10,000 source-scenario cluster resamples with seed 42, recomputing micro F1 for each resample. Their inclusion of zero supports an inconclusive difference; it does not demonstrate equality, equivalence or noninferiority. V6's higher precision and lower recall show a different error tradeoff. It is incorrect to describe OpenAI as better on every measurement.

Stock's zero primary F1 is strongly affected by output-contract failures and exact label matching. It does not establish that the base model lacks general knowledge. Zero complete-conversation exact matches in these reports also limit claims of conversational reliability.

Source: [original comparison](../policy-openai-slm-matched-20261008/README.md), [paired v6–OpenAI analysis](../policy-openai-slm-matched-20261008/openai-vs-v6.json), and [independent replay receipt](evidence/historical-matched-verification.json). The machine-readable source manifest records `human_review_completed: false`: these were agent-reviewed development cases, not a completed independently human-reviewed final cohort.

## Separate v7 r8 synthetic pilot

V7 is the experimental final-step adapter from a **pilot** of 200 training conversations, 1,040 training rows, one epoch and 65 optimizer steps. The `best-adapter` directory name does not establish behavioral checkpoint selection. This pilot did not complete full-corpus v7 training.

The separate synthetic development comparison has 200 validation conversations, 1,040 calls per arm per track, and a broader domain/template mixture than the historical 31-case study. It includes a seeded-state-repair diagnostic. These prompts and cases differ from the earlier OpenAI comparison.

| Rollout measurement | v6 | v7 pilot |
|---|---:|---:|
| Exact slot/value/status F1 | 0.8337087087 | 0.8784805458 |
| Slot-name F1 | 0.8712462462 | 0.9382260741 |
| Valid JSON | 1040/1040 | 1037/1040 |
| Parser-admitted calls | 970/1040 | 1020/1040 |
| Extra current-turn slot flags | 64 | 72 |
| Correction flag and exact tuples correct | 212/240 | 230/240 |
| Complete accumulated state after correction | 0/240 | 115/240 |
| Valid unknown turn with no updates | 199/200 | 200/200 |
| Valid unknown turn with prior retained | 199/200 | 199/200 |

The v7-minus-v6 rollout difference is **+0.04477184**, with paired 95% interval **[+0.02193610, +0.06628503]**, from 10,000 entity-group resamples over 20 groups, seed 42. This supports an improvement for that pilot cohort. It does not override the predeclared promotion gate: extra flags increased from 64 to 72, so **v7 failed promotion**. Extra flags count observed emissions, including some rejected or repeated fields; they are not all accepted fabricated requirements.

Mixed-language rollout F1 fell from **0.860177 to 0.825342** (108 calls per arm). The dense lifecycle category fell from **0.887902 to 0.807175** (122 calls per arm). These exploratory subgroup findings show why the aggregate improvement does not establish uniform improvement. Roman Urdu and English aggregates improved. [Raw replay diagnostics](evidence/pilot-offline-diagnostics.json) retain counts and exact confusion metrics.

Median rollout times were 6.658 s for v6 and 6.437 s for v7. V7 required three GPU sessions with interruption recovery; these descriptive timings do not establish a controlled speed advantage. See [original pilot report](../v7-r8-pilot-20261009/results/README.md), [paired analysis](../v7-r8-pilot-20261009/results/paired-comparisons.json), and [failure review](FAILURE_REVIEW.md).

**Do not compare v7's 87.85% pilot F1 directly with OpenAI's 50.86% historical F1.** Their cohorts and prompts differ. Such a comparison is confounded and cannot rank the models.

## Threats to validity and remaining work

These are small synthetic/development cohorts, with template dependence and incomplete independent human adjudication. Exact free-text boundaries and status conventions influence F1. Structured-output transport, output limits, local CUDA execution, network/API latency and hosted reasoning differ across providers. Provider-specific scaffolding must be reported; hosted timing is not a hardware-efficiency comparison. Fixed user scripts do not test a full system that generates follow-up questions and adapts to arbitrary user replies. Current results do not establish performance across all 79 slots or all real-world conversations.

The new prospective policy rejects evidence that is not an exact current-message substring. Historical scores above were **not** recomputed under that new policy. Offline failure analysis records whitespace mismatches separately; it does not repair old responses or change old scores.

A private new candidate contains 240 conversations and 1,180 turns, balanced across four domains and 80/10/10 English/Roman Urdu/mixed language. It is agent-authored and pending independent human review. It has twelve shared template groups, not 240 independent human requests. An allowlisted audit found no exact or normalized overlaps against 34,358 accessible training/development utterances, but lexical checks do not certify semantic independence. No models have consumed this cohort, no final freeze exists, and no outcomes are available.

Further comparison requires a new explicit scope/budget decision, actual review decisions bound to the exact cohort, and full registration. The current stopping point is verified existing evidence and offline preparation; another fine-tuning round is not justified by training loss alone.

Suggested paper conclusion: “The domain-focused v6 adapter and the hosted structured-output baseline exhibited different precision–recall tradeoffs on a small matched development cohort, with inconclusive paired differences. A later v7 pilot improved aggregate extraction agreement relative to v6 on a separate synthetic cohort but increased extra-slot emissions and failed its promotion gate. These results motivate controlled behavioral validation rather than a claim of universal superiority.”
