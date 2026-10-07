# Analysis of the remote v6 handoff at 9fbcadd

## Verification

The supplied handoff matches GitHub branch `experiment/v5-grounded-20260919` at
`9fbcadd0a15b5e6028dbc3e663d4cb6d7cfd140a`. I fetched the remote branch and read its
actual artifacts, rather than relying only on the pasted summary. All **11 result
artifacts** matched their recorded byte lengths and SHA-256 values in Git. The four
compressed reports contain 96 unique calls each: **384 scored extractions**.

The comparison uses the same v6 adapter, original prompt, cases, greedy settings
and prior-state protocol within each track. It tests **generation-time XGrammar
constraints**, not another fine-tuning round. The fixed-field-order restriction
is part of the tested configuration and can itself affect outputs.

## Published rollout results under the original wire policy

| Measurement | Original | Constrained |
|---|---:|---:|
| Outer JSON syntax validity | 94/96 | 96/96 |
| Full old application wire-contract admission | 93/96 | 77/96 |
| Exact slot/value/status F1 | 0.5503 | 0.3842 |
| Application-normalized F1, retaining old admission | 0.6036 | 0.4181 |
| Locally correct horizon changes | 32/32 | 12/32 |
| Horizon changes preserving other prior slots | 32/32 | 8/32 |
| Full correct state after correction | 3/32 | 0/32 |
| Valid unknown turns with no updates | 28/32 | 30/32 |
| Valid unknown turns retaining prior state | 28/32 | 27/32 |
| Median generation latency | 5.794 s | 7.974 s |

The published paired rollout F1 difference is -0.1661, with a 10,000-resample
conversation-bootstrap interval approximately [-0.263, -0.070]. This describes the
original decoder configuration and admission rule on this inspected development
cohort. The mean paired latency increase is 0.616 s; it is a different statistic
from the 2.180 s increase in medians.

The source decision to keep the original baseline is reasonable for that frozen
experiment. Better outer syntax and unknown-turn abstention did not produce better
overall extraction, correction handling or state retention. The readable-content
audit also reports omitted facts and extra correction slots. Those issues warrant
inspection independently of any parser rejection.

## New qualification: admission depends on the parsing policy

Our previously frozen shared policy is
`closed-schema-pydantic-once-20261006-1`, manifest SHA-256
`558581a35a0abc592c59d6f30bc7a2a3c0542368bfee668ac3018800b702189e`.

I applied that parser OFFLINE to the unchanged recorded raw outputs:

| Arm / track | Old wire admission | Shared frozen-policy admission |
|---|---:|---:|
| Original component | 93/96 | 93/96 |
| Original rollout | 93/96 | 93/96 |
| Constrained component | 78/96 | **96/96** |
| Constrained rollout | 77/96 | **96/96** |

The old guard required every candidate string to parse as embedded JSON. The new
common contract accepts ordinary text and JSON-encoded text inside the same string
envelope. For example, an outer update with `candidate_value: "percent"` is ordinary
text; the older guard demanded `candidate_value: "\"percent\""`. These are accepted
as the same decoded text by the new parser. Observed old failures include plain
`create_forecast`, `afternoon air temperature`, `upload`, `percent`, and `humidity_pct`.

Therefore, the claim “19 unusable constrained rollout outputs” is specific to the
old embedded-JSON guard. It must not be carried over as a failure of Pydantic itself
or as the admission rate for our new comparison. This is a retrospective **admission
diagnostic only**: no quality scores, repairs, model calls or new rollout trajectories
were produced. Parser acceptance does not restore omitted target units, choose correct
values/statuses or certify evidence.

## Why a fresh matched comparison is still necessary

The remote studies use the original **32 conversations / 96 calls per track** and
original labels. The fresh OpenAI results use **31 reviewed conversations / 93 calls
per track**. The revised cohort adds the prediction-problem label and excludes one
ambiguous frequency case; 62 later component contexts changed.

**Do not compare historical v6 F1 0.5503 directly with fresh OpenAI F1 0.5086.**
Different cohorts, labels, contexts and parser policies prevent that inference.
Old constrained rollout outputs also cannot simulate the states that would result
when newly admitted outputs are applied.

Recommended next action:

1. Preserve original weights, prompts and decoding as the reference. These studies
   do not by themselves establish that another training round is required.
2. Run the exact BF16 v6 adapter on the 31 reviewed cases with the shared parser,
   once with gold prior state and once with its own predicted state. The prepared
   handoff bundle implements this fresh run and verifies the relevant hashes.
3. Compare the returned reports with the already completed fresh OpenAI reports
   using the policy/context/gold pairing guard and scenario-paired bootstrap.
4. If constrained decoding is studied further, make it a separate fresh arm using
   that same cohort and parser. Investigate field order and correction errors as
   separate changes. Do not adopt it on the basis of JSON syntax alone.

All these cases remain agent-authored, agent-reviewed posthoc development material.
Independent human review and unseen conversations are still needed for stronger
publication claims. No new training, paid API calls or final-label access occurred
in this analysis. The local checkout and its existing edits were preserved.

## Easy explanation

**Professional:** The constrained decoder improved the outer representation but
the old application gate additionally required nested JSON encoding. Changing that
gate alters admission, so we must separate structural representation from semantic
requirement correctness and compare both systems under one frozen protocol.

**Roman Urdu:** Model ne `percent` sahi likha tha, lekin purana checker us ke andar
mazeed JSON quotes mang raha tha. Naya common checker dono tareeqon ko samajhta hai.
Is se missing facts ya ghalat horizon khud theek nahi hotay. Is liye ab dono models
ko aik hi 31-case dataset aur aik hi checking rule par dobara test karna hai.

Sources: the published
[findings](https://github.com/AbdullahUsman0/SLM-FineTuning-Testing/blob/9fbcadd0a15b5e6028dbc3e663d4cb6d7cfd140a/research-checkpoints/v6-constraints-20261007/findings.md),
raw result artifacts verified from Git at that commit, and the adjacent
`REMOTE_HANDOFF_ANALYSIS_9fbcadd.json` admission audit.
