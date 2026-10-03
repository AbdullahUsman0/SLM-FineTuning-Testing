# Interpretation of the v5 base versus final-adapter result

The final-step adapter learned this synthetic component task very well. Exact
non-intent micro F1 rose from 0.000000 to 0.999454; the paired source-cluster
95% interval for the improvement is [0.998586, 1.000000]. It returned valid
JSON and valid output schemas on all 948 calls, preserved all 27 explicit
correction tuples, and matched every clarification reference exactly.

There were 3,664 correct non-intent tuples, two false positives, and two false
negatives. Two of 231 scenarios contained a mistake; 715/717 reducer transitions
were exact. Forbidden-slot emissions occurred on two extraction calls (0.279%).
The loading smoke was therefore unrepresentative of this controlled validation
cohort, but its omissions still motivate independent natural-language tests.

Stock base's zero F1 is a result under the strict schema and tuple contract.
Only 21/717 extraction calls parsed successfully, versus 231/231 question calls.
The scorer gives invalid predictions no true positives and counts their readable
emissions as false positives. Successful extraction also failed exact slot/value/
status matching. This is not a claim that base has zero language understanding.

The two adapter failures were:

- `v5-20260919-r1/sales_demand/078`: assigned the unit "orders" to `target_description` instead of `target_unit`.
- `v5-20260919-r1/energy/024`: assigned the 26-day forecast generation schedule to observation `frequency` instead of `prediction_frequency`.

This is teacher-forced component evaluation: later turns receive gold context,
and questions receive oracle final slots. Validation was used for teacher-forced
loss during training, and the corpus uses shared synthetic clause machinery.
Neither this result nor the extremely low loss proves generalization to informal
requests, unseen real entities, or an end-to-end conversation. No sealed final
labels were accessed and no paid API was used.

The v6 experiment should train a fresh adapter from the pinned stock base, then
compare stock, v5 and v6 on the same v6 development cases, report family/subdomain
metrics, and measure v5 retention. Human-authored ambiguous, colloquial and
multi-fact cases are needed before deployment claims. The v6 shared-clause and
unresolved-overlap limitations must accompany its scores. V6 also has more
training examples, so practical improvement would not isolate a causal domain
effect without training-budget-matched controls.

Publication initially stopped on an exporter KeyError for a base-only emitted
slot (`holdout_window`). Both evaluations were already complete. The repaired
exporter takes the union of slot IDs and assigns zero counts to absent slots;
raw outputs, gold, scoring and bootstrap inputs were preserved.
