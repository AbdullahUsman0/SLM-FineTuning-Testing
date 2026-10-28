# Paper-use qualification after the saved-output audit

The published OpenAI component precision/recall/F1 of 43.73%/66.54%/52.78%
reproduces under the frozen r3 experimental scoring contract. It is not an
end-to-end updated-application accuracy estimate. Do not present that row, or
the v8-versus-OpenAI gap, as validated general production performance.

The component arithmetic is 171 true positives, 220 false positives and 86
false negatives: `F1 = 342 / (342 + 220 + 86) = 0.5277778`. Reproducing that
arithmetic does not establish that its admission/tuple definition measures
production-pipeline success or semantic extraction accuracy.

The audit did not change predictions, gold, primary scoring, the original ZIP
or its checksum. It made no model/API calls. `paper-validity-audit.json` and
`audit_saved_outputs.py` record the reproducible checks.

## Confirmed qualifications

1. **Parser fidelity differs.** The native `ExtractorResult` Pydantic parser
   accepts 101/101 saved OpenAI responses in each track. The experimental
   parser accepts 84/101 component and 79/101 rollout responses. The experiment
   additionally rejects duplicate slot IDs and candidate strings that are not
   valid inner JSON. The native decoder retains non-JSON strings. For example,
   plain `operating_cash_flow` is accepted by the native decoder but causes the
   experimental whole response to be rejected. Native parsing alone does not
   establish evidence validity or correct resulting state.

2. **Some penalties concern representation, not invented facts.** The current
   reducer ignores updates with `unmentioned` or `dont_care` status. OpenAI
   emits 81 such non-intent entries in the component track and 63 in rollout.
   The raw exact metric counts them as predictions, while this gold contains
   no such updates. Therefore these entries must not all be described as
   invented actionable requirements. Other wrong/unsupported positive updates
   remain; the audit does not establish that native performance would be high.

3. **Unknown handling needs separate definitions.** In 14/20 OpenAI component
   unknown turns (13/20 rollout), the response's non-intent updates consist
   solely of statuses the reducer ignores. The original raw-omission metric
   requires an empty list and records 0/20. Neither figure by itself establishes
   correct state preservation: deterministic recovery can still change state.

4. **The experiment is an adapted, fixed-script development condition.** Both
   models receive the augmented common schema and v8 annotation instructions,
   rather than the unmodified native-provider prompt. Assistant question
   context is removed between scripted turns. Native Structured Outputs versus
   local greedy decoding also differs. The 20 inspected synthetic conversations
   are related to the adapter's development domain/templates. This is not an
   independent adaptive-interview or general model-capability evaluation.

All 404 saved raw-score records and aggregate scores reproduce. Whole-response
rejection removes 16 otherwise exact tuple matches from OpenAI component and
15 from rollout; this is a measured contribution, not an explanation of the
entire gap. There is no replacement or selectively relaxed headline F1 here.

The remote fpy `origin/main` was fetched for this audit and remained at
`9bc2bd18c495514efa79a4371be1d706b9fc3824`; the experiment includes the verified
four-file bundled fixes. Unpublished changes on another PC are not verified.

## Instructions for the existing paper

Hold the main claim that these rows measure the updated production pipeline or
establish v8's general superiority. The original rows may be retained as an
explicitly named **strict-contract, exact raw-extraction development diagnostic**
with these qualifications. Report application-state results separately and
identify their configured parser and fixed-script assumptions.

Before a production-pipeline comparison is used as primary paper evidence,
freeze and test the actual native provider's prompt, parsing and context path;
prespecify actionable-update, harmless no-op, correction/unknown, semantic versus
exact-value and state metrics; and obtain independent review of the gold and a
representative set of disagreements for both models. Any changed experiment
needs a new version. Replaying saved outputs under another parser is post-hoc
analysis and cannot substitute for fresh rollout outputs on changed states.

Software tests and exact numerical replay establish implementation consistency.
They do not establish that a metric measures the intended research claim.
Independent human review remains incomplete. No further paid calls, training,
sealed-label access or budget renewal is authorized by this addendum.
