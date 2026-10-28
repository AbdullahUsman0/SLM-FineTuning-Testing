# Paper evidence checks and remaining limits

Automated checks completed before corrective main inference:
- All current fpy tests plus evaluator controls: 303 passed, API key removed, outbound network disabled except loopback.
- Shared response schema and slot value types visible to both arms; native current context retained; no gold/expected fields inserted in model input.
- Same frozen 20 main conversation IDs and raw extraction gold; approved invalid true-column state carried forward unchanged from r2.
- Primary scoring AST unchanged; perfect known-answer calibration: 257 TP, zero FP/FN over 101 main turns.
- Original 996-call archived v8 replay reproduces all old metrics, including F1 0.963953 in both tracks.
- Fresh CUDA legacy positive control: nine component pilot turns, F1 0.809524, nine valid JSON, eight admitted.
- Fresh current-application explicit-contract control: nine component pilot turns, F1 1.0, all nine admitted, 21 TP. These are disjoint pilot cases, not main results.
- Controls for syntax/schema failures, invalid values, duplicate fields/slots, literal-string double decoding, capped/refused outputs, context drift, journal partial tails and duplicate keys.
- Spending tests for exhaustion, pending/uncertain requests after restart, reasoning output billed once, saved-response recovery, distinct revision namespaces and one cumulative non-renewing ledger.

The corrective r3 pilot and main receipts must be complete before reporting matched main results. Pilot usage must pass affordability and every actual paid dispatch must pass its conservative bound. All prior r2 spending remains charged.

Paper claims that remain out of scope or incomplete:
- Independent human annotation/review and unseen-user generalization are incomplete.
- No sealed/private final labels, new training, additional model-size study, paid judging or live adaptive-question evaluation is authorized.
- The new revision is post-hoc corrective DEVELOPMENT, informed by the failed interface. It is not a preregistered independent final test.
- Exact canonical slot/value/status scoring is stricter than semantic equivalence. No numeric, synonym, wrapper or output repair is introduced to increase the primary score. Invalid responses remain in denominators; readable extra flags do not enumerate all hallucinations.
- Native OpenAI Structured Outputs versus local greedy generation is a decoding difference. This is a comparison of configured systems under a common visible contract.
- Deterministic application recovery/normalization is not raw model credit. Both layers, final conversation state, correction/unknown handling and latency must be reported separately.
- No significance, equivalence or universal-superiority claim is supported by descriptive point estimates on 20 related synthetic conversations.

Push the original failed-configuration evidence and this root-cause audit alongside the corrective main results. Preserve old v8/V8-D and historical hosted cohorts as separate conditions; do not overwrite their scores or replace them with unsupported slide figures.
