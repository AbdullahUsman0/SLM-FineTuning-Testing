# Fresh stock/v6/v7 r8 pilot comparison

All 6,240 scored extractions completed: 200 identical validation conversations, 1,040 calls per track per arm. All used the v7 prompt and same frozen wire policy. No paid APIs or sealed final labels.

| Arm | Track | Exact slot/value/status F1 | JSON valid | Admitted | Unmentioned flags | Median seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stock | component | 0.0015 | 832/1040 | 181/1040 | 2249 | 13.33 |
| v6 | component | 0.8192 | 1040/1040 | 983/1040 | 53 | 6.64 |
| v7 | component | 0.8775 | 1037/1040 | 1020/1040 | 71 | 6.71 |
| stock | rollout | 0.0007 | 994/1040 | 556/1040 | 1121 | 7.64 |
| v6 | rollout | 0.8337 | 1040/1040 | 970/1040 | 64 | 6.66 |
| v7 | rollout | 0.8785 | 1037/1040 | 1020/1040 | 72 | 6.44 |

v6 to v7 rollout F1 delta: +0.0448; 95% entity-group bootstrap interval [+0.0219, +0.0663].

The 20 source groups contain ten related conversations each. Intervals resample groups. These are synthetic development cases with shared templates; they do not establish unseen-user quality. Filename-language coverage is uneven. All failed outputs remain in denominators.

Rollout uses accepted predicted state and the original declared synthetic repair interventions. The repair category and rollout excluding repairs are reported separately. Component priors match the actual validation SFT. Correction, unknown, evidence and full-state metrics are in summary.csv; subgroup metrics are in stratified-summary.json. Supplemental numeric-canonical diagnostics treat JSON numbers 1 and 1.0 equally while preserving strings and booleans; the frozen primary exact-JSON tuple scorer and its intervals remain unchanged.

This evaluates extraction on fixed user scripts, not generated clarification questions or a live complete product conversation. The interactive V7 provider still requires JSON-encoded candidate strings; the shared comparison policy also admits ordinary candidate text. Benchmark admission therefore does not certify identical interactive-runtime admission.

The final-step pilot adapter remains experimental. No larger training round starts automatically. Quantitative criteria, human failure review and an explicit next-round decision are required. Loss alone cannot select the adapter. Latency is serial-arm synchronized extraction, excluding warmup/loading; session drift and output length can affect comparisons.

The ZIP includes immutable raw jobs, both raw reports and manifests for every arm, diagnostics, paired comparisons, training completion/provenance and the frozen continuation code. It excludes model weights, optimizer files, credentials and caches. A separate verified adapter package is placed in Downloads for local testing.

Publication recovery: a missing standard-library import in the ZIP member verifier was repaired in a versioned copy of the packager. Frozen evaluation sources, labels, raw outputs, scores and bootstrap intervals remain unchanged; original failed artifacts were preserved locally. See packaging-recovery.json. V7 spans three GPU sessions after two interruption recoveries, with three unscored warmups; latency comparisons are descriptive and do not establish a controlled advantage.
