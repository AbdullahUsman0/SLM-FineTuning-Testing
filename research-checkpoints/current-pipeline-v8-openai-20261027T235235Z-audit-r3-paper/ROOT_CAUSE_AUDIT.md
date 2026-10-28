# Root-cause audit of the zero v8 score

The zero was a reproducible operational score for a mismatched interface, not a reliable headline about the adapter. The first runner supplied current native instructions without the update-entry definition or typed slot metadata used by training; only the hosted arm received a machine-readable output schema. The pilot admitted only 2/9 v8 component calls (the admitted calls carried no correct positive tuples), yet a successful positive extraction was not a required technical gate before main execution. That was an evaluation setup/gating mistake.

Evidence, preserved rather than replaced:
- Original 100-case / 996-call archived replay reproduces F1 0.9639530174159578 in both tracks, with no fresh model calls.
- Scorer known-answer calibration on all 101 selected main turns: 257 TP, zero FP/FN, F1 1.0.
- Prior native current-interface pilot, nine component turns: F1 0.0, schema-admitted 2/9.
- Fresh CUDA legacy-interface positive control, same nine pilot turns: F1 0.809524, schema-valid 9/9. It uses the old pinned application and is a loading/behavior control, not a new main arm.
- Fresh CUDA current-application / explicit common-contract probe, same nine pilot turns: F1 1.000000, schema/admission 9/9, TP 21. It retains current application semantics and typed metadata, and exposes the same response schema to both systems.

The corrective r3 study retains the exact main case hash and scorer, the same adapter/base revision, current fpy plus bundled fixes, same decoding limits and authoritative shared budget. It is a post-hoc corrective DEVELOPMENT revision informed by the failed integration, not independent confirmatory evidence. The schema/instructions and all test/source hashes are frozen before new main outcomes. No old predictions are relabeled as fresh inference, no failed records are removed, and no main case is swapped out. The previous zero run remains in the evidence as an integration-failure diagnostic.

Full offline application/evaluator suite: 303 passed with API credentials removed and outbound network blocked. Human annotation remains incomplete; final/private labels are not accessed. No claim of universal superiority or equivalence is authorized.
