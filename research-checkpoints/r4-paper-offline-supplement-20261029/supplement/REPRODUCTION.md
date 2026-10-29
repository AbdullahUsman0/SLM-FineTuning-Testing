# Offline reproduction

The original evidence is nested under source-evidence/ and retains its exact SHA.
Code is under code/, outputs under deliverables/, logs under logs/. The copied
public fpy source/test snapshot is under reproduction/fpy (67 tracked files, with
the four original bundled fixes). No .env, model weights, caches or live ledger.

Use the recorded Python/dependency versions; do not replace CUDA packages. Existing
tzdata supplies Asia/Karachi. The two modified bundled test files are included.
The human-review importer is standalone standard-library Python and needs no model.

Prepare a NEW workspace (no inference or network):

    python code/prepare_analysis_workspace.py --output <NEW-folder> --repo <research-checkout-at-required-commit> --offline-deps <existing-isolated-dependencies>

Then run code from that prepared workspace's scripts directory, e.g.:

    python <NEW-folder>/scripts/run_stage.py coverage_and_intervals.py
    python <NEW-folder>/scripts/run_stage.py build_review_kit.py
    python <NEW-folder>/scripts/run_stage.py error_and_oracle_analysis.py
    python <NEW-folder>/scripts/run_stage.py fairness_and_methods.py
    python <NEW-folder>/scripts/run_stage.py write_paper_and_future.py
    python <NEW-folder>/scripts/run_stage.py test_supplement.py

Analysis/form generation does not load models or require weights. The byte-identical
original baseline replay verifier additionally HASHES the adapter at its recorded
path and provider files in the research checkout. On another PC use the already
verified main-PC path mapping or explicitly map only to the same hash-verified
adapter. Do not edit the frozen protocol, substitute weights or claim changed
contexts are a fresh experiment. Run scripts/reproduce_baseline.py only in a new
workspace with the correct paths/dependencies; it performs tests and saved replay,
not inference. No paid stages or spending-ledger access are allowed.

To import real human forms, use deliverables/human-review/import_reviews.py with
the coordinator reference manifest and a NEW output directory. Original labels
are never changed. Blank fields are pending; software does not certify reviewers.
