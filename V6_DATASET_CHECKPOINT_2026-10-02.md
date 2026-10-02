# Weather/economics requirements dataset checkpoint

Date: 2026-10-02. Branch: `experiment/v5-grounded-20260919`.
Candidate: `corpus-v6/v6-20261002-r4`.
Status: built and machine verified; human review pending; untrained.

The user explicitly approved 30% weather, 60% economics, and 10% combined
requests. Weather is an in-domain family in this version. The 79-slot schema,
v5 prompts and output contract remain unchanged. The model extracts requirements;
it does not predict weather or market measurements.

## Counts

| Item | Count |
| --- | ---: |
| Source scenarios | 2,000 |
| Weather / economics / combined sources | 600 / 1,200 / 200 |
| Subdomains | 20 |
| Synthetic target definitions | 84 |
| Training / validation / final sources | 1,400 / 300 / 300 |
| Training / validation / final SFT examples | 5,800 / 1,253 / 1,267 |
| Extraction / question examples | 6,320 / 2,000 |
| Total examples | 8,320 |
| Source corrections / confirmed conflicts | 200 / 120 |
| Source short answers / abstention | 160 / 160 |
| Medium / dense source scenarios | 1,000 / 360 |
| Extraction turns with 3–8 updates | 52.53% |
| Extraction turns with 9+ updates | 17.09% |
| Positive slots / schema slots | 78 / 79 |

All required slots have positive examples. `authentication_reference` has no
positive update supervision because the pinned prompt redacts secret references.

The build checked all source facts, canonical types, evidence spans, renderer
equivalence, reducer invariants, selected short-answer slots, prompt coverage,
question contracts and exact/normalized full input duplication. The development
verifier checked 1,700 scenarios and 7,053 examples; final integrity was checked
by streaming hashes, without parsing final labels.

There are zero exact/normalized full model-input duplicates, zero entity groups
crossing splits and zero named template families crossing splits. An exhaustive
representative cross-split Jaccard audit flagged **10,219 pairs**, all unresolved.
This is not a zero semantic leakage claim. Clauses and schema wording are shared.
Entity groups are fictional scopes; common target/instrument definitions recur
across splits. Do not claim unseen real-instrument or country transfer from this split.

## Provenance and files

Builder source prepared at `d8a01e6`; final clean fixture/materialization tests
at `5a96fde`. Normalized builder/prompt SHA-256 hashes are included in the manifest.
The application dependency is pinned to fpy
`04d52c015d1e3ecdefe92b87116f209361509b4b`.

Corpus manifest SHA-256:

```text
0782956c587b5996a2df3f7b799fb0b9e572942b063c8fc1b623ea691fc5aeb9
```

Public artifacts occupy approximately 4.2 MB as deterministic gzip JSONL and
metadata. Every compressed and uncompressed data file has a recorded hash.
Git line-ending conversion is disabled for corpus artifacts.

Final files are retained locally under ignored
`training-runs/v6-sealed/v6-20261002-r4/` and are absent from Git. Their copied
compressed file hashes match the manifest. The procedural seal is neither
cryptographic nor independently custodial; the public builder can regenerate data.

Read `V6_WEATHER_ECONOMICS_PLAN.md`, `corpus-v6/README.md`, and
`corpus-v6/reviews/v6-20261002-r4-sample.md` (20 deterministic training sources).
The sample is a starting point for review, not proof that all labels are correct.
Early local previews r1–r3 were superseded during semantic and wording checks;
only r4 is the published candidate.

## Verification

The full clean suite passes **201 tests**, including miniature mutation checks,
entity/template split guards, deterministic gzip output, artifact tampering,
review-approval rejection and successful development-only materialization.
The verifier reproduces development SFT from validated component cases and
verifies all public plus supplied sealed file hashes.

No model inference, training, live numerical-data download or paid API call was
performed for this dataset. Primary terminology references are listed in the plan.

## Next work

1. Finish the remote v5 base/final-adapter behavioral comparison. Its loading
   smoke missed explicit facts despite almost-zero teacher-forced loss.
2. Human-review this new corpus, resolve overlap flags, and add independent
   colloquial/ambiguous requests before strong generalization claims.
3. After manifest-specific review approval, use prepare-v6-training.py to create
   a new input directory. Preserve the frozen candidate and v5 adapter.
4. Measure token lengths with the pinned model tokenizer; do not assume v5's
   maximum length is sufficient. Run a matched behavioral pilot before full training.
5. Train a fresh specialist adapter from the same stock 2B base, rather than
   continuing the v5 adapter. Keep 10-step complete checkpoint attestations.
6. Compare stock, v5 and specialist on identical v6 validation; report family and
   subdomain metrics. Records supply entity-group cluster IDs for uncertainty.
7. Measure retention on v5 validation. For causal domain-effect claims add
   training-budget-matched controls; v6 has more training examples than v5.
8. Use final labels once only after adapter/output-method selection, with an
   independent custodian and no final feedback to training.
