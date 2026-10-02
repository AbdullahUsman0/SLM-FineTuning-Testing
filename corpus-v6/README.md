# Weather and economics v6 candidate

The first candidate is `v6-20261002-r2`: 2,000 source scenarios, 8,320 SFT
examples, 20 subdomains, and the unchanged pinned 79-slot schema.

Agreed mixture: 600 weather scenarios, 1,200 economics scenarios, and 200
weather–economics combined scenarios. Entity-group splits are 1,400 / 300 / 300
training / validation / final. Final labels are absent from this public package.

Read [the protocol](../V6_WEATHER_ECONOMICS_PLAN.md) and the generated
[review sample](reviews/v6-20261002-r2-sample.md).
The candidate is **machine validated, human review pending, untrained**.
v5's prior user-confirmed review does not constitute review of these new cases.

## Files

- `manifest.json` and `manifest.sha256.json`: frozen integrity/provenance.
- `taxonomy.json`: domain targets, units and scope families.
- `split-freeze.json`: entity/template assignments fixed before rendering.
- `splits/train.jsonl.gz`, `splits/validation.jsonl.gz`: component evaluation cases.
- `splits/smoke.jsonl.gz`: one validation case per subdomain, not an independent split.
- `sft/train.jsonl.gz`, `sft/validation.jsonl.gz`: model-ready messages.
- `statistics.json`: actual sizes, slot and density coverage.
- `overlap-summary.json`, `overlap-review-queue.jsonl.gz`: representative fuzzy overlap audit.
- `review-queue.jsonl.gz`: case IDs and required review reasons; no final labels.
- `review-sample.jsonl.gz`: one training source per subdomain.

Gzip is deterministic and includes compressed plus raw-content hashes. The
training materializer writes verified raw JSONL outside this immutable directory.
Public schema/prompt clauses are shared; zero semantic leakage is not claimed.
There are 78 positive slots; `authentication_reference` is excluded because the
prompt redacts secret references. All required slots have positive coverage.

```text
python scripts/verify-corpus-v6.py --output corpus-v6/v6-20261002-r2
```

To rebuild a **new** immutable revision, with the pinned sibling fpy checkout:

```text
python scripts/build-corpus-v6.py --output corpus-v6/v6-NEW-REVISION --sealed-output training-runs/v6-sealed/v6-NEW-REVISION --per-domain 100
```

To materialize inputs after recorded review, see the protocol. No trainer, model
download or paid API call is started by these corpus tools.
