# Completed v6 specialist training

Training and verification completed **2026-10-04 00:00:31 PDT**:
**726 optimizer steps, two epochs, 73 retained and independently hash-verified checkpoints**.
The final adapter's 372 tensors are finite and exactly equal checkpoint 726.
Selection is the final step; behavioral checkpoint selection remains pending.

| Measurement | Value |
| --- | ---: |
| Training / validation examples | 5,800 / 1,253 |
| Epoch 1 teacher-forced validation loss | 0.000744799617677927 |
| Epoch 2 / standalone validation loss | 0.00006398998812073842 |
| Last logged training window loss, step 725 | 2.80153704807e-05 |
| Standalone validation token accuracy | 0.9999930703059635 |
| Adapter weights + config bytes | 67334101 |
| Wall elapsed from trainer start through verification | 19.173 hours |

The wall time includes Windows interruption, recovery, validation, and verification.
The original attempt reached step 261. Recovery resumed the attested checkpoint
260 with the same source, input hashes and hyperparameters, replaying one step.
The Trainer's aggregate `train_loss` resets its accumulator on resume and divides
by restored global step; it is not an uninterrupted two-epoch average.
Logged token counters also reset on resume; use the preserved per-attempt events
when examining token budgets.

The pinned stock model is Qwen/Qwen3.5-2B at
`15852e8c16360a2fea060d615a32b45270f8a8fc`; this is a fresh specialist adapter.
Training code came from isolated commit `36290d6c28fe65f0014b622045889bf54d5cac48` with
fpy `04d52c015d1e3ecdefe92b87116f209361509b4b`. The corpus is frozen `v6-20261002-r4`,
30% weather / 60% economics / 10% combined, using the existing 79-slot extraction
contract. See [input lock](input-lock.json), [launch configuration](launch-config.json)
and [immutable trainer manifest](run-manifest.json) for all pins and settings.
No completion labels were truncated; maximum training length was 3,385 tokens
under the 3,584 limit. All model input files remain reproducible from the committed corpus.

## What the loading smoke showed

The adapter loaded on CUDA and returned valid extraction JSON for:

> I want to forecast daily maximum temperature in Celsius for the next 7 days using a CSV file.

It produced only `target_description`, containing the entire phrase from
"daily maximum temperature" through "CSV file". It did not return separate
frequency, unit, forecast-horizon or file-format updates. The exact response is in
[adapter-smoke.json](adapter-smoke.json). A successful loading smoke and low
teacher-forced loss therefore do not establish natural-language extraction quality.
This single observation is consistent with a gap between template-like supervision
and natural unquoted phrasing; that is an inference, not a measured generalization score.

Behavioral validation is **pending**. Next evaluation should compare stock, v5 and
v6 on identical v6 development cases with family/subdomain and entity-cluster
metrics, check v5 retention, and freeze independently authored natural-language
requests. Training-budget-matched controls are needed to isolate domain effects.
No numerical weather/economics prediction score or behavioral F1 is claimed here.

## Study records

- [Independent completion audit](completion-audit.json), [completion](completion.json),
  [teacher-forced evaluation](evaluation.json), [trainer state](trainer-state.json).
- [All metrics as CSV](metrics-history.csv), [full training events](training-events.jsonl.gz),
  [full training console](training-console.log.gz), [smoke console](smoke-console.log.gz).
- [Checkpoint attestations](checkpoint-attestations.json), [input token lengths](token-lengths.json),
  [recovery record](interruption-resume-260.json), [run notes](run-notes.md).
- [User authorization](authorization.json), [local source records](source-records.json),
  [publication hashes](artifact-hashes.json), [execution helper source](reproduction/), and
+  [audit/packaging source](reproduction/publish_v6_training_records.py).

The `.gz` logs are lossless; decompress with Python `gzip.open(path, 'rt', encoding='utf-8')`.
JSON values are preserved with normalized LF text; original local-byte hashes are
in `source-records.json`. `artifact-hashes.json` hashes the published Git files.
The execution helpers retain their original local paths for provenance. Adapt
them to a new environment and a new run directory before replaying; do not reuse
the old immutable launch configuration or overwrite the frozen experiment.

Independent human corpus review and 10,219 fuzzy-overlap flags remain unresolved.
Explicit user authorization permitted this experimental run; no formal human-review
approval or zero semantic leakage claim is fabricated. The sealed final set and
paid APIs were unused. The recovery task is now disabled and the trainer has exited.

Adapter location: `D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter`. Weights, checkpoints, optimizer/RNG state,
runtime environments, caches, and secrets stay outside Git. The corpus already
contains the verified development data; it is not duplicated in this publication.
