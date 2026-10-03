# V5 training audit and matched base versus final-adapter validation

Status: validation is complete. Read the [comparison and study files](results/README.md)
and [interpretation of the results](ANALYSIS.md).

The completed training run reached optimizer step 556 and epoch 2.0. Its final
adapter matches checkpoint 556 exactly; all 56 retained checkpoints passed the
independent hash audit. Read the [training summary](training-summary.md) and
[completion audit](training-audit.json), including the smoke response that omitted
several stated facts. Loading success and low teacher-forced loss are not evidence
of extraction quality.

The requested comparison uses all 231 frozen development validation scenarios:
717 extraction calls and 231 question calls per model. The stock base and final
adapter use Qwen/Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, identical v5 prompts, BF16 CUDA on an
RTX A4000, greedy decoding, 1,024 generated tokens, and no retry or repair.
The frozen cohort call-plan hash is
`7ce103ded235c5c9558868c9b99bf1519552fadbdcb50c25d4dd73cef19a9069`.

This is a component evaluation with gold context at each turn. It compares one
final-step adapter, rather than selecting among retained checkpoints. Development
validation was also used for teacher-forced validation loss during training.
The sealed final test and paid API evaluation are outside this run. Independent
formal corpus review remains pending.

Each completed scenario is saved immutably. An interrupted scenario is retained
and replayed. A hidden local supervisor runs base followed by adapter, packages
the completed reports, checks matched runtime/provenance and computes a paired
source-cluster bootstrap (10,000 samples, seed 42). It then commits and pushes
the study files to this branch as explicitly requested by the user. A local
scheduled task permits recovery after a Windows login; it disables itself after
the remote publication is verified.

Completed study files will appear in `results/`: a comparison README, metric JSON,
per-slot CSV, paired extraction errors, full lossless gzip reports with raw responses,
and artifact hashes. Model weights and checkpoints remain outside Git.

Evaluation entry point: [evaluate-v5-resumable.py](../../scripts/evaluate-v5-resumable.py).
Packaging entry point: [package-base-final-validation.py](../../scripts/package-base-final-validation.py).
Raw scoring and providers are unchanged. The new runner, interruption handling,
report packaging and matched-runtime gate passed 45 relevant offline tests.

Local live artifacts: `D:\SLM\FYP-model-runs\base-final-validation-20261002`.
The supervisor state and the two stdout/stderr logs record progress or failure.
No aggregate behavioral score is claimed before both complete reports pass pairing.
