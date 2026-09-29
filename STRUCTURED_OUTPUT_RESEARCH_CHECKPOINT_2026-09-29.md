# Structured-output research checkpoint

Date: 2026-09-29  
Branch: `experiment/v5-grounded-20260919`  
Status: implementation and protocol complete; empirical model measurements pending the final frozen 2B adapter.

## Research decision

The system does not choose between “JSON” and “Pydantic.” The trained model emits a JSON wire representation, the strict parser rejects malformed or ambiguous JSON, Pydantic constructs typed domain objects, and the application reducer enforces evidence and state semantics. JSON Schema can additionally constrain tokens during generation.

The controlled research question is whether post-generation retry or generation-time schema enforcement changes semantic extraction accuracy, structural reliability, or efficiency while the model, adapter, prompt, tokenizer, and decoding settings remain fixed.

## Prior-work boundary

Intent and slot filling, dialogue state tracking, schema-guided dialogue, question-based slot extraction, JSON Schema, validation libraries, and constrained decoding are established. The strongest defensible novelty statement concerns their combination and evaluation for:

- a private/local 2B model;
- incremental forecasting-requirements elicitation;
- 79 lifecycle slots;
- dense multi-slot updates;
- corrections, conflicts, indifference, and abstention;
- verbatim evidence grounding;
- forbidden-slot and hallucination measurement; and
- deterministic state transition evaluation.

This was a targeted literature and implementation review, not a systematic review. A separate paper becomes defensible only if the controlled results show a useful semantic/validity/cost tradeoff and the related-work search is expanded with a documented search protocol.

## Implemented experiment arms

1. **A — prompt JSON + Pydantic, no retry.** Existing exact v5 behavior.
2. **B — prompt JSON + Pydantic, one retry.** The first request is byte-for-byte the A prompt path. A compact validation code permits one deterministic second request. Both attempts, tokens, validity, and latency remain visible.
3. **C — XGrammar JSON Schema + Pydantic.** Uses `xgrammar==0.2.7` as a Hugging Face logits processor, preserving the same native Transformers model and LoRA adapter. Grammar compilation time is separate from generation time.
4. **D — slot-wise QA.** One control call and one binary probe for every non-intent slot, with a mandatory high-call-count gate. This is a semantic modeling baseline rather than an output-format ablation.
5. **E — deterministic rules.** Conservative partial coverage for explicit frequency, horizon, source mode/reference, output format, evaluation metric/folds, forecast type, and approval phrases. It reports zero model calls and exposes its recall limitation.
6. **F — OpenAI Structured Outputs.** Uses an explicit model snapshot, the same v5 prompts, the exported strict schema, and local Pydantic validation. Two independent cost gates and an environment-only key are required.

Outlines is retained as a possible sensitivity implementation. llama.cpp is retained as a deployment comparison because GGUF conversion and quantization would confound a controlled native-PEFT output-method ablation.

## Frozen validation protocol

The deterministic v2 cohort contains:

- 231 scenarios and source clusters;
- 717 extraction calls;
- 231 question calls;
- 948 total calls;
- exact non-intent slot micro F1 as the primary metric; and
- 10,000 paired source-cluster bootstrap resamples with seed 42.

The v2 preparation fixed an audit-only `SlotState.updated_at` timestamp in question contexts. It was generated twice in separate runs and produced the same call-plan SHA-256 both times:

`7ce103ded235c5c9558868c9b99bf1519552fadbdcb50c25d4dd73cef19a9069`

Every validation run rebuilds the plan and rejects changes to case bytes, order, context, gold labels, schema, prompts, scorer, adapter hashes, or controlled runtime settings.

## Durability and safety

- Validation runs are written one scenario at a time.
- Completed scenario reports are immutable and hashable.
- An interrupted scenario is preserved under `interrupted/` and replayed on resume.
- Resumption rejects a different adapter, code, input, schema, dtype, device, or runtime signature.
- Arm D requires explicit workload acceptance.
- Arm F requires both a CLI approval flag and `V5_OPENAI_COST_APPROVED=yes`.
- API keys are read from the process environment/Colab Secrets and are never serialized.
- The sealed final split is unsupported by the research runners.

## Recorded outputs and metrics

Each report retains raw text, parsed output, validation status, attempt details, exact predictions, contexts, label hashes, token counts, call counts, latency, reducer transitions, and safe error types. Aggregates include:

- exact inclusive and non-intent precision/recall/F1;
- per-slot F1;
- intent, correction, question, and final-state metrics;
- raw JSON, schema, and first-pass validity;
- retry and retry-success rates;
- duplicate and unknown slots;
- wrong-but-schema-valid outputs;
- forbidden and hallucinated slots;
- model calls, token totals, throughput, p50/p95 latency; and
- process/GPU peak-memory observations where the runtime exposes them.

## Execution package

- Protocol: `STRUCTURED_OUTPUT_COMPARISON_PLAN.md`
- Operational guide: `evaluation/v5-structured-output/README.md`
- Strict schemas: `evaluation/v5-structured-output/schemas/`
- Deterministic cohort: `evaluation/v5-structured-output/prepared-validation-v2/`
- Providers: `local_slm_lab/structured_output_providers.py`
- Single-run diagnostic: `scripts/evaluate-structured-output.py`
- Durable validation: `scripts/evaluate-structured-output-resumable.py`
- Paired comparison: `scripts/compare-structured-output.py`
- Colab runner: `notebooks/Qwen_2B_v5_Structured_Output_Research_Colab.ipynb`

## Empirical work that remains

The implementation is complete, but no quality claim is made before measurements. After training:

1. Freeze the selected adapter and its hashes.
2. Run smoke B, A, C, E.
3. Run validation in randomized order B, A, C, then E.
4. Compare A-B and A-C with the paired protocol.
5. Run D only if its call budget is worthwhile.
6. Run F only after API budget approval and key rotation.
7. Select the extraction method on validation.
8. Authorize a once-only final evaluation separately after review.

## Paper decision rule

This work can support a focused paper when the final experiments show one or more of the following with paired uncertainty and transparent cost:

- validation-guided retry improves end-to-end semantic success enough to justify its calls;
- constrained decoding improves operational validity without harming semantic slot F1;
- structural enforcement changes small-model semantic behavior in a measurable way;
- the local fine-tuned 2B extractor reaches a useful quality/cost/privacy point against the external ceiling; or
- the 79-slot evidence-grounded protocol reveals a validity-versus-correctness distinction missed by ordinary JSON-validity reporting.

If the only finding is that constrained decoding produces valid braces, it should remain an FYP ablation rather than a separate paper.
