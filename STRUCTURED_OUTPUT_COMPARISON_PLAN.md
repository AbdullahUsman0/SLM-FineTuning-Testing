# Structured output comparison for the forecasting requirements extractor

Date: 2026-09-28

## Decision

JSON and Pydantic are not competing output formats.

- **JSON** is the language-independent wire format emitted by the model and stored or sent between components.
- **JSON Schema** describes the allowed JSON structure.
- **Pydantic** is the Python runtime layer that validates JSON-derived values, constructs typed objects, reports errors, serializes objects, and can generate JSON Schema.
- **Constrained decoding** applies a schema or grammar while the model is generating tokens. Pydantic validation alone happens after generation unless its generated schema is supplied to such a decoder.

The current v5 system already uses JSON and Pydantic together. The model emits JSON; `local_slm_lab/v5_provider.py` rejects duplicate JSON keys, non-finite numbers, unexpected fields, duplicate slot updates, and non-JSON candidate strings; it then calls `ExtractorResult.model_validate(parsed)`. `ExtractorResult`, `SlotUpdate`, `QuestionOutput`, dialogue state, and the forecasting schema are Pydantic models in the pinned `fpy` repository.

The useful experiment is therefore **prompt-only JSON versus alternative enforcement strategies**, while keeping the trained model, adapter, prompts, data, and decoding settings fixed.

## What prior work covers

The broad task is established research:

1. Slot filling and intent detection have long been studied as structured language understanding tasks. SlotRefine is one example of a fast joint intent and slot model: https://aclanthology.org/2020.emnlp-main.152/
2. Dialogue state tracking represents a conversation as predefined slot-value state and updates it across turns: https://aclanthology.org/2021.tacl-1.34/
3. The Schema-Guided Dialogue dataset established multi-domain services with natural-language schema descriptions and unseen-service evaluation: https://ojs.aaai.org/index.php/AAAI/article/view/6394
4. Schema-guided DST research studies how slot descriptions and constraints support transfer to unseen services: https://aclanthology.org/2021.naacl-main.62/
5. QA-driven zero-shot slot filling treats each slot description as an extraction question: https://aclanthology.org/2021.acl-short.83/
6. Recent requirements-engineering studies use LLMs to generate or synthesize requirements from interviews and collaborative discussions: https://werpapers.dimap.ufrn.br/proceedings/WER2025/wer202511.html and https://arxiv.org/abs/2606.24060
7. Structured generation research evaluates JSON Schema constrained decoding independently of semantic correctness. JSONSchemaBench compares Guidance, Outlines, llama.cpp, XGrammar, OpenAI, and Gemini: https://arxiv.org/abs/2501.10868
8. XGrammar is an efficient grammar engine for JSON, regex, and context-free grammars: https://arxiv.org/abs/2411.15100
9. Pydantic can validate JSON and generate JSON Schema Draft 2020-12 representations: https://docs.pydantic.dev/latest/concepts/json/ and https://docs.pydantic.dev/latest/concepts/json_schema/

This search did **not** identify an exact published system combining all of the following:

- a private, local small language model;
- incremental forecasting-requirements elicitation;
- a fixed 79-slot schema covering data, target, time, evaluation, governance, deployment, and output requirements;
- multi-slot updates, corrections, conflicts, explicit indifference, and abstention;
- verbatim evidence for every user-grounded update;
- forbidden-slot and hallucination evaluation;
- deterministic state reduction after each turn; and
- direct comparison against a production LLM on identical cases.

The defensible novelty claim is this combination and its evaluation. It is not defensible to claim that requirement extraction, slot filling, dialogue state tracking, JSON output, or Pydantic validation is new. A final paper should call the literature check a targeted search unless a formal systematic-review protocol is completed.

## Research questions

**RQ1 — Semantic accuracy:** Does generation-time schema enforcement change exact non-intent slot F1 compared with the current prompt-only JSON baseline?

**RQ2 — Structural reliability:** Which method minimizes malformed JSON, schema-invalid output, duplicate updates, and invalid slot identifiers?

**RQ3 — Grounding and safety:** Does an enforcement method change unsupported updates, forbidden-slot inference, correction handling, or evidence grounding?

**RQ4 — efficiency:** What latency, throughput, output-token, retry, memory, and implementation costs accompany each method?

**RQ5 — model scale:** Under the same enforcement strategy, how do the historical 0.8B adapter and the final 2B adapter differ in semantic accuracy and structural reliability? Historical results may be reported separately when identical artifacts or cases are unavailable.

## Experimental arms

Run A–C first. They answer the JSON/Pydantic question without another training run.

| Arm | Generation | Validation and recovery | Purpose |
|---|---|---|---|
| A. Current baseline | Greedy prompt-only JSON | Strict JSON parser followed by Pydantic; no retry | Measures the trained model exactly as currently designed |
| B. Pydantic retry | Same generation as A | On a Pydantic error, return a compact validation error and allow exactly one deterministic retry | Tests whether post-generation validation plus repair improves end-to-end success |
| C. JSON Schema constrained | Same model, adapter, prompt, token limit, and greedy decoding; apply the exported strict JSON Schema with XGrammar, Outlines, or llama.cpp grammar | Pydantic still validates the result | Separates structural enforcement from semantic extraction quality |
| D. Slot-wise QA | Ask an extraction question for each active slot, then combine validated answers | Pydantic and state reducer | Stronger but much more expensive semantic baseline; useful only after A–C |
| E. Deterministic rules | Regex and dictionaries for a declared subset such as file format, source mode, frequency, horizon, dates, metrics, and booleans | Typed normalization and Pydantic | Non-neural precision and latency baseline; report coverage explicitly |
| F. External model | Pinned OpenAI structured-output model on the identical call plan | Provider JSON Schema plus local Pydantic | System-level quality ceiling; requires approved API spending |

Do not label B as “Pydantic instead of JSON.” Its accurate label is **prompted JSON plus Pydantic validation and one repair attempt**. Arm C is **JSON Schema constrained decoding plus Pydantic validation**.

## Fair-comparison controls

1. Freeze the final chosen 2B checkpoint by path and SHA-256 before evaluating any arm.
2. Use the existing frozen v5 validation scenarios and `build_call_plan`; do not touch the sealed final set during method selection.
3. Use the same full-precision model, adapter, device, dtype, prompt version, chat template, context, `max_new_tokens`, and greedy decoding for A–C.
4. Generate the JSON Schema once from the pinned Pydantic models, harden it to match the v5 exact-field contract, and record its SHA-256.
5. Do not silently repair outputs for Arm A. Preserve raw text before parsing in every arm.
6. Permit at most one retry in Arm B. Count both attempts, tokens, and latency. A successful retry is not a first-pass success.
7. Warm up each backend before timing. Alternate or randomize arm order by scenario to limit thermal and caching bias.
8. Keep state transitions identical by using the pinned `apply_extraction` reducer.
9. Select the strategy on validation only. Run each selected system once on the sealed final set using the existing once-only gate.
10. Use the existing scenario-cluster paired bootstrap with 10,000 resamples and seed 42 for differences in semantic metrics.

## Metrics

### Primary

- Exact non-intent slot micro F1 over `(slot_id, canonical candidate_value, status)`.

### Semantic and state

- Non-intent precision and recall.
- Per-slot precision, recall, and F1.
- Intent macro F1.
- Correction detection and corrected-value accuracy.
- Exact final dialogue-state accuracy.
- Evidence-text substring validity and evidence exactness.
- Forbidden-slot call rate and forbidden-slot rate.
- Hallucinated-slot call rate and count.
- Question contract accuracy for the separate question-generation operation.

### Structural

- Raw JSON validity.
- Raw schema validity.
- First-pass validity.
- Duplicate JSON key and duplicate slot-update counts.
- Unknown slot identifier rate.
- Wrong-valid-schema rate: schema-valid calls whose semantic extraction is not exact.
- Retry rate and retry success rate for Arm B.

### Efficiency

- Median and p95 end-to-end latency per call.
- Generation tokens per second.
- Prompt and completion tokens.
- Peak VRAM and host RAM.
- Number of model calls per scenario.
- Schema compilation time, reported separately from generation time.

Structural validity must not be used as a proxy for extraction quality. A constrained decoder can guarantee braces, field names, and types while still selecting the wrong slot or value.

## Expected interpretation

- If Arm C raises schema validity but leaves exact slot F1 unchanged, constrained decoding is valuable operationally but does not replace fine-tuning for semantic extraction.
- If Arm C lowers F1, report the validity-correctness tradeoff rather than choosing it automatically.
- If Arm B improves end-to-end success, report its extra calls and latency. Compare first-pass results separately so retries do not hide model weakness.
- If the 2B model beats the 0.8B model on identical cases, attribute the result to the complete system unless model, data, prompt, runtime, and adapter are all controlled.
- If domain-specific data improves finance cases but harms other forecasting domains, present both in-domain and cross-domain metrics rather than only their aggregate.

## Execution order after the final training run

1. Verify the final checkpoint and write adapter SHA-256, repository commit, `fpy` commit, model revision, package versions, and hardware to the run manifest.
2. Evaluate every retained epoch checkpoint with Arm A on v5 validation and choose the behavioral checkpoint using the predeclared v5 rule.
3. Run Arms A–C on that one checkpoint and the exact same validation call plan.
4. Decide whether the added cost of D and E is justified. They are separate modeling baselines, not output-format ablations.
5. Freeze the winning extraction strategy.
6. Run the external OpenAI comparison on the same validation plan if spending is approved.
7. Run the once-only sealed final evaluation for the frozen systems.
8. Save raw outputs, parsed outputs, validation errors, per-case scores, aggregate metrics, bootstrap differences, latency data, manifests, and SHA-256 hashes.

## Paper wording

Suggested methods wording:

> The extractor emits a language-independent JSON wire representation. A strict parser rejects duplicate keys and non-finite values, after which Pydantic constructs and validates the typed extraction object. We separately evaluate prompt-only generation, validation-guided retry, and JSON Schema constrained decoding to distinguish syntactic conformance from semantic requirement extraction.

Suggested novelty wording:

> We study evidence-grounded, incremental requirements extraction for time-series forecasting with a locally deployable 2B language model and a 79-slot lifecycle schema. The evaluation covers dense multi-slot turns, corrections, conflicts, abstention, forbidden inference, state transitions, and structured-output reliability.

