# v5 structured-output research package

Status: implementation complete; model-dependent measurements pending the final chosen 2B adapter.

This directory contains a validation-only ablation package for the forecasting requirements extractor. It is deliberately separate from the production/FYP elicitation path. No arm changes model training and no development command reads the sealed final split.

## Research conclusion behind the design

JSON and Pydantic solve different layers of the same pipeline:

1. JSON is the model's language-independent wire representation.
2. JSON Schema declares the permitted JSON structure and can guide generation.
3. Pydantic validates parsed values and constructs the Python domain models.
4. Application validation enforces semantics that JSON Schema cannot express, including unique slot updates, evidence grounding, forbidden inference, and valid state transitions.

The broad ideas of intent/slot filling, dialogue state tracking, schema-guided dialogue, question-based slot extraction, Pydantic validation, and constrained decoding are established prior work. The defensible contribution is the evaluated combination: a local 2B model performing incremental, evidence-grounded extraction into a 79-slot forecasting lifecycle schema, with corrections, abstention, conflicts, forbidden-slot checks, and deterministic state reduction.

Primary references used for the implementation:

- Schema-Guided Dialogue: https://ojs.aaai.org/index.php/AAAI/article/view/6394
- QA-driven zero-shot slot filling: https://aclanthology.org/2021.acl-short.83/
- JSONSchemaBench: https://arxiv.org/abs/2501.10868
- XGrammar paper: https://arxiv.org/abs/2411.15100
- XGrammar Hugging Face integration: https://github.com/mlc-ai/xgrammar/blob/main/docs/start/quick_start.md
- Pydantic JSON and JSON Schema: https://docs.pydantic.dev/latest/concepts/json/ and https://docs.pydantic.dev/latest/concepts/json_schema/
- OpenAI Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs

## Implemented arms

| Arm | Implementation | Role | Runtime gate |
|---|---|---|---|
| A | `V5TransformersProvider` | Exact prompt-only greedy JSON baseline | chosen adapter required |
| B | `V5RetryTransformersProvider` | Exact A first attempt, strict validation, at most one deterministic repair | chosen adapter required |
| C | `V5XGrammarTransformersProvider` | Same model/prompt/greedy path with exported JSON Schema token masking | `xgrammar==0.2.7` |
| D | `V5SlotWiseTransformersProvider` | One control probe plus one probe per non-intent slot | `--allow-high-call-count` |
| E | `V5RuleProvider` | Conservative partial-coverage regex/dictionary baseline | no model |
| F | `V5OpenAIStructuredProvider` | External Structured Outputs ceiling with the same v5 prompts and local Pydantic validation | two cost gates and environment-only key |

Arm D makes roughly 79 model calls per extraction turn and is not an output-format ablation. Arm E intentionally sacrifices recall and reports its zero-call behavior. Arm F is a system ceiling and is not treated as a controlled local-runtime ablation.

## Frozen cohort

`prepared-validation-v2/` freezes 231 validation scenarios and 948 calls. It supersedes v1 because v2 fixes a volatile audit-only timestamp in question-request context hashes:

- 717 extraction calls
- 231 question calls
- scenario-cluster bootstrap: 10,000 resamples, seed 42
- primary metric: exact non-intent slot micro F1
- call-plan SHA-256: recorded in `prepared-validation-v2/study-manifest.json`

The committed call plan contains IDs and hashes rather than duplicated label text. Every validation runner rebuilds the plan and rejects a different input, order, context, gold label, schema, prompt, or scorer.

## Installation

Install the pinned `fpy` package first, then:

```bash
python -m pip install -r training/requirements-structured-output.txt
python -m pip install -e ../fpy
```

## Diagnostic smoke run

Smoke runs can skip warmup and do not claim validation results:

```bash
python scripts/evaluate-structured-output.py \
  --arm A --split smoke \
  --cases corpus-v5/v5-20260919-r1/splits/smoke.jsonl \
  --model Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --adapter /path/to/frozen-adapter \
  --output /new/path/arm-a-smoke.json \
  --skip-warmup
```

Use B or C in `--arm` for their smoke checks. Arm E omits the local model, revision, and adapter arguments.

## Durable validation run

Use the resumable runner for validation. It keeps one model loaded per session, writes one immutable report per scenario, preserves an interrupted scenario, and resumes by skipping verified completed scenarios.

```bash
python scripts/evaluate-structured-output-resumable.py \
  --arm A \
  --model Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --adapter /path/to/frozen-adapter \
  --run-dir /durable/results/arm-a
```

The validation execution order generated with seed 42 is **B, A, C**. Each backend performs a separate smoke-cohort warmup before timing. Complete E after A-C. Run D only if its estimated call count is acceptable. Run F last after approving the paid budget.

Arm F additionally requires:

```bash
export V5_OPENAI_COST_APPROVED=yes
export OPENAI_API_KEY=...  # process environment or Colab Secret; never a file or notebook cell
python scripts/evaluate-structured-output-resumable.py \
  --arm F --model EXPLICIT_MODEL_SNAPSHOT \
  --openai-cost-approved \
  --run-dir /durable/results/arm-f
```

## Paired comparisons

```bash
python scripts/compare-structured-output.py \
  --left /results/arm-a/combined-report.json \
  --right /results/arm-c/combined-report.json \
  --output /results/comparisons/a-vs-c.json
```

For A-C the comparison refuses a different model, revision, adapter hashes, dtype, device, token limit, prompt hash, or runtime signature. It reports paired semantic F1 intervals, structural validity, first-pass validity, wrong-but-schema-valid outputs, retries, unknown slots, forbidden inference, latency, tokens, and model calls.

## Interpretation rules

- Structural validity is never a proxy for correct extraction.
- Arm B reports first-pass and repaired outcomes separately.
- Arm C can improve syntax while leaving slot F1 unchanged or worse.
- D and E are modeling baselines rather than JSON/Pydantic ablations.
- Validation selects the method. The sealed final set remains once-only after the method and adapter are frozen.
- Domain-specific claims require both in-domain and cross-domain reporting. The general v5 cohort remains the primary comparison because requirements extraction is largely domain-independent, while finance/weather vocabulary can be reported as strata.
