# Fairness and six-method coverage audit

This is NOT a completed two-model-by-six-method factorial study. A-F describe method families across different cohorts/configurations. J (JSON mode) is an additional historical control, not a seventh row silently merged into A-F.

| Dimension | Local system | Hosted system |
|---|---|---|
| model/base revision | Qwen/Qwen3.5-2B @ 15852e8c16360a2fea060d615a32b45270f8a8fc | gpt-5-mini-2025-08-07 |
| adapter | {"adapter_config.json": "a1ce14418c5c22375ccfd773338ef1b09117ec9b9e72db034c71ade886b4664c", "adapter_model.safetensors": "c83e0f1a458125970eb70ff360ebead02618f2f48b5bbb59e2ca91a5b7c4c0df"} | No task-specific adapter documented; hosted parameters/training not inspectable |
| training/development exposure | Fine-tuned synthetic forecasting domain/annotation policy; related generator/templates; these public validation cases were previously inspected | No task-specific fine-tuning documented; pretraining/data exposure unknown; these cases/prompts were inspected during study development |
| instructions | Same frozen R3 augmented common instructions; SHA b0fcc0cb06d2d3aa753ab3a3c81ace8e28bd44d5674e5386591fc50fb1d0a7a6 | Same instruction string; schema also enforced through hosted transport |
| schema and input | Same 79-slot registry and visible response schema; same component inputs; own rollout state | Same registry/schema/script; component identities verified; own rollout state |
| admission/values | Native Pydantic candidate decoder once; no extra inner-JSON rejection; capped/refused excluded | Identical native parser/one-decode policy |
| scoring | Actionable non-intent slot/decoded-value/status exact multiset; ignore unmentioned/dont_care and report counts | Identical rules; duplicate predictions retained; no synonym/paraphrase/numeric repair |
| decoding | {"dtype": "bfloat16", "max_new_tokens": 1024, "sampling": false, "seed": 42, "thinking": false} | {"max_output_tokens": 2048, "max_retries": 0, "reasoning_effort": "minimal", "store": false} |
| output budget | 1024 new tokens; greedy; thinking disabled | 2048 output tokens including reasoning; minimal reasoning; native Structured Outputs |
| runtime | {"active_adapters": ["default"], "base_model_class": "Qwen3_5ForConditionalGeneration", "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc", "chat_template_sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80", "devices": ["cuda:0"], "dtype": "torch.bfloat16", "gpu": "NVIDIA RTX A4000", "model_class": "PeftModel", "torch": "2.11.0+cu128"} | Pinned snapshot via direct api.openai.com; host/network transport; cannot inspect server hardware |
| fresh/reused | 202 main + 18 pilot responses reused from R3 only after identical input and state-trajectory proof; newly rescored | 202 main + 18 pilot responses fresh for R4; 132 already dispatched calls replayed on R4.1 continuation without redispatch |
| latency definition | Original warm CUDA generation wall time, synchronized; load/warmup excluded; earlier session/thermal conditions retained | Synchronous request wall time including network/provider generation; no automatic retries; no question-generation calls |
| pipeline/question context | Current fpy plus bundled fixes; same reducer/recovery; fixed scripted turns; question text cleared before next input | Same application path; R4.1 wrapper bounds unscored static-question length error for both; not an unmodified production-prompt benchmark |
| review/validity | Agent-authored/inspected synthetic development; independent human review incomplete; no sealed final labels | Same cohort and limitations; no general superiority/equivalence claim |

## A-F evidence matrix

| Method | Configuration | Completed/documented scope | Verification and limits |
|---|---|---|---|
| A | Prompt-only JSON, no retry | R4 v8 local greedy A-style: 20 matched development conversations, 202 main reused emissions; historical hosted A: 32 natural conversations, 96 logical calls/track (documented separately) | R4 404-call replay and source ZIP verified; historical A narrative/source provenance only. No matched v8-vs-hosted A ablation; do not equate distinct prompts/cohorts |
| B | One validation-guided retry | Historical gpt-5-mini-2025-08-07: 32 natural conversations plus separate 16-scenario coverage diagnostic | Implementation and teacher report present; primary historical B raw completion receipt unavailable on GPU checkout. No R4 v8 retry experiment; do not call this a complete matched B comparison |
| C | Local XGrammar JSON-schema constrained decoding | Configured v6 Qwen3.5-2B: 32 three-turn conversations; original vs constrained, component/rollout; 384 extractions | Completion 384 calls and stored artifact hashes rechecked; exact receipt outcomes in receipt_verification.json. No hosted XGrammar arm; no v8 C arm; configured backend result is not universal constrained-decoding evidence |
| D | Slot-wise probes with aggregation | Documented hosted pilot: ONE weather conversation, three turns/track; 159 component and 237 rollout API calls | Teacher-report provenance and implementation only; primary raw D receipt unavailable here. Pilot not a full comparison; zero pilot F1 does not establish inherent slot-wise inferiority |
| E | Deterministic rules; zero LLM calls | Documented 231-scenario synthetic validation: 717 extraction + 231 question calls; separate historical 32 natural conversations | Checkpoint records report SHA and completion; 41MB full report absent here, so raw performance not newly reverified. No model identity for E; not OpenAI-powered; no fresh R4 E comparison |
| F | Hosted native Structured Outputs | R4 gpt-5-mini-2025-08-07: 20 conversations, 202 fresh main + 18 pilot calls; separate verified historical 31 conversations/186 calls | R4 sealed ZIP and all 404 score/state replays; historical checkpoint manifest checked. R4 is one configured local-vs-hosted comparison, not six arms for both models |

Raw receipts for the historical hosted A/B/D and full E rule result are not present in this checkout. Their documented scopes are retained with a lower evidence tier; this supplement does not manufacture independent verification. Supply those exact public receipts from the main PC if a full provenance recheck is needed.

Historical 0.8B and 2B training runs differ in data, prompts, adapters and available raw evidence; they are not a controlled model-size experiment. The 48-case 2B-v3 comparison and older 0.8B claims must remain separate. No size conclusion is derived here.

Distinct hosted schema enforcement and local greedy decoding remain confounds of configured-system comparison. Reused local timing and fresh hosted network timing are descriptive, not hardware/energy or general speed claims.
