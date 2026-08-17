# Current experiment status

Generated JSON and CSV files in this directory are intentionally ignored by
Git because they can contain model output. Keep only reports that are safe to
retain and never commit credentials or private user prompts.

As of 2026-08-11:

- Corpus v1: 200 scenarios; 140 train, 30 validation, 30 test.
- Corpus v2: 320 scenarios; 224 train, 48 validation, 48 test.
- Corpus-v2 SFT records: 420 train and 90 validation examples; zero overlength records.
- Local lab tests: 27 passing.
- Original `fpy` regression tests: 165 passing.
- Local Qwen Q4 smoke case: 271,020.4 ms and invalid/truncated structured JSON.
  This is a baseline failure, not evidence that fine-tuning failed; no adapter
  has been trained yet.
- OpenAI smoke case: blocked with `401 invalid_api_key`. The saved diagnostic
  contains no API-key fragment.
- LoRA training: completed in Colab for four epochs / 68 steps and restored
  under `training-runs/qwen35-08b-lora-v1/`.
- Best validation checkpoint: epoch 1 / step 17 (`eval_loss=0.227544`).
- `best-adapter` is byte-identical to checkpoint 17; later epochs show probable
  overfitting and are retained for comparison.
- Offline adapter integrity: passed (372 tensors / 10,822,656 LoRA parameters;
  all tensors finite).
- Compatible isolated CPU runtime: installed under ignored `.peft-deps/`.
- Held-out PEFT and qualitative cross-questioning generation: pending because
  the unauthenticated official base-model download stalled after a partial
  cache was created. The partial cache was preserved for later resumption.
- SLM runtime safeguards: compact prompt, deterministic intent confirmation,
  tolerant JSON parsing, repeated-question protection, and debug traces added.
- Behavioral evaluation: conversation-v2 cases and metrics are ready; real v2
  results require the new Colab adapter.

Do not label either model the winner until a trained adapter and the production
OpenAI client both complete the same frozen test split.
