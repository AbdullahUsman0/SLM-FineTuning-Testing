# Local SLM / Forecasting-REI Project Handoff

**Prepared:** 2026-08-25 (Asia/Karachi)  
**Purpose:** Continue this work from a new ChatGPT/Codex account without losing the project decisions, experiments, metrics, artifacts, or known issues.

This document is the operational handoff. It separates confirmed measurements from interpretation and from unfinished work. It intentionally does **not** contain an API key or any credential.

## 1. Executive status

The work is a standalone local-model experiment for the forecasting requirements-elicitation pipeline. The production OpenAI/LLM-REI project is in the sibling `fpy` repository and is not modified by the local SLM runtime.

The local model is Qwen3.5-0.8B. The project uses LoRA supervised fine-tuning (SFT), not reinforcement learning. Runtime questions are generated deterministically from the forecasting schema; the small model is used primarily for grounded extraction of values from the current user message.

Three LoRA stages exist:

| Stage | Status | Main result |
|---|---|---|
| v1 | Completed | 4 epochs / 68 steps. Best validation checkpoint was epoch 1, step 17. `eval_loss=0.227544`, token accuracy `0.961447`. Later epochs had worse validation loss, indicating likely overfitting. |
| v2 | Completed | 320-scenario corpus, 420 train and 90 validation SFT records. Recorded best evaluation: `eval_loss=0.177027`, token accuracy `0.972241`, epoch 3. Behavioral testing exposed severe extraction sparsity despite the improved token metrics. |
| v3 | Dataset completed; training blocked before step 1 | 1,049 unique extraction-only examples, 865 train and 184 validation. Training failed during LoRA injection because Colab had `torchao 0.10.0`; PEFT requires `torchao >= 0.16.0`. Fix committed and pushed as `ca661aa`. |

The next action is to install the corrected `torchao` version in the current Colab session and rerun v3 training. No v3 optimizer step or checkpoint has been produced yet.

## 2. Repositories and separation

### Local SLM repository

- Local path: `C:\Users\Mahad Enterprises\OneDrive\Desktop\FYP\local-slm-lab`
- GitHub: `https://github.com/AbdullahUsman0/SLM-FineTuning-Testing`
- Current pushed branch: `main`
- Latest pushed commit: `ca661aa Require compatible torchao for Colab training`
- Previous important commits:
  - `60b8eba Prepare focused Qwen extraction corpus v3`
  - `4e0159f Fix reproducible Colab training setup`
  - `6086f0c Prepare Qwen LoRA corpus v2 and Colab training`

### Forecasting application dependency

- Repository: `https://github.com/int-abd-5/fpy.git`
- The experiments use the pinned commit:
  `04d52c015d1e3ecdefe92b87116f209361509b4b`
- The local SLM project imports read-only forecasting schema/contracts from `fpy`.
- The local SLM project does not replace or retrain the production OpenAI pipeline.

### Working-tree warning

At handoff time, the local repository also contains uncommitted evaluation/runtime edits and untracked evaluation scripts/configs from the experimentation process. They were intentionally not included in the v3 dataset commit. Do not run a destructive reset or checkout until those files have been reviewed.

## 3. System design decisions

### Runtime flow

1. A user message enters the forecasting elicitation engine.
2. The forecasting schema determines the next unresolved slot and the deterministic question to ask.
3. The local Qwen model, when needed, extracts only grounded values from the current user message and compact dialogue state.
4. Guardrails validate JSON, slot IDs, evidence grounding, intent, and corrections.
5. The state reducer updates the canonical forecasting state.
6. The deterministic schema question asks for the next unresolved requirement.
7. A human confirmation gate is required before the final forecasting specification is handed to later pipeline stages.

This design intentionally prevents a 0.8B model from inventing arbitrary questions or changing the interview contract.

### Why deterministic questions are not a harmful hard-code

The questions are generated from the versioned schema, not from one Bitcoin or weather example. This provides reliable slot coverage, one-question compliance, repeat protection, and stable evaluation. The model still generalizes the user’s natural-language answer across domains, spelling styles, short answers, corrections, and Roman Urdu.

The model should not be promoted based only on its ability to generate question wording. Question wording can be made more natural later after extraction reliability is strong.

### Reinforcement-learning decision

Full RL/RLHF is not implemented. For this project it would add complexity and instability before the extraction contract is reliable.

The recommended improvement loop is:

```text
Production conversation
    -> redacted feedback event
    -> local validation and confidence checks
    -> OpenAI reviewer only for flagged/sampled cases
    -> human approval and correction
    -> golden evaluation examples / SFT examples
    -> periodic LoRA retraining
```

This is human-in-the-loop active learning plus periodic SFT, not online reinforcement learning.

## 4. Base model and environments

### Base model

- Hugging Face model: `Qwen/Qwen3.5-0.8B`
- Local inference model: Qwen3.5-0.8B GGUF, Q4 quantization through llama.cpp
- Training is performed from the original Hugging Face model, not from a quantized GGUF.
- GGUF is for CPU inference; it is not the recommended training input.

### Recorded successful training environment

- GPU: Tesla T4
- Python: 3.12.13 in the completed runs
- PyTorch: 2.11.0+cu128
- Transformers: 5.15.0
- Datasets: 5.0.1
- Accelerate: 1.14.0
- PEFT: 0.20.0
- TRL: 1.9.2
- LoRA rank: 16
- LoRA alpha: 32
- Effective batch size: 16
- Maximum sequence length: 3,072
- Seed: 42

Current Colab showed Python 3.13.15 and a Tesla T4. It successfully loaded the model, but v3 failed at PEFT adapter injection because the preinstalled `torchao` was too old.

## 5. Dataset/corpus history

### Corpus v1

- Version: `forecasting-llmrei-200-v1`
- 200 scenarios
- Categories: complete, missing required, ambiguous, correction, conflicting, multi-series, probabilistic/covariate, governance/privacy, robustness, intent boundary
- Splits: 140 train, 30 validation, 30 held-out test
- SFT: 266 train, 57 validation
- Test scenarios never enter SFT.
- SHA-256:
  - train: `40ca18238b1a3888429f6aad3b6dc14a91c41681dac1039af9e85b4fafde5810`
  - validation: `8b2cc76aa3ed1154678e083f2546fabc8bb5b9a1bda854222fa5f5fddd26a377`

### Corpus v2

Corpus v2 preserved the original 200 scenarios and added 120 conversational-intent scenarios.

- Version: `forecasting-llmrei-320-v2`
- 320 scenarios
- Splits: 224 train, 48 validation, 48 test
- SFT: 420 train, 90 validation
- SFT task mix:
  - 289 extraction records
  - 221 question-generation records
- Maximum recorded training example: 2,955 tokens; no record exceeded 3,072.
- Added coverage for `yes`, `yeah`, `haan`, short forecast requests, Roman Urdu, negative intent, prior assistant-question context, and state progression.

Important limitation: v2 mixed extraction and question-generation tasks. It also used full `messages` records without explicitly setting completion-only loss. Therefore its high token accuracy was not a clean measure of assistant JSON extraction; easy system/user tokens could dominate the metric.

### Corpus v3

Corpus v3 was designed to correct the v2 problem.

- Version: `forecasting-llmrei-extraction-1049-v3`
- 320 reviewed source scenarios
- 224 train scenarios, 48 validation scenarios, 48 test scenarios
- 1,049 unique SFT records:
  - 865 train
  - 184 validation
- 233 unique reviewed base extraction records
- 272 selected-slot short-answer records
- 272 selected-slot natural-sentence records
- 272 selected-slot non-answer records
- Every one of the 16 required forecasting slots receives targeted coverage.
- Exact duplicate prompt/completion records: zero.
- Contradictory completions for the same prompt: zero.
- Test examples in SFT: zero.
- Format: explicit conversational `prompt` and `completion` fields.
- Trainer loss scope: assistant completion only (`completion_only_loss=True`).
- Runtime question policy: deterministic schema questions.

The v3 artifact is already checked into GitHub under `corpus-v3/`. Its manifest contains checksums and byte sizes.

## 6. Fine-tuning run 1: v1

### Configuration

- Run directory: `training-runs/qwen35-08b-lora-v1/`
- Base: `Qwen/Qwen3.5-0.8B`
- Method: LoRA SFT
- Epochs configured: 4
- Completed: 4 epochs / 68 steps
- Learning rate: `1e-4`
- Max length: 3,072
- Effective batch size: 16
- LoRA rank: 16
- LoRA alpha: 32
- GPU: Tesla T4
- Training time: approximately 4,898 seconds
- Final training loss: approximately `0.1564`

### Validation history

| Epoch | Eval loss | Mean token accuracy |
|---:|---:|---:|
| 1 | 0.2275 | 0.9614 |
| 2 | 0.2947 | 0.9558 |
| 3 | 0.2897 | 0.9564 |
| 4 | 0.2888 | 0.9566 |

### Best adapter decision

- Best checkpoint: `checkpoint-17` (epoch 1)
- `best-adapter/adapter_model.safetensors` is byte-identical to `checkpoint-17/adapter_model.safetensors`.
- Adapter SHA-256:
  `8E2D4210DCE52BEDEF12F67E8F1C4FB621323E5D7B38BDA962EFC3A5092D4B77`
- Final epoch checkpoint 68 has a different hash.
- `best-adapter` therefore contains epoch-1 weights, not epoch-4 weights.
- LoRA adapter: 372 tensors, 10,822,656 trainable parameters, all finite float32 tensors.

### Interpretation

Training loss continued down while validation loss worsened after epoch 1. This is evidence of probable overfitting. The v1 best adapter is retained and should remain a baseline.

The v1 adapter does not include the Qwen base weights. Base-model download is required for inference. A local CPU generation smoke test was blocked when an unauthenticated Hugging Face download stalled after approximately 1 GB; the partial cache was preserved.

## 7. Fine-tuning run 2: v2

### Configuration

- Run directory: `training-runs/qwen35-08b-lora-v2/`
- Base: `Qwen/Qwen3.5-0.8B`
- Method: LoRA SFT
- Epochs configured: 4
- Learning rate: `1e-4`
- Warmup ratio: 0.0
- Seed: 42
- Max length: 3,072
- Effective batch size: 16
- LoRA rank: 16
- LoRA alpha: 32
- Evaluation: each epoch
- Saving: each epoch
- `load_best_model_at_end=True`
- Metric: `eval_loss`
- `greater_is_better=False`
- Early-stopping patience: 1
- No overlength records.

### Recorded result

`training-runs/qwen35-08b-lora-v2/evaluation.json` records:

- Eval loss: `0.17702747881412506`
- Mean token accuracy: `0.9722413321336111`
- Evaluation epoch: `3.0`
- Evaluation runtime: approximately 62.93 seconds
- Evaluation tokens: 821,325

The local v2 handoff retains `best-adapter` and run metadata, but not a complete per-epoch metric table like v1. Treat epoch 3 as the best recorded v2 evaluation, not as proof that v2 generalizes better than v1.

### Behavioral issue found after v2

The high token-level metric did not translate into robust extraction. In the raw 16-scenario component evaluation, local Qwen v2 often returned valid but empty updates. The application guardrails and deterministic recovery improved some conversational behavior, but that means orchestration metrics must be separated from raw model metrics.

## 8. Fine-tuning run 3: v3 status

### Intended configuration

- Corpus: `corpus-v3/sft/train.jsonl` and `corpus-v3/sft/validation.jsonl`
- Output: `/content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v3`
- Epochs: 3
- Learning rate: `5e-5`
- Early-stopping patience: 1
- Max length: 3,072
- Resume: `auto`
- Fresh adapter from the Qwen base; do not initialize from v1 or v2.

### Actual status

The T4 GPU and Qwen base model loaded successfully. The run stopped before the first optimizer step while PEFT attempted to inject LoRA:

```text
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

No v3 checkpoint, adapter, evaluation result, or training loss exists yet.

### Fix

The repository now includes `torchao>=0.16.0` in `training/requirements.txt`.

For the existing Colab runtime, run:

```python
import importlib.metadata
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "--upgrade", "torchao>=0.16.0"
], check=True)

print(importlib.metadata.version("torchao"))
```

Then rerun the streaming v3 training cell. Do not delete the v3 output directory unless a future run creates an incompatible checkpoint; currently there is no v3 checkpoint.

## 9. Model-quality evaluations

The evaluation suite has several layers. They must not be conflated:

1. Raw component evaluation: tests extraction and question calls directly.
2. Conversation evaluation: tests state progression over a dialogue.
3. Guided-generalization evaluation: tests the orchestration/guardrails and may make zero model calls, so it is not a raw-model benchmark.
4. Paired OpenAI comparison: runs the same cases through local Qwen and the pinned OpenAI client.

The OpenAI reference used in these reports is:
`gpt-5-mini-2025-08-07`.

### Raw 16-scenario component evaluation

Files:

- Local: `results/v2-components-corrected-16.json`
- OpenAI: `results/openai-components-corrected-16.json`

| Metric | Local Qwen v2 | OpenAI gpt-5-mini |
|---|---:|---:|
| Provider success | 1.0000 | 1.0000 |
| Structured validity | 1.0000 | 1.0000 |
| Schema validity | 1.0000 | 1.0000 |
| Intent accuracy | 1.0000 | 1.0000 |
| Slot precision | 1.0000 | 0.4478 |
| Slot recall | 0.0109 | 0.3727 |
| Slot F1 | 0.0216 | 0.4068 |
| Slot-ID F1 | 0.0216 | 0.7661 |
| Matched value accuracy | 1.0000 | 0.5310 |
| Non-empty update accuracy | 0.0667 | 1.0000 |
| Empty-update collapse rate | 0.9333 | 0.0000 |
| Joint extraction accuracy | 0.1765 | 0.0000 |
| Correction detection | 1.0000 | 1.0000 |
| Forbidden inference rate | 0.0000 | 0.4000 |
| Question contract accuracy | 0.7692 | 0.9231 |
| Question slot relevance | 0.9231 | 1.0000 |
| Question exact match | 0.5385 | 0.0000 |
| Mean latency | 5,291 ms | 15,140 ms |

Interpretation: local Qwen was conservative and precise when it emitted a value, but it emitted too few updates. OpenAI extracted more values but also produced forbidden inferences in this test. The local raw-model result was not close to OpenAI on recall or F1.

### Paired safe conversation evaluation: 11 cases

File: `results/paired-v3-safe/comparison.json`  
Local provider label: `local-qwen-v2`  
OpenAI model: `gpt-5-mini-2025-08-07`

| Metric | Local Qwen v2 | OpenAI |
|---|---:|---:|
| Case success | 0.1818 | 0.0909 |
| Intent accuracy | 0.9091 | 0.7273 |
| Exact final state | 0.1818 | 0.1818 |
| Slot precision | 1.0000 | 0.6591 |
| Slot recall | 0.1875 | 0.6042 |
| Slot F1 | 0.3158 | 0.6304 |
| Slot-ID F1 | 0.3158 | 0.8696 |
| Safety | 1.0000 | 1.0000 |
| State progression | 0.6111 | 0.7778 |
| Unexpected inference | 0.0000 | 0.0909 |
| Stalled turns | 0.3889 | 0.2222 |
| Consecutive question repeats | 0.0000 | 0.0000 |

Interpretation: the local system was safer and more conservative but stalled more often. OpenAI extracted more and progressed farther. The local hybrid F1 was roughly half of OpenAI’s (`0.3158 / 0.6304 ≈ 50%`), which explains the earlier intuition that the local system had reached roughly 50–60% quality. This is not the same as raw model accuracy because guardrails and deterministic recovery are included.

### Paired one-case guided-generalization result

File: `results/paired-v2-vs-openai-checkpointed/comparison.json`

This result contains one requested case and is too small for a reliable model claim. It reported local Qwen ahead on that single case, with local slot F1 1.0 versus OpenAI 0.5455, but it must not override the 11-case comparison.

### Guided local result

File: `results/local-v2-guided-generalization.json`

- 4 cases, 73 turns
- All reported metrics were 1.0
- `structured_model_calls=0`
- `deterministic_turn_rate=1.0`

This validates orchestration and deterministic guards, not Qwen’s raw extraction ability. It must not be used as evidence that the model beats OpenAI.

### Conversation guardrail result

Files:

- `results/v2-conversations-v3-final.json`
- `results/v2-conversations-v3-guarded.json`

The final local conversation run reported:

- 12 cases, 19 turns
- Intent accuracy: 0.9167
- Slot precision: 1.0
- Slot recall: 0.1923
- Slot F1: 0.3226
- Safety: 1.0
- State progression: 0.6316
- Stalled turns: 0.3684
- Repeated questions: 0.0

The guarded run increased state progression to 0.9474 and lowered stalled turns to 0.0526, but it also changed raw extraction precision/recall and had unexpected inference in that particular diagnostic. Keep raw and guarded metrics in separate columns.

## 10. What the quality metrics mean

- **Intent accuracy:** correctly identifies whether the request is forecasting, not forecasting, ambiguous, or unsupported.
- **Slot precision:** of the slots the model emits, how many are correct. High precision with low recall means the model is too conservative.
- **Slot recall:** of the gold slots, how many the model emits. This was the main local weakness.
- **Slot F1:** balance between precision and recall; the main extraction metric.
- **Slot-ID F1:** whether the model selected the correct field, even if the value was wrong.
- **Matched value accuracy:** correctness of values for correctly matched slots.
- **Empty-update collapse:** percentage of cases where the model returns no updates when updates were expected.
- **Forbidden inference:** emits a slot that the current message did not establish.
- **State progression:** whether the conversation moves toward completion.
- **Stalled turn rate:** no useful state progress on a turn.
- **Question contract:** exactly one valid question, about the selected slot.
- **Exact final state:** every required final value is correct; this is intentionally strict.

The promotion gate should prioritize held-out slot F1, recall, evidence grounding, forbidden-inference rate, empty-update collapse, state progression, and no regression in safety. Training loss and token accuracy alone are insufficient.

## 11. OpenAI reviewer design

OpenAI can be used as an asynchronous reviewer while Qwen handles normal user interaction. This preserves the cost and privacy advantages of local inference.

Recommended reviewer output:

```json
{
  "verdict": "correct | partially_correct | wrong | ambiguous | unsafe",
  "error_types": [
    "missed_slot",
    "wrong_value",
    "wrong_intent",
    "unsupported_inference",
    "invalid_evidence"
  ],
  "corrected_updates": [],
  "confidence": 0.95,
  "usable_for_training": true,
  "requires_human_review": true
}
```

Reviewer policy:

- Local validator accepts clear, grounded outputs without an API call.
- Send only low-confidence, invalid, corrected, sampled, or disagreement cases to OpenAI.
- Redact secrets and personal data before sending.
- Store model version, reviewer prompt version, schema version, consent, redacted input, Qwen output, reviewer output, and human correction.
- Human approval is required before adding an example to SFT.
- Never train automatically on unreviewed user data.
- Keep the frozen test set isolated from future training data.
- Use structured JSON output and `store=false` where supported.

This is active learning/periodic SFT, not online RL. A useful cadence is a review batch after 100–500 approved events or after a measurable production failure pattern appears.

## 12. Dataset discovery and caching context

The separate dataset subsystem plan is a dual catalog:

- **Production catalog:** verified provenance, usable rights, quality, reproducible access.
- **Research catalog:** AutoForecast/Monash and similar benchmark references until each underlying dataset independently passes licensing and provenance review.

AutoForecast was used as a benchmark-source and meta-learning reference. It was not treated as an authoritative production dataset catalog, and its linked research traces were not automatically treated as redistributable.

For Pakistan-first searches, planned official sources include PMD WIS2, PMD request-based climatology, NDMA publications/e-library/MHVRA, FFD, SBP, PBS, Finance Division, FFC, NEPRA, Power Division, WAPDA, SUPARCO, PCRWR, and provincial PDMAs. Open Data Pakistan is discovery-only; resolve every result to the original publisher and independently verify rights.

Cache policy:

- Store reusable public downloads in a SHA-256 content-addressed object store.
- Keep immutable raw versions separate from normalized derivatives.
- Store catalog metadata, rights decisions, freshness, query identity, checksums, and lineage in SQLite.
- Reuse an object only when rights, query identity, and freshness policy allow it.
- Keep purchased PMD data, uploads, microdata, and restricted sources user-scoped.
- Do not cache data whose redistribution terms do not permit it; retain metadata only.
- “Stale” means retained but not trusted for automatic reuse until revalidated. “Deleted” means the object is removed according to retention policy; metadata/audit records may remain.

## 13. Commands for the next account

### Clone and test locally

```powershell
git clone https://github.com/AbdullahUsman0/SLM-FineTuning-Testing.git
cd SLM-FineTuning-Testing
python -m unittest discover -s tests -v
```

### Rebuild and verify v3 corpus

```powershell
python scripts\build-corpus-v3.py
python scripts\verify-corpus-v3.py
```

### Current interactive local SFT server

```powershell
& "C:\Users\Mahad Enterprises\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  .\scripts\chat-sft-server.py `
  --config .\config.v2-evaluation.json
```

The server supports `/state`, `/confirm`, and `/quit`. The question wording is deterministic from the schema.

### Current PEFT chat

The base model must be available locally and the adapter must be supplied separately:

```powershell
python scripts\chat-peft.py --variant best --device cpu --dtype bfloat16 --max-new-tokens 512 --debug
```

The 8 GB Windows machine may hit paging-file or memory errors when loading the full base model. Colab or a GPU machine is preferred for PEFT evaluation.

### Colab v3 setup

1. Select a T4 GPU.
2. Clone the GitHub repository at commit `ca661aa`.
3. Clone `fpy` at commit `04d52c015d1e3ecdefe92b87116f209361509b4b`.
4. Install `training/requirements.txt`.
5. Run `scripts/build-corpus-v3.py`.
6. Run `scripts/verify-corpus-v3.py`.
7. Run the two test-discovery commands, not `python -m unittest tests.test_corpus_v3 ...`.
8. Install/upgrade `torchao>=0.16.0` in the current session if needed.
9. Train using `training/COLAB_V3.md`.
10. Save output directly to `/content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v3`.

### v3 training command

```bash
python training/train_lora.py \
  --train corpus-v3/sft/train.jsonl \
  --validation corpus-v3/sft/validation.jsonl \
  --output /content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v3 \
  --epochs 3 \
  --learning-rate 5e-5 \
  --early-stopping-patience 1 \
  --max-length 3072 \
  --resume-from-checkpoint auto
```

## 14. Files to preserve or transfer

### Source and reproducibility

- `README.md`
- `TRAINING.md`
- `training/COLAB_V3.md`
- `training/requirements.txt`
- `training/train_lora.py`
- `local_slm_lab/corpus_v3.py`
- `scripts/build-corpus-v3.py`
- `scripts/verify-corpus-v3.py`
- `tests/test_corpus_v3.py`
- `tests/test_training_helpers.py`
- `corpus-v3/`
- `evaluation/`
- `results/` reports that are safe to retain

### Model artifacts

The GitHub repository intentionally does not contain large model checkpoints. Transfer these separately if needed:

- `training-runs/qwen35-08b-lora-v1/`
- `training-runs/qwen35-08b-lora-v2/`
- Future `/content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v3/`
- Base Qwen weights or GGUF files, if locally cached

The LoRA adapter alone is not a complete model; it requires the matching Qwen base model.

## 15. Security and data handling

An OpenAI API key was previously pasted during the conversation. It is not recorded in this handoff. Treat that key as exposed: revoke/rotate it before using the new account, and store the replacement only in a local `.env` or secret manager. Never commit `.env`, API keys, user uploads, purchased datasets, or raw private conversations.

The current project `.gitignore` excludes model/runtime/result directories where appropriate, but always inspect `git status` before committing.

## 16. Recommended continuation order

1. In Colab, upgrade `torchao` to `>=0.16.0`.
2. Rerun v3 training from the fresh Qwen base.
3. Record the full run manifest, per-epoch metrics, best checkpoint, and final adapter checksum.
4. Evaluate v3 on the untouched 48-scenario test split.
5. Run the same component and conversation suites against v1, v2, v3, and pinned `gpt-5-mini-2025-08-07`.
6. Compare recall, slot F1, empty-update collapse, evidence grounding, forbidden inference, state progression, stalls, repeats, latency, and memory.
7. Promote v3 only if it improves held-out extraction without safety or regression failures.
8. Add the bounded OpenAI reviewer queue after the evaluation baseline is stable.
9. Periodically create a new approved SFT dataset from human-reviewed corrections; do not continuously fine-tune from raw traffic.

## 17. Bottom line

The first two rounds are not wasted. They provide:

- working LoRA artifacts and checkpoint-selection evidence;
- the initial and expanded scenario corpora;
- the Colab/Drive workflow;
- the local PEFT and llama.cpp evaluation tools;
- the guardrail and deterministic-question architecture;
- baseline comparisons against OpenAI.

The third round is a controlled correction of the main v2 weakness: task mixing and misleading token-level loss. It has not trained yet because of the dependency issue. Once `torchao` is corrected, v3 is the first run whose training objective is aligned with the actual production use case: grounded structured extraction from the current user message.
