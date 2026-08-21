# Fine-tuning and A/B evaluation

This experiment uses supervised fine-tuning (SFT) with a LoRA adapter. It does
not use reinforcement learning. Corpus v3 contains 320 reviewed scenarios and
1,049 unique extraction-only prompt/completion examples. Its 48 held-out scenarios never
enter training.

## Why not train the downloaded GGUF

The Q4 GGUF is for efficient CPU inference. Train the original Hugging Face
checkpoint, save a small LoRA adapter, merge it if needed, and then convert and
quantize the merged model to GGUF. Repeatedly training an already quantized GGUF
is not the supported path.

The default base is the post-trained `Qwen/Qwen3.5-0.8B`, which is intended for
prototyping and task-specific fine-tuning. A CUDA GPU is required by the script.
The current 8 GB Windows CPU machine should remain the inference/test machine;
use a Colab, Kaggle, or other temporary GPU for training.

## GPU training

Copy this repository to a GPU environment, then run:

```bash
python -m pip install -r training/requirements.txt
python scripts/build-corpus-v3.py
python scripts/verify-corpus-v3.py
python training/train_lora.py
```

The v3 script defaults to `corpus-v3/`, writes a separate
`training-runs/qwen35-08b-lora-v3/` run, checks every tokenized record before
training, trains on assistant completion tokens only, and stops early when
validation loss fails to improve. Exact Colab commands are in
`training/COLAB_V3.md`.

The run writes the input checksums, versions, seed, hyperparameters, GPU name,
validation loss, and best adapter under `training-runs/`.

The completed Colab run is restored under
`training-runs/qwen35-08b-lora-v1/`. Its `best-adapter` is byte-identical to
epoch-1 `checkpoint-17`, which achieved the best validation loss (`0.227544`).
See that run's `AUDIT.md` before using or moving its artifacts.

Do not tune against `corpus-v3/splits/test.jsonl`. Make decisions using validation;
run the test set only for the final candidate.

V1 and v2 are retained as baselines. V3 starts a fresh LoRA adapter from the
same Qwen base because v2 mixed extraction with question generation and did not
explicitly limit loss to completion tokens. Runtime question wording remains a
deterministic schema responsibility; Qwen performs contextual extraction.

## Apples-to-apples evaluation

Start the local llama.cpp server and run the frozen test split:

For structured evaluation, start it with one 8192-token slot so complete
extraction responses are not truncated:

```powershell
.\scripts\start-local-server.ps1 -Evaluation
```

```powershell
python scripts/evaluate-components.py `
  --provider local `
  --output results/local-baseline-test.json
```

Run the same cases through the exact production OpenAI structured client. This
uses `../fpy/.env`, consumes API credits, and does not print the API key:

```powershell
python scripts/evaluate-components.py `
  --provider openai `
  --output results/openai-test.json
```

After converting the fine-tuned model to GGUF and restarting llama.cpp with that
file, rerun the local command with a different output name. Compare reports:

```powershell
python scripts/compare-reports.py `
  --local results/local-finetuned-test.json `
  --openai results/openai-test.json `
  --output results/final-comparison.csv
```

Promotion requires higher validation/test task metrics with no regression in
schema validity, forbidden inference, or one-question compliance. A lower
training loss alone is not evidence of a better interviewer.

The original application's configured snapshot is
`gpt-5-mini-2025-08-07`. Keep that pinned value for the first historical A/B
comparison. It is not the current flagship model, so a later experiment may add
a newer OpenAI model as a third arm without replacing the historical baseline.

## Test the restored PEFT adapter separately

The PEFT tools below live only in this experiment and do not modify or write to
`../fpy`. Use a dedicated environment matching the successful Colab versions:

```powershell
python -m pip install --target .peft-deps -r training\requirements-inference.txt
```

First validate artifact routing without loading or downloading the base model:

```powershell
python scripts\evaluate-peft.py `
  --variant best `
  --check-artifacts
```

Run the same held-out structured evaluation separately for the base, best
epoch-1 adapter, and optional final epoch-4 checkpoint:

```powershell
python scripts\evaluate-peft.py --variant base --output results\peft-base-test.json
python scripts\evaluate-peft.py --variant best --output results\peft-best-test.json
python scripts\evaluate-peft.py --variant final --output results\peft-final-test.json
```

For manual questioning and cross-questioning with raw structured-output traces
saved inside the transcript:

```powershell
python scripts\chat-peft.py --variant best --debug
```

The SLM path uses compact prompts, deterministic intent recovery, robust JSON
object parsing, and repeated-question protection. These safeguards can be
tested with the retained v1 adapter immediately, but the adapter was trained on
the old prompt/data profile. Retrain v2 before judging the final improvement.

Run end-to-end behavioral metrics (intent accuracy, state progression,
structured-output validity, and consecutive-question repetition):

```powershell
python scripts\evaluate-conversations.py `
  --variant best `
  --output results\conversation-v1-with-guardrails.json
```

After restoring the v2 adapter, evaluate it without changing the v1 mapping:

```powershell
python scripts\evaluate-conversations.py `
  --variant custom `
  --adapter training-runs\qwen35-08b-lora-v2\best-adapter `
  --output results\conversation-v2.json
```

The original Hugging Face base weights are not included in the handoff. The
first real PEFT run must download `Qwen/Qwen3.5-0.8B`, so Colab or another GPU
environment is preferable. Evaluate `best` before spending time merging or
converting it to GGUF for llama.cpp.
