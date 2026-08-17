# Local SLM Lab

This is a standalone experiment for testing a small language model locally.
It reuses read-only forecasting contracts from `../fpy` but keeps all SLM
prompts, guardrails, adapters, transcripts, and training changes here. It does
not modify the OpenAI production pipeline.

## Initial experiment

- Runtime: `llama.cpp` (`llama-server`)
- First model: Qwen3.5 0.8B, GGUF, Q4_0 quantization
- Local API: `http://127.0.0.1:8080/v1`
- Purpose: measure local-model quality, speed, and memory before considering fine-tuning

The local HTTP API is only an interface. It does not send prompts to an external
provider when `llama-server` is bound to `127.0.0.1`.

## Project layout

```text
local-slm-lab/
  config.example.json       Local server and generation settings
  evaluation/cases.jsonl    Small, versioned evaluation set
  corpus/                   Frozen 200-scenario corpus and leakage-safe splits
  corpus-v2/                320-scenario compact-prompt conversational corpus
  local_slm_lab/            Dependency-free Python client and evaluator
  models/                   Local model files (ignored by Git)
  results/                  Evaluation outputs (ignored by Git)
  tests/                    Offline unit tests
```

## Step 1: verify the scaffold

From this directory:

```powershell
python -m unittest discover -s tests -v
```

These tests are offline and do not require a model.

## Step 2: install a local runtime

Download a current Windows CPU build of `llama.cpp` from its official releases:

https://github.com/ggml-org/llama.cpp/releases

Extract it somewhere outside this Git repository and confirm:

```powershell
llama-server --version
```

Do not download anything from an unverified model mirror.

## Step 3: download the first model

Use the official Qwen model and the `ggml-org` llama.cpp conversion. Start with
the 563 MB `Q4_0` file, which is the smallest 4-bit conversion in that repository.
Put the `.gguf` file in `models/`; model files are deliberately excluded from Git.

Official model card:

https://huggingface.co/Qwen/Qwen3.5-0.8B

https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF

Record the exact repository, filename, license, and SHA-256 checksum before use.

## Step 4: start the model server

```powershell
.\scripts\start-local-server.ps1
```

Binding to `127.0.0.1` keeps the server local to this computer. A 4096-token
context is intentionally conservative for an 8 GB machine.

## Step 5: run one prompt

In a second terminal:

```powershell
python -m local_slm_lab.chat "Extract the forecasting target: predict Bitcoin closing price tomorrow"
```

## Step 6: run the evaluation set

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-comparison.ps1
```

The comparison runner requires no Python. It runs the frozen
`evaluation/comparison-v1.jsonl` suite and creates timestamped JSON and CSV files
under `results/`. Open the CSV and score every answer:

- `2`: correct and useful
- `1`: partially correct but needs repair
- `0`: incorrect, invented, unsafe, or failed

Keep the suite unchanged after recording a baseline. Later, run the same prompts,
generation settings, automated checks, and human rubric against the LLM-REI
implementation. Do not use these comparison cases as fine-tuning examples.

## 200-scenario fine-tuning experiment

The larger corpora and common structured evaluation harness are documented in
[`corpus/README.md`](corpus/README.md), [`corpus-v2/README.md`](corpus-v2/README.md),
and [`TRAINING.md`](TRAINING.md). They use
the production forecasting schema, extractor contract, and LLMREI-long question
rules from `../fpy`, while keeping this experiment a separate project.

The completed Colab LoRA run is stored in the ignored `training-runs/` area.
Its base/adapter evaluator and interactive questioning tool are separate from
the production `fpy` application; see `TRAINING.md` for commands and safeguards.

The recommended v2 Colab workflow is available as
[`notebooks/qwen35_lora_v2_colab.ipynb`](notebooks/qwen35_lora_v2_colab.ipynb):
GitHub supplies code/data, Drive receives checkpoints, and interrupted free
Colab sessions automatically resume from the latest checkpoint.

For the current interactive PEFT chat on an 8 GB Windows CPU machine:

```powershell
python scripts\chat-peft.py --variant best --device cpu --dtype bfloat16 --max-new-tokens 512 --debug
```
