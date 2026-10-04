# Test the completed v6 adapter yourself

The experimental v6 run completed 726 optimizer steps / two epochs.
The final-step adapter is not yet selected by behavioral quality. Its
teacher-forced validation loss was 0.00006398998812073842; behavioral v6
validation remains pending. The initial loading smoke missed several slots.

## Run on the remote GPU PC

Use PowerShell in the existing training checkout and its existing environment:

```powershell
Set-Location D:\SLM\SLM-FineTuning-Testing
git pull --ff-only origin experiment/v5-grounded-20260919
& .\.venv\Scripts\python.exe scripts/chat-peft.py `
  --variant custom `
  --adapter "D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter" `
  --adapter-manifest research-checkpoints/v6-training-20261003/results/completion.json `
  --base-model Qwen/Qwen3.5-2B `
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc `
  --prompt-version v5 --device cuda --dtype bfloat16 --max-new-tokens 1024 --debug
```

V6 uses the same all-schema prompt protocol named `v5`; the flag selects that
protocol, not the v5 adapter. The command verifies the adapter weights and config
against the published completion record before loading.
The sibling `D:\SLM\fpy` must use the training dependency commit
`04d52c015d1e3ecdefe92b87116f209361509b4b` for matching behavior.

Use `/state` after each message to inspect recorded slot values, `/reset` before
each independent case, and `/quit` to finish and save a timestamped transcript
under `results/`. Clarification wording is supplied by the schema; debug traces
show the model's raw extraction, including failures. The pipeline can normalize
or recover values, so inspect both raw traces and the recorded state.
This is the requirement extraction pipeline, not a weather/price forecaster.

## Run on the prepared Intel laptop

The authoring laptop has Intel graphics, 32 GB RAM and no CUDA GPU. Its local
CPU server uses the existing Q4_K_M stock base and the converted F16 v6 LoRA.
These are quantized manual tests, not the matched BF16 CUDA research evaluation.
The NPU is not used. The HF cache contains tokenizer/config files; this path
does not require downloading the separate 4.55 GB HF model weights.

On this PC the server was started in the background on port 8086. From the
repository root, launch chat with:

```powershell
.\scripts\chat-v6-local.ps1
```

If the server is stopped, run this in one PowerShell terminal first:

```powershell
.\scripts\start-v6-local.ps1
```

Then run chat in a second terminal. The launcher uses the bundled Python
runtime, an exported pinned fpy snapshot, v5/v6 prompts, and adapter hash checks.
It saves timestamped transcripts and supports `/state`, `/reset` and `/quit`.
The base server has already loaded the converted adapter; the chat hash check
verifies the original adapter files but cannot independently attest what an
arbitrarily supplied HTTP server loaded. Local server logs and the runtime
manifest identify the loaded files.

Prepared artifacts (ignored by Git):

- `runtime/llama-b11026-cpu/llama-server.exe`
- `models/Qwen3.5-2B-Q4_K_M.gguf`
- `runtime/adapter-downloads/v6-adapter-step726-F16.gguf`
- `runtime/adapter-downloads/v6-local-runtime.json`
- `runtime/fpy-v6-pinned/src/`

A fresh clone on another PC does not include these artifacts. Use the existing
HF setup later in this guide or prepare this llama.cpp runtime separately.
The adapter conversion used the existing b11026 converter and cached pinned
base configuration, with `--outtype f16`. The local base GGUF hash is recorded;
an exact underlying HF weight-revision match has not been established here.

The October 4 local smoke extracted 10 explicitly stated weather requirements
and correctly changed a 7-day horizon to 14 days. Both raw outputs passed strict
JSON/schema parsing. The first call took about 53 seconds at approximately
17.5 generated tokens/second with eight CPU threads. These two illustrative
turns do not establish overall quality or performance across the dataset.

## Manual cases

These newly written illustrative cases are not an independently reviewed or
frozen benchmark. They do not access the sealed final labels. Expected checks
describe meaning; free-text wording and evidence must follow the actual schema.

### 1. Explicit weather requirements

```text
Please build a time-series forecast. Forecast daily maximum temperature for Lahore. The target is measured in Celsius. Each observation covers 1 day. Predict the next 7 days. Read the target from the tmax column. The timestamp column is date. I will upload the data. The uploaded file format is csv. The source is weather.csv.
```

Check separate `target_description`, `target_unit` (Celsius), `frequency`
({"periods":1,"unit":"day"}), `forecast_horizon` ({"periods":7,"unit":"day"}),
`target_column` (tmax), `time_column` (date), `source_mode` (upload),
`file_format` (csv), and `source_reference` (weather.csv).

Then, in the same dialogue:

```text
Correction: predict the next 14 days instead of 7 days.
```

Check that the horizon changes to 14 days, `correction_detected` is true,
and the other recorded facts are retained.

### 2. Informal weather phrasing

```text
I want to forecast daily maximum temperature in Celsius for the next 7 days using a CSV file.
```

This repeats the published loading smoke. Look for separate unit, frequency,
horizon and format updates, rather than every fact folded into the target
description. Do not invent a filename or column names. Compare with case 1
to observe any dependence on explicit, training-like phrasing.

### 3. Inflation

```text
Forecast Pakistan's year-over-year CPI inflation rate, not the CPI index level. It is measured in percent and observed monthly. Predict the next 6 months. The target column is cpi_yoy and the time column is month. I will upload inflation.csv in csv format. This forecast will support budget planning.
```

Check rate versus index meaning, percent unit, one-month frequency,
six-month horizon, columns, source and `business_goal`. No assumed future
inflation values or unmentioned seasonal adjustment.

### 4. Equities: observation versus forecast cadence

```text
Forecast the split-adjusted closing price of ACME shares. The price is measured in USD. Observations are daily and follow a trading calendar. Predict the next 10 days. Generate a new forecast every 7 days. The target column is adjusted_close. I will upload stocks.csv in csv format.
```

Check `frequency` = one day, `prediction_frequency` = seven days,
`forecast_horizon` = ten days, `calendar_type` = trading, USD unit,
and adjusted-price meaning. The seven-day generation cadence must not replace
the observation frequency.

### 5. Crypto

```text
Forecast the BTC/USD spot price, not futures or funding rates. The target is measured in USD per BTC. There is one observation every 1 hour. Predict the next 24 hours. The timestamps use UTC. The target column is spot_close. I will upload crypto.csv in csv format.
```

Check spot meaning, USD per BTC unit, one-hour observation frequency,
24-hour horizon, UTC timezone and target column. Do not invent exchange,
leverage, derivatives metadata or actual future prices.

### 6. Weather and economics together

```text
Forecast hourly electricity demand for Lahore. The target is measured in megawatts. Each observation covers 1 hour. Predict the next 48 hours. The target column is load_mw. Only historical observations of temperature_c are available. I will upload demand.csv in csv format. This forecast will support dispatch planning.
```

Check demand as target, MW unit, hourly frequency, 48-hour horizon,
`past_covariates` including temperature_c and dispatch-planning goal.
Do not mark historical temperature as a known future covariate.

### 7. Unknown requirements

```text
I want a forecast of inflation, but I do not know the data source, column names, frequency or forecast horizon yet.
```

Check that missing information is asked for instead of fabricated. Unknown
values are not the same as explicit indifference (`dont_care`).

## Download and test on another PC

The final adapter weights and configuration are now published in
[`adapters/qwen35-2b-v6-20261003-step726`](adapters/qwen35-2b-v6-20261003-step726/README.md).
The actual 67.3 MB weights are stored directly in Git. In your checkout, use:

```powershell
git pull --ff-only origin experiment/v5-grounded-20260919
```

For a new checkout, clone this branch:

```powershell
git clone --branch experiment/v5-grounded-20260919 --single-branch https://github.com/AbdullahUsman0/SLM-FineTuning-Testing.git
cd SLM-FineTuning-Testing
```

Git LFS is unnecessary for this adapter: the repository's exhausted LFS budget
prevented that upload, so the actual file is included in a regular clone/pull.

The 79-slot contracts come from the sibling `../fpy` repository. If absent,
clone it, then pin it to the training commit. Use a clean sibling checkout;
preserve any unrelated local work in an existing `fpy` checkout.

```powershell
git clone https://github.com/int-abd-5/fpy.git ../fpy
git -C ../fpy checkout --detach 04d52c015d1e3ecdefe92b87116f209361509b4b
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r training/requirements-inference.txt
& .\.venv\Scripts\python.exe -m pip install -e ../fpy
```

The commands above use the checked-in inference dependency versions. For an
NVIDIA GPU, use a CUDA-enabled PyTorch installation compatible with your driver;
the training PC used `torch 2.11.0+cu128`. Check GPU availability before choosing
`--device cuda`:

```powershell
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
& .\.venv\Scripts\python.exe scripts/chat-peft.py `
  --variant custom `
  --adapter adapters/qwen35-2b-v6-20261003-step726 `
  --adapter-manifest research-checkpoints/v6-training-20261003/results/completion.json `
  --base-model Qwen/Qwen3.5-2B `
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc `
  --prompt-version v5 --device cuda --dtype bfloat16 --max-new-tokens 1024 --debug
```

For CPU inference, replace `--device cuda --dtype bfloat16` with
`--device cpu --dtype float32`; allow extra RAM and slower generation.
The NPU is not used by this script. The first launch downloads the separate
stock base weights (approximately 4.55 GB) and tokenizer from the pinned Hugging
Face revision, unless already cached. No API key or paid API call is required.
Set `HF_HOME` before launching if you already keep that cache elsewhere.

The expected hashes are in the completion record:

- Weights: `e7add7b8b02d24615ed89067c7761c3985a4a6afb7c84ca2a72e3b06fd84e74d`
- Config: `38f60de9678324e91fa882ed86c70d09e4a39b27ae28884374ddc3e6fece15ef`

The published weights and config are copied without modification from the
previously audited adapter. Optimizer checkpoints are unnecessary for inference.
