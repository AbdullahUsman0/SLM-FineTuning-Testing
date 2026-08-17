param(
    [switch]$Evaluation
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $projectRoot "runtime\llama.cpp-b10278\llama-server.exe"
$model = Join-Path $projectRoot "models\Qwen3.5-0.8B-Q4_0.gguf"

if (-not (Test-Path -LiteralPath $server)) {
    throw "llama-server is missing: $server"
}

if (-not (Test-Path -LiteralPath $model)) {
    throw "Model is missing: $model"
}

$contextSize = if ($Evaluation) { 8192 } else { 4096 }
$parallelSlots = if ($Evaluation) { 1 } else { 4 }

& $server `
    --model $model `
    --alias local-qwen `
    --host 127.0.0.1 `
    --port 8080 `
    --ctx-size $contextSize `
    --parallel $parallelSlots `
    --threads 4 `
    --reasoning off
