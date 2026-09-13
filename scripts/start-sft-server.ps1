param(
    [ValidateSet("v1", "v2")]
    [string]$Variant = "v2",
    [switch]$Evaluation
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $projectRoot "runtime\llama.cpp-b10278\llama-server.exe"
$model = Join-Path $projectRoot "models\Qwen3.5-0.8B-Q4_0.gguf"
$adapter = Join-Path $projectRoot "models\Qwen3.5-0.8B-LoRA-$Variant-f16.gguf"
$alias = "local-qwen-$Variant"
$port = if ($Variant -eq "v2") { 8081 } else { 8082 }

foreach ($requiredFile in ($server, $model, $adapter)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required local model artifact is missing: $requiredFile"
    }
}

$contextSize = if ($Evaluation) { 8192 } else { 4096 }

Write-Host "Starting $Variant on http://127.0.0.1:$port (press Ctrl+C to stop)..."
& $server `
    --model $model `
    --lora $adapter `
    --alias $alias `
    --host 127.0.0.1 `
    --port $port `
    --ctx-size $contextSize `
    --parallel 1 `
    --threads 4 `
    --reasoning off
