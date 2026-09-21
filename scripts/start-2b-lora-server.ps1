param(
    [int]$Threads = 8,
    [int]$ContextSize = 4096,
    [int]$Port = 8085
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$server = Join-Path $root "runtime\llama-b11026-cpu\llama-server.exe"
$modelFile = Join-Path $root "models\Qwen3.5-2B-Q4_K_M.gguf"
$adapterFile = Join-Path $root "training-runs\qwen35-2b-lora-v3-repro-20260918\adapter-F16-LoRA.gguf"

foreach ($path in @($server, $modelFile, $adapterFile)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required runtime artifact is missing: $path"
    }
}

Write-Host "Serving experimental Qwen3.5-2B v3 LoRA on http://127.0.0.1:$Port (Ctrl+C to stop)"
& $server `
    --model $modelFile `
    --lora $adapterFile `
    --alias local-qwen35-2b-v3-lora `
    --host 127.0.0.1 `
    --port $Port `
    --ctx-size $ContextSize `
    --parallel 1 `
    --threads $Threads `
    --n-gpu-layers 0 `
    --reasoning off
