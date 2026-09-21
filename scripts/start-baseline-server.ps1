param(
    [ValidateSet("08b", "2b")]
    [string]$Model = "2b",
    [ValidateSet("cpu", "vulkan")]
    [string]$Backend = "cpu",
    [int]$Threads = 8,
    [int]$ContextSize = 4096
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$server = Join-Path $root "runtime\llama-b11026-$Backend\llama-server.exe"
$gpuLayers = if ($Backend -eq "vulkan") { 99 } else { 0 }

if ($Model -eq "2b") {
    $modelFile = Join-Path $root "models\Qwen3.5-2B-Q4_K_M.gguf"
    $alias = "local-qwen35-2b-base"
    $port = 8083
} else {
    $modelFile = Join-Path $root "models\Qwen3.5-0.8B-Q4_0.gguf"
    $alias = "local-qwen35-08b-base"
    $port = 8084
}

foreach ($path in @($server, $modelFile)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required runtime artifact is missing: $path"
    }
}

Write-Host "Serving $alias with $Backend on http://127.0.0.1:$port (Ctrl+C to stop)"
& $server `
    --model $modelFile `
    --alias $alias `
    --host 127.0.0.1 `
    --port $port `
    --ctx-size $ContextSize `
    --parallel 1 `
    --threads $Threads `
    --n-gpu-layers $gpuLayers `
    --reasoning off
