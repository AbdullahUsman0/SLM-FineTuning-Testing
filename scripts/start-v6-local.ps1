param(
    [int]$Threads = 8,
    [int]$Port = 8086
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$server = Join-Path $root 'runtime\llama-b11026-cpu\llama-server.exe'
$modelFile = Join-Path $root 'models\Qwen3.5-2B-Q4_K_M.gguf'
$adapter = Join-Path $root 'runtime\adapter-downloads\v6-adapter-step726-F16.gguf'
foreach ($path in @($server, $modelFile, $adapter)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required local artifact is missing: $path"
    }
}
Write-Host "Serving quantized v6 manual tests on http://127.0.0.1:$Port (Ctrl+C to stop)"
& $server --model $modelFile --lora $adapter --alias local-qwen35-2b-v6 `
    --host 127.0.0.1 --port $Port --ctx-size 8192 --parallel 1 `
    --threads $Threads --threads-batch $Threads --n-gpu-layers 0 --reasoning off
exit $LASTEXITCODE
