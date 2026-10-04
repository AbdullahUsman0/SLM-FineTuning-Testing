param(
    [int]$Port = 8086,
    [string]$PythonPath = (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$fpySource = Join-Path $root 'runtime\fpy-v6-pinned\src'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python runtime missing: $PythonPath. Pass -PythonPath with a Python that has the fpy dependencies."
}
if (-not (Test-Path -LiteralPath (Join-Path $fpySource 'forecasting_assistant\domain\schema.py'))) {
    throw 'Pinned fpy snapshot missing. See V6_MANUAL_TEST_GUIDE.md for local setup.'
}
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
    if ($health.status -ne 'ok') { throw 'Server not ready' }
} catch {
    throw "Start scripts/start-v6-local.ps1 first; the server on port $Port is not ready."
}

$previousFpySource = $env:SLM_FPY_SRC
Push-Location $root
try {
    $env:SLM_FPY_SRC = $fpySource
    & $PythonPath scripts/chat-peft.py --variant custom `
        --adapter adapters/qwen35-2b-v6-20261003-step726 `
        --adapter-manifest research-checkpoints/v6-training-20261003/results/completion.json `
        --base-model Qwen/Qwen3.5-2B `
        --revision 15852e8c16360a2fea060d615a32b45270f8a8fc `
        --prompt-version v5 --server-url "http://127.0.0.1:$Port/v1" `
        --max-new-tokens 1024 --debug
    $chatExitCode = $LASTEXITCODE
} finally {
    Pop-Location
    $env:SLM_FPY_SRC = $previousFpySource
}
exit $chatExitCode
