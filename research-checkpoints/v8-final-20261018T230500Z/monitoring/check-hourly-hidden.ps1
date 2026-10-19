$ErrorActionPreference = 'Stop'
$control = 'D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z'
$config = Get-Content -LiteralPath "$control\hourly-watchdog-config.json" -Raw | ConvertFrom-Json
& $config.python -B "$control\hourly-watchdog.py" --config "$control\hourly-watchdog-config.json" >> "$control\hourly-task-console.log" 2>&1
exit $LASTEXITCODE