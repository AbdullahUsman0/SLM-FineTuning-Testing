$ErrorActionPreference = 'Stop'
$control = 'D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z'
$config = Get-Content -LiteralPath "$control\hourly-watchdog-config-r2.json" -Raw | ConvertFrom-Json
& $config.python -B "$control\hourly-watchdog-r2.py" --config "$control\hourly-watchdog-config-r2.json" >> "$control\hourly-task-console-r2.log" 2>&1
exit $LASTEXITCODE
