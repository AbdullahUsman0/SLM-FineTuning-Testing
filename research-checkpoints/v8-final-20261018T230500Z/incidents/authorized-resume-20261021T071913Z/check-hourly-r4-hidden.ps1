$ErrorActionPreference = 'Stop'
$control = 'D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z'
$config = Get-Content -LiteralPath "$control\hourly-watchdog-config-r4.json" -Raw | ConvertFrom-Json
& $config.python -B "$control\hourly-watchdog-r4.py" --config "$control\hourly-watchdog-config-r4.json" >> "$control\hourly-task-console-r4.log" 2>&1
exit $LASTEXITCODE
