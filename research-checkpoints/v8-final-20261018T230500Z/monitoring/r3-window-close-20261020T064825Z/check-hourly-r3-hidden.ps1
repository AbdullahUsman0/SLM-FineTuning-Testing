$ErrorActionPreference = 'Stop'
$control = 'D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z'
$config = Get-Content -LiteralPath "$control\hourly-watchdog-config-r3.json" -Raw | ConvertFrom-Json
& $config.python -B "$control\hourly-watchdog-r3.py" --config "$control\hourly-watchdog-config-r3.json" >> "$control\hourly-task-console-r3.log" 2>&1
exit $LASTEXITCODE
