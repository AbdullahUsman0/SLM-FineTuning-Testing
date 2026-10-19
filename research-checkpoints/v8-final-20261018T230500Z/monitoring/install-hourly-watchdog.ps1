$ErrorActionPreference = 'Stop'
$control = 'D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z'
$active = Get-Content -LiteralPath 'D:\SLM\active-v8-final.json' -Raw | ConvertFrom-Json
$taskName = 'SLM-v8-final-hourly-20261018T230500Z'
$config = [ordered]@{
    repo = 'D:\SLM\SLM-FineTuning-Testing'
    run = $active.run
    control = $control
    python = $active.python
    followup_source = $active.followup_source
    followup_config = $active.followup_config
    run_fingerprint = '541805a394f01f67a3b01e4bf5ca5bf623466c0e8e002a7eeb3a3510c46dc47a'
    watchdog_sha256 = (Get-FileHash -LiteralPath "$control\hourly-watchdog.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    interval_seconds = 3600
    stall_seconds = 3600
    max_recoveries_per_24h = 2
    task_name = $taskName
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText("$control\hourly-watchdog-config.json", ($config | ConvertTo-Json -Depth 8), $utf8)
$launcher = @'
$ErrorActionPreference = 'Stop'
$control = 'D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z'
$config = Get-Content -LiteralPath "$control\hourly-watchdog-config.json" -Raw | ConvertFrom-Json
& $config.python -B "$control\hourly-watchdog.py" --config "$control\hourly-watchdog-config.json" >> "$control\hourly-task-console.log" 2>&1
exit $LASTEXITCODE
'@
[IO.File]::WriteAllText("$control\check-hourly-hidden.ps1", $launcher, $utf8)
$principalName = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $control + '\check-hourly-hidden.ps1"') -WorkingDirectory $control
$hourly = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Hours 1)
$logon = New-ScheduledTaskTrigger -AtLogOn -User $principalName
$principal = New-ScheduledTaskPrincipal -UserId $principalName -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Task already exists; preserve existing schedule rather than overwrite it.' }
try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($hourly,$logon) -Principal $principal -Settings $settings -Description 'Hourly local-only fixed v8 run monitor. Verify SHA checkpoints; same-run recovery only; no paid APIs.' | Out-Null
    Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath "$control\hourly-task.xml" -Encoding Unicode
    Start-ScheduledTask -TaskName $taskName
    $receipt = [ordered]@{status='registered_and_started'; task=$taskName; installed_UTC=[DateTime]::UtcNow.ToString('o'); user=$principalName; hourly_seconds=3600; additional_trigger='AtLogOn'; requires_logged_in_session=$true; credentials_stored=$false}
    [IO.File]::WriteAllText("$control\hourly-installation.json", ($receipt | ConvertTo-Json), $utf8)
    $receipt | ConvertTo-Json
} catch {
    $receipt = [ordered]@{status='task_registration_failed'; task=$taskName; at_UTC=[DateTime]::UtcNow.ToString('o'); reason=$_.Exception.Message}
    [IO.File]::WriteAllText("$control\hourly-installation.json", ($receipt | ConvertTo-Json), $utf8)
    throw
}
