param([string]$RunDir)
$ErrorActionPreference = 'Stop'
$v6Run = if ($RunDir) { $RunDir } else { (Get-Content -LiteralPath 'D:\SLM\active-v6-training-run.txt' -Raw).Trim() }
$v6RunResolved = [IO.Path]::GetFullPath($v6Run)
if (-not $v6RunResolved.StartsWith('D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-')) { throw 'Unexpected v6 run path' }
if (Test-Path -LiteralPath (Join-Path $v6Run 'completion.json')) { exit 0 }
$v6Config = Get-Content -LiteralPath (Join-Path $v6Run 'launch-config.json') -Raw | ConvertFrom-Json
$v6Stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$v6Process = Start-Process -FilePath $v6Config.python -ArgumentList @('-u',(Join-Path $v6Run 'supervise_v6_experiment.py'),$v6Run) -WorkingDirectory $v6Config.repository -WindowStyle Hidden -RedirectStandardOutput (Join-Path $v6Run "supervisor-$v6Stamp.stdout.log") -RedirectStandardError (Join-Path $v6Run "supervisor-$v6Stamp.stderr.log") -PassThru
$v6Process.Id | Set-Content -LiteralPath (Join-Path $v6Run 'supervisor.pid')
Write-Output "V6 supervisor PID $($v6Process.Id)"
