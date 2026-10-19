# Hourly local monitoring of the existing v8 final run

Installed on the GPU PC on 2026-10-19 UTC, without changing frozen training sources, settings, counters, checkpoints or the run path.

Task: `SLM-v8-final-hourly-20261018T230500Z`. Runs a hidden one-shot check every hour, plus at this user's Windows sign-in. The initial task execution returned success (0); see schedule-verification.json and initial-live-check.json. No credentials are stored. Windows must remain signed in; a locked screen is supported. Checks cannot run while the PC is off. After a reboot, signing in triggers a check.

Each check verifies frozen launch inputs, run identity and the newest valid complete checkpoint's entire SHA-256 inventory. It checks actual process command lines, current output age and both supplied-worker and followup status. A stale PID or status file alone does not establish that training is alive. The two Python process IDs per venv launch are a launcher and its child, not two training jobs.

If the worker has exited and no trainer/development child is alive, it saves an incident and calls the unchanged supplied `run.py start --run SAME_PATH`. Training automatically resumes from a verified checkpoint, never from zero. At most two recovery attempts are permitted per rolling 24 hours; further failures remain preserved for review. A live process with no output for over one hour is recorded as `stalled_requires_review`; it is never killed automatically. Long validation can also trigger this flag. Scientific/publication failures remain preserved for review rather than being blindly retried. Completed v8 development does not trigger a duplicate diagnostic while the followup handles v6.

Original worker and followup continue to inhibit system sleep while permitting screen lock. The monitor uses no GPU inference and no paid APIs. It does not access final-test labels. No training interruption was induced to test recovery. Nine offline tests passed, including rejected corrupt/missing checkpoint selection, orphan process protection and an actual harmless hidden-child launch with spaces in its path. Live task execution verified checkpoint 120 and observed optimizer step 125 without launching any worker.

Live files (outside Git):

- `D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z\hourly-last-check.json`
- `D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z\hourly-checks.jsonl`
- `D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z\hourly-task-console.log`
- `hourly-recoveries.jsonl` and unique `hourly-recovery-*` folders in the same control directory, if recovery becomes necessary.

Read the latest check in PowerShell:

```powershell
Get-Content D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z\hourly-last-check.json
```

Local records are automatic; the task does not send chat/email notifications or create hourly Git commits. This folder is an installation snapshot, not a live status feed. Development results remain pending; training loss establishes no behavioral winner.