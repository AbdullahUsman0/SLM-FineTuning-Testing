"""Read the live v6 run without changing trainer state."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import psutil

run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(Path(r'D:\SLM\active-v6-training-run.txt').read_text().strip())
status = json.loads((run / 'monitor-status.json').read_text(encoding='utf-8'))
status['run'] = str(run)
if (run / 'process.json').exists():
    process = json.loads((run / 'process.json').read_text(encoding='utf-8'))
    try:
        actual = psutil.Process(process['trainer_pid'])
        status['trainer_alive'] = actual.is_running() and any(Path(arg).name == 'train_lora.py' for arg in actual.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        status['trainer_alive'] = False
    timestamp = status.get('at_utc', status.get('completed_utc'))
    status['status_age_seconds'] = (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()
    status['status_is_stale'] = status.get('status') not in ('complete', 'training_complete_smoke_failed') and status['status_age_seconds'] > 90
    text = (run / 'full-console.log').read_text(encoding='utf-8', errors='replace')
    # Earlier progress belongs to the interrupted attempt, not the resumed one.
    text = text.rsplit('Resume decision:', 1)[-1]
    progress = re.findall(r'(\d+)/726 \[([^\]\r\n]+)\]', text)
    if progress:
        status.update(step=int(progress[-1][0]), total_steps=726, progress_timing=progress[-1][1])
    training_start = re.search(r'\d+/726 \[', text)
    evaluation = re.findall(r'(\d+)/1253 \[([^\]\r\n]+)\]', text[training_start.start():]) if training_start else []
    if evaluation:
        status['last_observed_validation_progress'] = {'step': int(evaluation[-1][0]), 'total': 1253}
events = run / 'full/events.jsonl'
if events.exists():
    metrics = [event for line in events.read_text(encoding='utf-8').splitlines()
               if (event := json.loads(line)).get('event') == 'metrics']
    if metrics:
        status['latest_metrics_event'] = metrics[-1]
print(json.dumps(status, indent=2))
