# Repeated reboot after checkpoint-1940 restoration

Windows booted again at 2026-10-22 07:49:01 PDT. The existing automatic recovery had successfully restored checkpoint 1940, replayed the prior unsaved steps, and reached 1945. Its success through 1943 was verified and published before the later reboot. None of steps 1941-1945 were durably checkpointed: checkpoint 1940 remains the latest complete saved state. Five optimizer steps must be replayed if training resumes.

Actual worker/trainer processes were absent at inspection. Their old status/heartbeat files were stale. The monitor correctly stopped automatic retries at the unchanged limit of two attempts per rolling 24 hours. No journal was reset and no additional manual restart was launched during this status inspection. No power or training setting was changed.

All 11 checkpoint-1940 file hashes, original run fingerprint, and frozen source/input locks were independently verified again. The reboot cause remains unresolved. Successful optimizer restoration is not a hardware-reliability fix. Training, final adapter, remaining validation and development comparisons remain incomplete; a dependable completion ETA cannot be established while these reboots continue.

Complete checkpoints and logs remain preserved locally. This metadata record includes no weights, optimizer states, credentials, caches, final-test labels or paid API calls.
