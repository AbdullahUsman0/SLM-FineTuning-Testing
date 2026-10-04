# V6 experimental run

Explicit user request authorizes training; independent corpus and overlap review remain pending.
The 10,219 overlap flags remain unresolved; no semantic leakage or generalization clearance is claimed.
Development artifacts were machine-verified and copied byte-for-byte without accessing final labels.
A fresh specialist starts from stock Qwen/Qwen3.5-2B, not the previous generalist.
Matched behavioral pilot not run; user requested full background experimental fine-tuning.
Two epochs, LR5e-5, effective batch16, max_length3584, checkpoint every10, no truncation or early stopping.
Final step adapter is not a behaviorally selected winner. Teacher-forced loss is not task-quality evidence.
Training is pinned in an isolated detached worktree; checkpoints, weights, environment and cache are local.
On reboot, the login recovery task starts the same guarded supervisor and resumes a hash-verified checkpoint.

Recovery from checkpoint 260 after Windows restart; interrupted attempt had reached 261. Source and input hashes reverified; same hyperparameters.
