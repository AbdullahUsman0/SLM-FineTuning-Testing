# Unexpected reboot and same-run recovery

Windows recorded an unexpected reboot (Kernel-Power41/EventLog6008). Its underlying cause is not established. Both detached workers had exited, while their old status files still said running. Those stale files and system events are preserved here.

The last pre-interruption observed step was125. Twelve complete checkpoint markers remain; checkpoint120 was selected and all its artifact hashes verified. Training resumed in the SAME original run directory using the unchanged supplied worker and --resume-from-checkpoint auto. The run fingerprint is unchanged, and new optimizer progress beyond120 is recorded in resume-confirmation.json. The five previously observed but unsaved steps121-125 are replayed. No step counter, hyperparameter, source, input, epoch count or checkpoint was edited or reset.

The first auxiliary companion relaunch did not remain alive; its source/config were preserved and it was relaunched using hidden Windows Start-Process. Fresh heartbeats were observed beyond one minute after that launch. The supplied training worker was not duplicated or changed. The identical v8/v6 development diagnostics, local adapter packaging and scientific-record publication remain queued. No paid APIs or final-test labels were accessed. Development results remain pending.

The initial post-resume progress-bar ETA is misleading because it jumps over restored steps. Use sustained optimizer timing to estimate remaining work; do not interpret the initial displayed minutes as a real completion estimate.
