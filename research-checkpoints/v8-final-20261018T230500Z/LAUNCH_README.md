# Historical launch snapshot: v8 training is still in progress

The fixed two-epoch experimental run started from source commit bd36ff605d1d7e6c5f79de6e65d60785e0afe49d. LAUNCH_STATUS.json records an observed live CUDA worker and an independently SHA-verified complete checkpoint. This snapshot is not a completed training result or development evaluation. No final adapter or measured development scores exist at this launch snapshot.

The exact run and environment paths refer to the GPU training PC. Read LAUNCH_HANDOFF.md for live status and recovery commands. The supplied worker owns training and the v8 diagnostic; the detached companion owns the same-cohort v6 diagnostic, local adapter packaging and final scientific-record publication. The final-step best-adapter folder will not be behaviorally selected. No paid API calls or final-test labels are authorized. Human annotation remains pending; training loss/token accuracy is not extraction F1.

When completion.json, comparison.json and README.md arrive later, use those for final development measurements. Keep this dated launch record as provenance. Do not start a duplicate trainer because the public record is a snapshot. The frozen training/diagnostic code and settings were not changed while creating this metadata. No weights or optimizer states are published here.
