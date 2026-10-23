# Completed v8 training: adapter transfer and paper handoff

The fixed two-epoch run completed all 2,234 optimizer steps and its final teacher-forced validation. The supplied worker verified all 224 retained complete checkpoints. This folder preserves the training receipt, runtime/provenance, final validation metrics and adapter transfer information. It does not contain weights in Git.

The separately published experimental adapter prerelease is `v8-adapter-20261022`. Read ADAPTER_RELEASE.json for exact ZIP/checksum URLs, SHA-256, member inventory and actual publication status. It is ready to transfer only when that receipt says `published_prerelease_download_verified`. The download_adapter.py helper checks those bytes and extracts into a NEW directory; use an existing Python interpreter and `--output <new local folder>`.

Give LOCAL_CODEX_HANDOFF.md to Codex on the other PC. It covers safe checkout updates, downloading/verifying the adapter and updating the existing paper using only supported evidence. Inference version/prompt constraints are in ADAPTER_README.md and inference-dependencies.json.

At this snapshot, the v8 development diagnostic is running and the matched v6 diagnostic, behavioral summaries, failure analysis and final result packaging remain pending. Their later publication belongs in the parent results folder. This is inspected synthetic development data with incomplete independent human review, no sealed-final access, zero new paid API calls and no established model winner. Training loss/token accuracy are not extraction F1.
