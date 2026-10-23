# Message for Codex on the other PC

Continue our FYP requirement-extraction research and update the existing research-paper draft on THIS PC. Download the completed experimental v8 adapter and inspect the published evidence. Do the work, preserve existing edits/models/runs, and do not invent behavioral results.

Repository: https://github.com/AbdullahUsman0/SLM-FineTuning-Testing
Branch: experiment/v5-grounded-20260919
GPU-run identity: qwen35-2b-lora-v8-final-20261018T230500Z
Current training evidence: research-checkpoints/v8-final-20261018T230500Z/training-completed-20261023T002215Z/
Adapter prerelease: https://github.com/AbdullahUsman0/SLM-FineTuning-Testing/releases/tag/v8-adapter-20261022

1. Locate the local checkout and read applicable AGENTS.md instructions. Fetch and fast-forward the branch (`git pull --ff-only origin experiment/v5-grounded-20260919`) when safe. If edits/divergence prevent that, use a separate clean worktree/checkout. Never reset, force-push, overwrite unrelated edits, or assume a D: path on the GPU PC exists here.

2. Read this folder's ADAPTER_RELEASE.json, RESEARCH_STATUS.json, training-completion.json, training-evaluation.json, training-run-manifest.json, inference-dependencies.json and ADAPTER_README.md. Read training/final_round/REMOTE_GPU_PROMPT.md and EXPERIMENT_PLAN.md for protocol and limitations. ADAPTER_RELEASE.json must say `published_prerelease_download_verified`; otherwise pull the later publication commit before claiming the upload is ready.

3. Use the existing Python interpreter to run this folder's download_adapter.py with `--output <NEW local download directory>`. It downloads the exact ZIP and checksum listed in ADAPTER_RELEASE.json, verifies the archive SHA-256 and every member, rejects unsafe paths, and extracts into a new folder. Preserve this extracted best-adapter path for local testing. Never overwrite an existing adapter. Independently confirm these two SHA-256 values:

   - adapter_model.safetensors: c83e0f1a458125970eb70ff360ebead02618f2f48b5bbb59e2ca91a5b7c4c0df
   - adapter_config.json: a1ce14418c5c22375ccfd773338ef1b09117ec9b9e72db034c71ade886b4664c

4. This is an EXPERIMENTAL FINAL-STEP v8 LoRA. Load it only on stock Qwen/Qwen3.5-2B revision 15852e8c16360a2fea060d615a32b45270f8a8fc, using its recorded Qwen3_5ForConditionalGeneration architecture. The adapter config's revision is null, so explicitly pin the base in the loader. Do not stack it onto v5/v6/v7. The package excludes base weights, optimizer checkpoints and caches. Reuse the local compatible environment; do not replace CUDA PyTorch or start training. A non-CUDA laptop may download and verify the artifact without running the CUDA diagnostic. Any later CPU/GGUF manual test is a different runtime, not a substitute for the research comparison. Older v5/v6 interactive prompts are not automatically the frozen v8 prompt; use corpus-v8 instructions/schema and the supplied evaluator for matched testing.

5. Locate the EXISTING paper/manuscript we drafted on this PC. Preserve a backup/version history and update that draft, its tables and supporting experiment notes. If it cannot be found, report the missing draft rather than fabricating a replacement paper. Add the v8 experiment as a separately identified synthetic-development cohort. Verified facts available now:

   - Dataset corpus-v8/v8-final-20261018-r1; manifest SHA-256 abedd97e642152786db56d3e98a0f997d1c241bf6549365c56ddd01ac5cab2e9.
   - 3,600 training conversations / 17,860 training examples; 800 development-validation conversations / 3,980 validation examples. No training on validation and no truncation. Conversation-level mixture: 30% weather, 60% economics, 10% combined; 80% English, 10% Roman Urdu, 10% mixed.
   - Fresh stock-base LoRA, rank16/alpha32/dropout0.05, two epochs, seed42, initial LR5e-5, effective batch16, BF16 CUDA, completion-only loss, early stopping disabled. GPU NVIDIA RTX A4000 16GB. Pinned fpy revision 04d52c015d1e3ecdefe92b87116f209361509b4b. Use the recorded manifest for exact runtime versions.
   - 2,234/2,234 optimizer steps completed, with all 224 complete checkpoints verified by the supplied worker. This was one run resumed from complete optimizer/RNG checkpoints after multiple unexpected Windows reboots; do not describe it as uninterrupted or treat reboot downtime/checkpoint hashing as training throughput. The reboot cause was not established.
   - Final teacher-forced development-validation loss 0.0033881000708788633; mean token accuracy 0.999569866570396. The existing trainer evaluates at the final epoch and explicitly evaluates again. These are teacher-forced metrics, not extraction F1 or conversation success. Folder name best-adapter means fixed final step, not behavioral selection. Independent human review is incomplete.

6. Behavioral evaluation was still running when this training-only snapshot was prepared. Do not add extraction F1, JSON success rates, invented-slot rates, correction/unknown handling, latency comparisons, confidence intervals or winner claims until their completed evidence exists. The remote worker and followup will later push results under research-checkpoints/v8-final-20261018T230500Z/: completion.json, comparison.json, summary.csv, granular-domain-breakdown.json, v8/summary.json, v6/summary.json and compressed raw journals/failure findings. Pull again then. Require completion.json to state `verified_training_and_local_development_complete`, inspect v6_status, and match corpus/case-selection hashes, instructions, decoding and admission policy across arms before using a paired comparison. Report regressions as well as improvements, separate component from rollout, and keep English/Roman Urdu/mixed and domain breakdowns distinct. Missing v6 results remain missing.

7. Preserve earlier paper tables and label their own cohorts/protocols. Do not compare older v5/v6 scores directly with this new cohort. The new diagnostic is 100 inspected synthetic development conversations, not a new final test. Independent human annotation, sealed final testing and a final hosted/OpenAI comparison remain pending. No new confidence-interval/significance method was implemented. No paid API calls, new fine-tuning, private 240-conversation cohort, or sealed final labels are authorized.

Return the actual local adapter path, verified ZIP checksum, paper files changed, updated methods/training-results tables, source files for each number, and a clear list of pending behavioral comparisons. Do not claim every result is available merely because the adapter can be downloaded.
