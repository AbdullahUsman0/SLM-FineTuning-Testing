# Prompt for the agent on the remote GPU PC

Continue my FYP requirement-extraction project on THIS GPU server. You have my authorization to pull the prepared final dataset, run the fixed full-data experiment on the local CUDA GPU, keep the job working in the background while my client laptop is locked/disconnected, and save the results. Execute the work; do not stop after giving instructions. Do not claim that a model won before the measurements exist.

Repository: https://github.com/AbdullahUsman0/SLM-FineTuning-Testing
Branch: `experiment/v5-grounded-20260919`
Expected Windows checkout: `D:\SLM\SLM-FineTuning-Testing`. Locate it if different. Preserve existing edits, runs, caches and adapters. Fetch and fast-forward the correct branch; if the checkout is dirty or diverged, use a separate clean checkout/worktree rather than resetting or discarding work. Never force-push.

## Read and verify first

1. Read `training/final_round/README.md`, `EXPERIMENT_PLAN.md` and `TRAINING_AUTHORIZATION.json`.
2. Dataset: `corpus-v8/v8-final-20261018-r1`. Manifest SHA-256: `abedd97e642152786db56d3e98a0f997d1c241bf6549365c56ddd01ac5cab2e9`.
3. Verify 3,600 training conversations / 17,860 examples and 800 DEVELOPMENT-validation conversations / 3,980 examples. Keep 30% weather, 60% economics, 10% combined and 80% English, 10% Roman Urdu, 10% mixed at conversation level.
4. This is synthetic experimental development data. Machine checks are complete; independent human review is NOT complete. My training authorization allows this experimental run, but you must not fabricate a review ledger or mark the labels human-approved. Do not modify the older review gates.
5. Do not load sealed final labels, the private 240-conversation prospective candidate or its labels. Do not train on validation. Do not use or transmit API keys. No new paid API calls or rented compute are authorized.

## Establish the GPU environment

Inspect the OS, CUDA driver, GPU/VRAM, free RAM, free output-disk space and existing virtual environment. Reuse the previous working CUDA environment if its dependencies match `training/final_round/requirements.txt`; otherwise build an isolated environment. Preserve CUDA PyTorch. The historical working stack is documented in the README. Use the official wheel source suitable for the driver if installation is necessary. Record exact versions and freeze dependencies.

Use a dedicated clean fpy dependency checkout at commit `04d52c015d1e3ecdefe92b87116f209361509b4b`. Do not reset a sibling fpy containing unrelated work. The helper selects this source explicitly. Use the existing `D:\SLM\hf-cache` if available. Expose one GPU with CUDA_VISIBLE_DEVICES=0. The trainer requires CUDA BF16; do not silently switch to CPU, NPU, offload, quantization, a different architecture or a different model revision.

Run `training/final_round/test_final_round.py --fpy <pinned-checkout>`, then `data_tools.py --fpy <pinned-checkout>` using the training Python. Verify all dataset checks before model loading. Use the exact commands from the README with paths adjusted for this machine.

## Run the frozen experiment

Create a NEW unique directory such as `D:\SLM\FYP-model-runs\qwen35-2b-lora-v8-final-<UTC timestamp>`. Record its exact path outside Git. Run `run.py prepare --fpy <pinned-checkout> --run <new-run>` and inspect the complete tokenizer preflight. It must cover every example, disable thinking and prohibit truncation. Verify the trainer's actual expected optimizer-step schedule, single-GPU effective batch 16 and available disk. If the checks fail, fix the technical problem or report it precisely; never bypass a checksum or invent a passing receipt.

Train fresh LoRA on pinned stock `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. Do not stack on an earlier adapter. Use the frozen two epochs, seed 42, learning rate 5e-5, rank16/alpha32/dropout0.05, completion-only loss, max allowed length4096, no truncation, early-stopping patience0 and COMPLETE checkpoints every ten optimizer steps. Preserve all checkpoints. Expected optimizer steps: 2,234. Do not add epochs or alter labels/hyperparameters after seeing results.

Use `run.py start --run <same-run>` so a detached worker owns training and its local development evaluation. On Windows it also inhibits idle sleep without keeping the display on. On Linux use an appropriate persistent service/session and sleep inhibitor where supported. Screen lock must not be a dependency of the process. It cannot survive the server sleeping, powering off or losing its login session under a policy that kills user jobs; explain any relevant machine restriction. My client laptop can lock/disconnect after the independent remote job is verified.

Do not launch duplicate trainers. Verify a real advancing global step and a COMPLETE checkpoint, not just an assigned PID. Check the worker logs and GPU process. If interrupted, inspect the failure and start the SAME run with unchanged code and inputs. Resume only from a complete SHA-attested checkpoint. Never manually edit global_step or restart from zero while claiming continuation. Preserve failures and partial checkpoints. Technical repairs must be documented; changes to the frozen experiment need a new version/run rather than silently editing an in-progress run.

## Save and check results

The worker saves manifests, losses, optimizer progress, adapter/checkpoint hashes and logs, then runs the versioned 100-conversation development diagnostic in both component and rollout tracks. Inspect `training-completion.json` and `development-evaluation/summary.json`. Retain all raw calls and the context/gold hashes; resume the diagnostic's journal instead of discarding partial work. Check loading used the adapter's recorded model architecture and active LoRA parameters, with no missing adapter keys or CPU offload.

When training and its diagnostic finish, run the SAME diagnostic once on the existing v6 adapter using the README command and a separate new output directory, if that verified adapter is available. Locate and verify its recorded hashes before using it. Keep the same corpus, selection, instructions, decoding and admission policy. If the v6 files are missing, report that instead of substituting another model. Save a side-by-side DEVELOPMENT comparison of exact F1, precision/recall, JSON/schema validity, admitted calls, extra/forbidden slots, correction state, complete state and latency. Report English/Roman Urdu/mixed and domain breakdowns and identify regressions. Do not compare these new numbers directly with old cohorts. This script does not implement a new final significance test, so do not invent confidence intervals or a definitive winner.

Package the new adapter locally with its config, training completion receipt, README and SHA-256 file. Verify the archive by reading its members and comparing extracted hashes. Keep optimizer checkpoints available separately for resume. Do not put weights, datasets, credentials or large checkpoints into normal Git history. Prepare a compact, credential-checked results folder under `research-checkpoints/v8-final-<timestamp>` containing provenance, summaries, exact commands, interpretation, failure analysis and compressed raw diagnostic outputs with a manifest. Commit and push only those new scientific result records and reproduction scripts to the experiment branch; preserve all unrelated files. If upload or credentials block publication, retain the local verified bundle and report the exact blocker. Do not publish private final labels.

Finish with the branch/commit, GPU and runtime, completed/expected steps, run path, verified checkpoint count, adapter location/hashes, new development results, known limitations and any pending tasks. Training loss or token accuracy is not extraction F1. The final-step folder name `best-adapter` does not mean behaviorally selected. Independent human validation and the later held-out OpenAI comparison remain separate pending work, with paid API budget still zero.
