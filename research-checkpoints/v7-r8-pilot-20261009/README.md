# Authorized v7 r8 training pilot

The user explicitly approved starting the proposed pilot after receiving the remote r8 review findings. This authorizes the limited pilot with the disclosed filename-language and synthetic-template limitations. Individual human sample and overlap reviews remain pending; their ledger and the original materializer were not changed or falsely signed. A separate pilot-only authorization record documents why these inputs were prepared.

The pilot is running on the remote RTX A4000 using the cached stock Qwen/Qwen3.5-2B at revision 15852e8c16360a2fea060d615a32b45270f8a8fc. No v5/v6 adapter initializes this run. Its frozen r8 source manifest is 4e59ba3eafa689325f94b108719da02b955e5c4f5de5571f208053e4c905da72 and its prompt is v7-user-approved-extraction-1.

One complete ten-conversation entity group per domain per split supplies 200 training / 200 validation conversations, 1040 extraction rows each. Every input row is byte-identical to its source gzip line. Each split preserves 80/10/10 English/Roman Urdu/mixed and 30/60/10 weather/economics/combined mixtures. Groups remain disjoint.

Training uses one epoch (65 optimizer steps), BF16, LoRA rank 16/alpha 32/dropout .05, learning rate 5e-5, batch 1, accumulation 16, seed 42, four torch threads, no early stopping or truncation, maximum length 3584, and complete resumable checkpoints every ten steps. The pinned trainer is unchanged. The supervisor requests system-awake state while alive, records failures without automatic restart, and verifies checkpoint and final-adapter hashes on completion.

Run directory:

```text
D:\SLM\FYP-model-runs\qwen35-2b-lora-v7-r8-pilot-20261010T002204Z
```

Read pilot-status.json, console.log, stderr.log and pilot/events.jsonl there for live progress. D:/SLM/active-v7-pilot-run.txt points to this run. On completion, completion.json records verified adapter hashes, metrics and checkpoint steps. The adapter named best-adapter is the experimental final-step pilot adapter; no behavioral winner has been selected.

This commit records startup provenance, not completed results. The pretraining behavioral protocol requires fresh stock/v6/pilot comparisons with the same v7 prompt and approved convention before a larger round is decided. The actual evaluator must be frozen before scored calls. Loss alone cannot establish conversational quality.

No full training round, historical benchmark changes, final-label access/generation or paid APIs were started. Base weights, optimizer files, credentials and caches are excluded from this publication.
