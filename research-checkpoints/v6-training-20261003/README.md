# V6 weather/economics specialist: experimental background training

The user requested analysis of v5 results, a pull of the latest code, and
background fine-tuning on the newly generated domain-specific v6 corpus.
The latest source pulled was `afccf38` on `experiment/v5-grounded-20260919`.

Training uses a new specialist adapter from stock Qwen/Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. It does not continue the v5 adapter.
The unchanged 79-slot contract extracts forecasting requirements and asks
clarifications; it does not predict weather or economic measurements.

Frozen data: `corpus-v6/v6-20261002-r4`, manifest SHA-256
`0782956c587b5996a2df3f7b799fb0b9e572942b063c8fc1b623ea691fc5aeb9`.
The source mixture is 30% weather, 60% economics, and 10% combined requests.
The development verifier checked 1,700 sources and 7,053 examples, with zero
exact/normalized full model-input duplicates and zero entity/template families
crossing splits. Final labels were not accessed.

| Input | Examples | Maximum full conversation tokens |
| --- | ---: | ---: |
| Training | 5,800 | 3,385 |
| Validation | 1,253 | 3,347 |

All examples fit the 3,584-token limit without truncation. Token counting uses
the pinned tokenizer and `enable_thinking=False`, matching the trainer.
Settings: two epochs, learning rate 5e-5, batch 1 with gradient accumulation 16,
BF16 CUDA on RTX A4000, LoRA rank 16/alpha 32/dropout 0.05, cosine schedule,
seed 42, no early stopping, complete checkpoints every 10 optimizer steps.
Expected total: 726 optimizer steps. Every checkpoint is retained; automatic
recovery resumes only a complete hash-verified checkpoint from this run.
The final saved adapter is a final-step candidate, not a behavioral winner.

Independent human label review and fuzzy-overlap review remain pending, with
10,219 unresolved flagged pairs. The explicit user training instruction takes
precedence over the repository's review-first workflow for this experimental
run. No review ledger, approval, or resolved-leakage claim is fabricated. The
existing review-required materializer retains its approval checks; this run
materializes the verified development bytes separately and records the user's
authorization honestly. A matched behavioral pilot has not been run.

Training runs from an isolated detached worktree at
`D:\SLM\SLM-v6-training-20261003`, keeping its code stable when the main checkout
changes. The pinned fpy dependency remains
`04d52c015d1e3ecdefe92b87116f209361509b4b`.

Local run: `D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z`.
The pointer `D:\SLM\active-v6-training-run.txt` identifies this run.
Inspect `launch-config.json`, `authorization.json`, `inputs/input-lock.json`,
`token-lengths.json`, `full-console.log`, and `monitor-status.json` there.
A hidden supervisor and a login-triggered recovery task continue the run;
source/input hashes are checked before every launch. They do not load sealed
final labels or call a paid API. Weights, checkpoints and caches remain outside Git.

On completion, the supervisor verifies retained checkpoints, finite adapter
tensors and exact equality with the final checkpoint, then performs one synthetic
weather loading smoke. It records a completion summary and disables recovery.
Teacher-forced loss and loading smoke are not behavioral quality evidence.

Subsequent quality evaluation should compare stock, v5 and v6 on identical v6
validation cases, report weather/economics/combined and subdomain results with
entity-cluster uncertainty, and check v5 retention. Natural, independently
authored utterances and training-budget-matched controls are required before
strong generalization or causal specialization claims. The sealed final split
remains unused until a separately authorized final evaluation.
