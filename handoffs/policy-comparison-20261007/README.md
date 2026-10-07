# Temporary GPU handoff for the reviewed-case comparison

This directory lets the remote GPU PC pull the portable evaluation bundle from
GitHub. It adds no training or production inference changes.

## Run on the GPU PC

1. Pull `experiment/v5-grounded-20260919` normally, preserving local changes.
2. Verify the ZIP SHA-256, then extract it into a NEW directory:

```powershell
Set-Location 'D:\SLM\SLM-FineTuning-Testing'
git pull --ff-only origin experiment/v5-grounded-20260919
$zipPath = '.\handoffs\policy-comparison-20261007\policy-gpu-handoff-20261007.zip'
$expected = 'b6667d9a2467a4f58724e67165a601e9420fe6d38d70d3a9939591135aa0bf99'
if ((Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'ZIP checksum mismatch' }
Expand-Archive -LiteralPath $zipPath -DestinationPath 'D:\SLM\policy-comparison-20261007'
Set-Location 'D:\SLM\policy-comparison-20261007'
.\scripts\run-policy-gpu.ps1 -PrepareOnly
.\scripts\run-policy-gpu.ps1
```

If the extraction directory already exists, inspect it first and use a new empty
folder; do not overwrite prior runs. The existing CUDA environment/adapter paths
are defaults in the script. See [REMOTE_GPU_PROMPT.md](REMOTE_GPU_PROMPT.md) for
Codex instructions, pinned dependency checks, controls and returning results.

## Analysis first

[Verified remote handoff analysis](REMOTE_HANDOFF_ANALYSIS_9fbcadd.md) explains why
XGrammar's published 32-case scores cannot yet be compared with the new reviewed
31-case OpenAI results. The old inner-JSON guard rejects ordinary text which the
new common parser admits. This does not repair semantic omissions or corrections.

The standalone ZIP preserves the exact code/parser/cohort snapshot. Run the
extracted scripts rather than substituting latest repository modules. It contains
73 verified files, including the original pinned fpy source, and excludes model
weights, API keys, training data and final-test labels. Both fresh tracks have
31 scenarios and 93 extraction calls.

The OpenAI side is already complete: component F1 0.4844, rollout F1 0.5086,
estimated usage cost $0.1598171. A matched SLM result is still pending. No winner
is claimed. The cohort is agent-reviewed posthoc development.

These transfer files can be removed in a later commit after the remote PC has
copied them; existing experimental checkpoints should remain preserved.
