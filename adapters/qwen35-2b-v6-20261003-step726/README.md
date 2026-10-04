# Qwen3.5-2B v6 specialist adapter, final step 726

This directory contains the actual LoRA weights in Git and their unchanged
PEFT configuration. They are byte-identical to the completed v6 run's final
adapter and checkpoint 726. See [manifest.json](manifest.json) for hashes and
provenance, and [training results](../../research-checkpoints/v6-training-20261003/results/README.md)
for the complete audit and limitations.

Base: `Qwen/Qwen3.5-2B`, revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`.
Architecture: `Qwen3_5ForConditionalGeneration`. LoRA rank 16, alpha 32.
Corpus: frozen `v6-20261002-r4`, weather/economics requirement extraction.
The adapter uses the all-schema protocol named `v5`, with thinking disabled.
The runtime obtains its tokenizer and chat template from the pinned base.

Download from the repository root:

```powershell
git pull --ff-only origin experiment/v5-grounded-20260919
```

Follow [the local testing guide](../../V6_MANUAL_TEST_GUIDE.md#download-and-test-on-another-pc)
to install the pinned dependencies and launch interactive tests. The adapter is
67,332,688 bytes; the separate stock base weights are approximately 4.55 GB.
The chat command verifies both adapter file hashes before loading.
Git LFS is unnecessary for this adapter. Its LFS upload was rejected because the
repository's LFS budget was exhausted; the 67.3 MB file is stored directly in Git.

This is a final-step experimental candidate; behavioral checkpoint selection
and full v6 inference validation remain pending. The single loading smoke
returned valid JSON but omitted separate frequency, unit, horizon and file-format
updates. Low teacher-forced loss does not establish extraction quality.
