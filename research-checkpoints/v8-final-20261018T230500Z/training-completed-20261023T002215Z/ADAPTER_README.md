# Experimental final-step v8 adapter

This is the fresh stock-base LoRA trained on corpus-v8/v8-final-20261018-r1 for two epochs, seed 42, 2,234 optimizer steps. Base: Qwen/Qwen3.5-2B, revision 15852e8c16360a2fea060d615a32b45270f8a8fc. Load this adapter onto that stock base; never stack it onto another adapter. The architecture is Qwen3_5ForConditionalGeneration. adapter_config.json has no pinned revision value, so the base revision must be supplied explicitly by the loader.

The folder name best-adapter refers to the fixed FINAL STEP. Behavioral selection, development comparison and independent human review are pending at this packaging snapshot. This is not a final-test or OpenAI comparison, and no model winner has been established. The final teacher-forced development-validation loss is 0.0033881000708788633; it is not extraction F1.

See inference-dependencies.json and training-run-manifest.json for exact versions. The validated stack used Python 3.12.10, CUDA torch 2.11.0+cu128, transformers 5.15.0, peft 0.20.0, accelerate 1.14.0, safetensors 0.8.0 and tokenizers 0.22.2. The extraction pipeline also uses pydantic 2.13.5 and jsonschema 4.26.0. Preserve a working CUDA installation; do not replace it with a CPU wheel. The base weights are downloaded/cached separately and are excluded here.

## Reproduce the frozen local development diagnostic

From the repository branch experiment/v5-grounded-20260919, use the dedicated clean fpy dependency at commit 04d52c015d1e3ecdefe92b87116f209361509b4b. Substitute local paths and an existing compatible CUDA Python. Use a NEW output directory; do not touch the running remote evaluation:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
& <existing-CUDA-python> -B training/final_round/evaluate_development.py `
  --fpy runtime/fpy-final-pinned `
  --adapter <extracted-bundle>/best-adapter `
  --output <new-local-development-output>
```

The script pins the base revision, uses the v8 corpus instructions/response schema, disables thinking and sampling, and requires BF16 CUDA. It selects the same 100 development conversations for component and rollout tracks. It does not call a paid API or read final-test labels. A CPU/quantized manual run is a separate testing configuration and cannot replace these CUDA results. The older chat-peft.py v5/v6 prompts are not automatically the frozen v8 evaluation prompt.

The training receipt records all 224 complete resumable checkpoints. This transfer archive contains adapter files and metadata only, not optimizer/RNG checkpoints. It excludes base weights, datasets, caches and credentials. Keep the ZIP checksum and ARCHIVE_MANIFEST.json; verify member hashes before loading.
