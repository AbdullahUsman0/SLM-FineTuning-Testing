# Paste this prompt into Codex on the remote GPU PC

Run a **fresh evaluation**, using the supplied self-contained
`policy-gpu-handoff-20261007.zip`. The model is already fine-tuned. Preserve all
existing adapters, training runs, evaluation reports and final-test labels.

1. Extract the ZIP into a new `D:\SLM\policy-comparison-20261007` directory.
   This bundle contains the reviewed development cases, frozen parser, exact fpy
   snapshot and evaluation code. It can run without pulling uncommitted changes
   from GitHub. Verify every `handoff-manifest.json` file checksum before running.
2. Use the existing CUDA environment at
   `D:\SLM\SLM-FineTuning-Testing\.venv\Scripts\python.exe` (inspect and adjust
   that path if necessary). Verify torch.cuda.is_available(), GPU name and BF16
   support. Keep the installed CUDA PyTorch/Transformers/PEFT stack. Add only the
   missing or mismatched parser/scorer dependencies:

   ```powershell
   & 'D:\SLM\SLM-FineTuning-Testing\.venv\Scripts\python.exe' -m pip install `
     pydantic==2.13.5 jsonschema==4.26.0 numpy==2.3.5
   ```

3. Set the working directory to the extracted bundle and HF_HOME to
   `D:\SLM\hf-cache`. The v6 adapter is expected at
   `D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter`.
   The runner checks both adapter hashes, pinned base revision/weight checksum,
   original fpy bytes, policy hash and reviewed contexts. Use CUDA BF16; do not
   silently substitute CPU, quantized weights, another adapter or a different pin.

4. Verify/preflight, then execute the v6 run:

   ```powershell
   Set-Location 'D:\SLM\policy-comparison-20261007'
   .\scripts\run-policy-gpu.ps1 -PrepareOnly
   .\scripts\run-policy-gpu.ps1
   ```

   This performs 31 scenarios / 93 extraction calls in each of two tracks:
   **component** receives reviewed gold prior state; **rollout** receives its own
   predicted prior state with the fixed user script. Both tracks use the same
   shared parsing policy as the fresh OpenAI run. There is one unscored warmup.

5. The runner atomically saves each completed three-turn scenario. Re-run the
   identical command and directory to resume. Completed scenarios are skipped;
   an interrupted GPU scenario may repeat its three calls. No model training or
   training checkpoints are involved. If a fingerprint check fails, diagnose it;
   do not edit the frozen manifest or reset the output directory to hide it.

6. If GPU time permits, also run the stock control and v5 comparator using unique
   run directories (v5 adapter is independently verified by the runner):

   ```powershell
   .\scripts\run-policy-gpu.ps1 -Arm base -RunDir training-runs/policy-stock-bf16-20261007
   .\scripts\run-policy-gpu.ps1 -Arm v5 `
     -Adapter 'D:\SLM\FYP-model-runs\qwen35-2b-lora-v5-anydesk-20261001T141017Z\full\best-adapter' `
     -RunDir training-runs/policy-v5-bf16-20261007
   ```

7. Return a ZIP containing the v6 run directory (and optional stock/v5 run
   directories): `completion.json`, `run-manifest.json`, `runtime.json`,
   `component-report.json`, `rollout-report.json`, and `jobs/`. Include a checksum
   file. Tell me the actual status and any failures. No weights or API key are
   required in this results ZIP. Do not execute paid OpenAI calls on this PC;
   Codex on my main PC is running that arm.

This is agent-reviewed **posthoc development** material, not an independent final
test. One ambiguous weather scenario is excluded. Never import the old response
cache or compare old labels to new labels. An unchanged source parser/prompt does
not mean an unchanged experiment: 62 later gold contexts changed after review.

After results return, the main PC will use `compare-policy-comparison.py` to check
policy/gold/context consistency and compute scenario-paired 10,000-resample
bootstrap intervals. Keep conclusions limited to this development cohort.
