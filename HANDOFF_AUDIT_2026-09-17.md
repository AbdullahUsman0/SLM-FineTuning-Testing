# Handoff audit — 2026-09-17

## Checkout and local setup

- Cloned `main` at `7cc08f39272038f4c84ef4d41e64f436ce778742` (2026-09-14); this matched `origin/main` when checked.
- Cloned the required `int-abd-5/fpy` sibling repository and checked out pinned commit `04d52c015d1e3ecdefe92b87116f209361509b4b`.
- The new laptop is a Lenovo 21LVS44F00 with Intel Core Ultra 5 125U (12 cores, 14 threads), 32 GB RAM, integrated Intel Graphics, and an Intel AI Boost NPU. No CUDA GPU is present. The current `training/train_lora.py` explicitly requires CUDA, so training must use Colab or another suitable GPU environment unless the trainer is redesigned.
- The bundled Python 3.12 runtime ran all 54 offline tests successfully after `tests/test_corpus_v4.py` was changed to mock the live paraphrase call. The v2 and v3 corpus verifiers pass.

## What Git actually contains

| Round | Versioned corpus | Versioned result evidence | Missing from Git |
| --- | --- | --- | --- |
| v1 | 200 scenarios; 266 train and 57 validation SFT examples | Historical descriptions in handoff/report | Adapter, checkpoints, raw held-out output |
| v2 | 320 scenarios; 420 train and 90 validation SFT examples | Historical descriptions in handoff/report | Adapter, checkpoints, raw held-out output |
| v3 | 320 scenarios; 865 train and 184 validation SFT examples | `results/evaluation.json`, `results/run-manifest.json`, `results/v3-run-summary.json` for the training run; reported behavioral metrics in the progress report | Adapter and checkpoints; raw held-out component and conversation reports |
| v4 | Builder, tests, notebook, and Colab instructions | Handoff says 420 scenarios / 2,991 SFT examples and reports an interrupted epoch 1 | Generated SFT corpus, paraphrase cache, checkpoints/adapter, training metrics, raw held-out reports |

The repository ignores `.cache/`, `corpus-v4/`, `models/`, `runtime/`, `training-runs/`, and most generated files in `results/`. Thus a fresh clone does not contain every record or model artifact mentioned by the handoff. A Drive copy may still hold v4 artifacts, but that cannot be established from this repository.

## Reproducibility checks

1. Rebuilt only the deterministic v4 scenarios and split files locally under ignored `corpus-v4/`: 420 scenarios (294 train, 78 validation, 48 test). This is **not** the complete v4 SFT corpus. Rebuilding the reported 2,991 SFT examples requires the original `.cache/paraphrases_v4.json`; fresh API output may differ. A test build with synthetic, unique paraphrases yielded 3,593 records, showing that the reported count depends on the missing paraphrase output and deduplication.
2. The v3 corpus has a line-ending portability bug. The Windows checkout's CRLF files match `corpus-v3/manifest.json`; the LF Git blobs (and Linux checkout) match the recorded historical training inputs in `results/run-manifest.json`:

   | File | Windows CRLF / corpus manifest SHA-256 | Git LF / recorded training SHA-256 |
   | --- | --- | --- |
   | v3 train | `f43e88f8c24db0701d3e62df2f7e1a04a8cda93e736809fe24ddecaced833674` | `ccb4b172e382f54388bfc86eead0a2eb163c7c31218088f73faf573279c0b5ee` |
   | v3 validation | `93c39338ad1f85a003d69a3728de5ecc2f79d29c715085487a2eed6ede3f9bad` | `69a72c386325ff1a4b3769ff90ddbe5649adb6c0bde966b64a5e0bd5c30e0fa5` |

   The records are identical after line-ending normalization. The pinned upstream `scripts/verify-corpus-v3.py` checks raw bytes against CRLF hashes and therefore fails on a clean Linux/Colab checkout. A Colab preflight verified the pinned Git blobs, converted them to manifest-matching CRLF with a backup, then ran the verifier. This local checkout now has an uncommitted verifier fix that accepts either line-ending form while rejecting content changes; it was checked against all six corpus files in both forms. The tracked `eval_loss=0.00138258` and `eval_mean_token_accuracy=0.999691` remain historical training summary claims; the 0.8B adapter is still absent.
3. The reported v3 component slot F1 of `0.0751` and OpenAI `0.4068` appear only in prose; the underlying per-case held-out reports are absent. The current `slot_micro_f1` checks exact `(slot_id, candidate_value, status)` tuples. The September 14 token-F1 addition measures **question text**, not slot values. Fresh v3/v4 comparisons must run both candidates on the same held-out cases with the same evaluator revision.
4. v4 has no completed training or held-out result in Git. Its quality relative to v3 is unknown. The handoff says an epoch-1 checkpoint was kept in an ephemeral Colab path and may have been lost when that session ended.
5. The v4 builder sets every paraphrased example's evidence to the entire paraphrase but retains the original gold slot values. Its structural validator checks evidence containment, not whether a paraphrase preserved all labeled facts. Review paraphrases for changed facts before using them for final training.

## Recommended next experiment

1. Preserve or recover the v3 adapter, v4 paraphrase cache and generated manifest, and any v4 checkpoint/adapter from Drive or the old machine. The exact v3 training input bytes are the pinned LF Git blobs. Record SHA-256 hashes and the model revision. Avoid publishing private prompts or credentials.
2. Finish and evaluate the 0.8B v4 experiment first, if its exact data and checkpoint can be recovered. Otherwise regenerate it with a new run identifier and record the new data hashes. Use a held-out component report, including slot F1, nonempty-update rate, forbidden inference, validity, latency, and conversation progression.
3. Test the **base** `Qwen/Qwen3.5-2B` locally in a verified GGUF quantization (start with Q4_K_M) on CPU. The 32 GB RAM makes this practical; measure real latency and memory rather than assuming the larger model is faster. Try Intel GPU offload via llama.cpp SYCL only after a stable CPU baseline. The NPU needs a separate compatible OpenVINO/WindowsML path and should not be assumed active in llama.cpp.
4. If 2B base extraction is promising, fine-tune it on a frozen, reviewed extraction-only corpus using a cloud CUDA GPU. Keep 0.8B v3/v4 and 2B base/fine-tuned results on the same untouched test cases. Promote based on task quality and end-to-end latency, not validation loss alone.

The 2B model is a **candidate**, not a measured improvement. More parameters may improve extraction quality but will increase compute and usually latency on this 15 W class laptop. A 3B/4B candidate can be considered after the 2B benchmark establishes the quality-speed tradeoff.

## 2026-09-18 local paired baseline

The stock, unfine-tuned 0.8B and 2B models were run through the same `local-sft` provider and current evaluator on held-out v3 scenario indices `0,3,6,9,12,15,18,21,24,27`. These cover all ten scenario categories but only the `production_output` test domain. Both used `llama.cpp` b11026, CPU only, 8 threads, 4096 context, `--reasoning off`, temperature 0, and a 1024-token output limit. The complete per-case reports are local, ignored files in `results/`.

| Measure | 0.8B Q4_0 | 2B Q4_K_M |
| --- | ---: | ---: |
| Exact slot micro F1 | 0.0243 | 0.1759 |
| Slot recall | 0.0123 | 0.0988 |
| Slot precision | 1.0000 | 0.8000 |
| Empty-update collapse | 0.9000 | 0.0000 |
| Intent accuracy | 1.0000 | 1.0000 |
| Forbidden slot inference | 0.0000 | 0.0000 |
| Median call latency | 2.42 s | 7.53 s |
| Observed server working set | ~1.39 GB | ~2.4 GB |

The 0.8B run had one 245-second generation outlier, so its mean latency is not representative. The 2B model improved extraction recall on this small paired sample, while taking about three times longer at the median. Provider-level structured validity includes guardrail/recovery behavior and does not prove that every raw model response was valid. This sample does not establish a production winner or the benefit of fine-tuning. Use `scripts/start-baseline-server.ps1` to restart either local base model.

Because these ten v3 test scenarios informed model selection, treat them as exploratory cases. Use validation data for further prompt/model choices and reserve a new untouched final test set for the eventual comparison.

Model provenance: `ggml-org/Qwen3.5-0.8B-GGUF` file `Qwen3.5-0.8B-Q4_0.gguf`, SHA-256 `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`; `unsloth/Qwen3.5-2B-GGUF` file `Qwen3.5-2B-Q4_K_M.gguf`, SHA-256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`. Both local files matched the publisher's LFS SHA-256 metadata. The Windows CPU runtime initially crashed in `MSVCP140.dll`; installing Microsoft's signed Visual C++ redistributable 14.51 resolved that startup failure.

## 2026-09-18 Intel acceleration check

No v3/v4 adapter or v4 paraphrase cache was found in this machine's Desktop, Documents, or Downloads folders. Recovery from the old Drive folder is still needed.

With the same 2B Q4_K_M file and llama.cpp b11026, `llama-bench` (512 prompt tokens, 128 generated tokens, three repetitions, eight threads) measured:

| Backend | Prompt tokens/s | Generation tokens/s |
| --- | ---: | ---: |
| CPU | 68.97 ± 10.60 | 18.89 ± 0.29 |
| Intel Graphics via Vulkan | 300.93 ± 2.24 | 18.68 ± 0.05 |
| Intel Graphics via SYCL | 421.19 ± 17.14 | Did not finish in a practical time |

The same ten-case component check on Vulkan had median call latency 5.86 s versus 7.53 s on CPU, with slot F1 0.1538 versus 0.1759. These small-sample outputs differ across backends, so keep CPU as the reproducible evaluation default; Vulkan is an optional runtime speed experiment. The launcher accepts `-Backend cpu` or `-Backend vulkan`.

The [llama.cpp OpenVINO validation table](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md) lists Qwen3.5-2B inference on NPU as failed, including on a newer Intel Core Ultra platform. This machine's NPU is present, but it is not a validated acceleration path for this model in the current runtime.

## Codex handoff folder search

The handoff the user identified is the tracked `CODEX_HANDOFF_2026-08-25.md` in this repository. It describes earlier runs and explicitly says model checkpoints must be transferred separately. The newer `CONTINUE-HERE.md` points to Google Drive for v4, but says its epoch-1 checkpoint lived in an ephemeral Colab runtime. A recursive filename/folder search of this machine's user profiles found no `paraphrases_v4.json`, `adapter_model.safetensors`, `best-adapter`, `FYP-model-runs`, or v4 checkpoint. `Documents/Codex` contains only task work/outputs directories and no files. The original handoff references the previous machine's `C:\Users\Mahad Enterprises\OneDrive\Desktop\FYP\local-slm-lab`, which is not a path on this machine. The handoff provides evidence of what was done; it does not recover the weights or exact v4 training examples.

`training/COLAB_2B_V3_REPRO.md` provides a fresh 2B run using tracked, hash-verified v3 SFT as an available fallback. This is distinct from the historical v3 run and should be superseded by a reviewed exact v4 corpus if that cache is later recovered. The run status is updated below.

## 2026-09-19 user-reported 2B training result

The user ran the fresh Qwen3.5-2B LoRA notebook in Colab against v3 SFT after resolving the CRLF manifest mismatch. The run reached epoch 2.0 and saved a 67,332,688-byte `best-adapter/adapter_model.safetensors` with SHA-256 `538da1675b195ce1ca119e7a8cbc84bcf21d1b2f6928412d0bea8587fa2eaf13` under `MyDrive/FYP-model-runs/qwen35-2b-lora-v3-repro-20260918/`. The training validation summary is `eval_loss=0.0014435567427426577` and `eval_mean_token_accuracy=0.9996431084430736`. The shared Drive folder is now accessible, and the downloaded adapter's size and SHA-256 match the user's console output. The downloaded `run-manifest.json` records a **Tesla T4**, Python 3.13.15, 2 epochs, 865/184 CRLF v3 SFT records, no truncation, and an effective batch size of 16. The local adapter and manifest are under ignored `training-runs/qwen35-2b-lora-v3-repro-20260918/`. `eval_num_tokens=0.0` is reported by the trainer and should not be interpreted as zero evaluated tokens.

The full `scripts/evaluate-peft.py` component evaluation on the 48 v3 validation scenarios wrote `validation-component-report.json` to the same Drive folder. The loss/token accuracy alone are insufficient to judge extraction quality or promote this adapter; the behavioral result is analyzed below.

## 2026-09-19 2B validation component report

The user supplied the completed 48-scenario report, preserved locally as ignored `results/qwen35-2b-lora-v3-validation.json` (209,417 bytes; SHA-256 `b682e45d796f71812ea8cd55f155a26ea0916f8cdcb0c89d1ba5637a1c090d30`). It contains 90 calls: 51 extraction and 39 question calls. Overall exact `(slot_id, value, status)` micro precision/recall/F1 are **0.7761 / 0.1884 / 0.3032**. Intent accuracy is 0.9020, empty-update collapse 0.0667, forbidden inference 0, question contract accuracy 0.9231, median call latency 8.53 seconds on the Colab GPU, and three extraction calls failed with invalid JSON (all second turns of correction scenarios).

The aggregate slot score is dominated by intent. Of 276 gold updates, 42 are intent; the model exactly matches 40 intent updates. Excluding intent, it exactly matches only **12 of 234** gold updates: precision 0.5455, recall 0.0513, F1 0.0938. It misses all 27 `file_format` values, 38 of 39 `target_description` values, 37 of 39 `problem_statement` values, and 26 of 27 `source_mode` and `source_reference` values. Most extraction calls return just one update even when the gold label contains 3–10. The 865-record training split has 193 reviewed-base examples, 224 selected-short, 224 selected-sentence, and 224 selected-non-answer examples; the latter 672 strongly emphasize zero or one update. This is a plausible contributor to the one-update behavior, not a proven causal attribution.

All 12 correct non-intent updates occur in the three `complete` scenarios. The model gets **zero** non-intent updates correct across the other 45 validation scenarios, including missing-required, ambiguity, corrections, conflicts, privacy, and conversational requests. Two of three unsupported-intent boundary cases are classified as `create_forecast`. These failures matter more for deployment than the high training token accuracy.

Do not promote this adapter for the forecasting extractor yet. The report uses **validation** scenarios and must not be compared directly with historical 0.8B **test** claims or the earlier ten-case 2B base test. An untouched final test set remains necessary after model/data choices are frozen.

### Paired stock-2B comparison on the same validation scenarios

The stock 2B Q4_K_M GGUF was run locally on all 48 identical v3 validation scenarios using the same component scorer (`results/qwen35-2b-base-v3-validation.json`; 90 calls). This is a model/runtime comparison, with quantized CPU llama.cpp for stock and full-precision GPU Transformers PEFT for the adapter, so latency is **not** directly comparable and quality differences may include quantization/backend effects.

Local report SHA-256: stock `4e87d44f12d16a8ba8f6db08ff721d2609a2bb1063e4d1e6d8128861b1124bf3`; LoRA `b682e45d796f71812ea8cd55f155a26ea0916f8cdcb0c89d1ba5637a1c090d30`.

| Metric | Stock 2B | 2B LoRA |
| --- | ---: | ---: |
| Exact slot micro F1, including intent | 0.2412 | 0.3032 |
| Exact non-intent slot F1 | 0.0714 | 0.0938 |
| Correct non-intent slots / 234 gold | 9 | 12 |
| Intent accuracy | 0.9608 | 0.9020 |
| Provider success | 1.0000 | 0.9667 |
| Invalid extraction JSON calls | 0 | 3 |
| Empty-update collapse | 0.0000 | 0.0667 |
| Question contract | 0.8462 | 0.9231 |

The adapter improves the aggregate slot F1, but its non-intent gain is just three additional correct values and it loses the two stock-model correct correction values. A scenario-cluster paired bootstrap (10,000 resamples, seed 42) estimates the aggregate F1 difference at +0.0620 with a 95% percentile interval of +0.0194 to +0.1129; for **non-intent** F1 the difference is +0.0223 with an interval of -0.0352 to +0.1013. This is exploratory validation, not an untouched final test or a deployment claim. The main next experiment should rebalance the corpus toward grounded multi-slot extraction and correction turns, then repeat the same validation checks before a fresh final test.

### Local adapter conversion and smoke check

The public Drive `best-adapter` folder was downloaded locally. Its 67,332,688-byte safetensors file matches SHA-256 `538da1675b195ce1ca119e7a8cbc84bcf21d1b2f6928412d0bea8587fa2eaf13`; `adapter_config.json` names `Qwen/Qwen3.5-2B`, rank 16, alpha 32. The pinned llama.cpp b11026 `convert_lora_to_gguf.py` successfully converted all 372 adapter tensors to an ignored F16 LoRA GGUF using Transformers 5.15.0 (the converter's older requirements pin of 4.57.6 does not recognize `qwen3_5`). The GGUF adapter loads with the local Q4_K_M base in llama-server; `scripts/start-2b-lora-server.ps1` launches this experimental configuration on port 8085.

The converted `adapter-F16-LoRA.gguf` is 33,664,640 bytes, SHA-256 `e2b63e04851446ea3a733041d39deb7d0753244f548f3b97ea656c689f5ef7f5`.

A four-scenario smoke check (`results/qwen35-2b-lora-local-v3-smoke4.json`) completed on the laptop CPU. It verifies execution but is too small for a quality comparison: exact slot F1 is 0.5854 on these selected cases, intent accuracy 0.8, two question calls returned invalid JSON, median call latency 20.15 seconds and p95 78.86 seconds. The CPU LoRA/GGUF output differs materially from the full-precision GPU PEFT output on the same cases. Quantization, runtime behavior, or conversion fidelity may contribute; equivalence has not been established. Do not treat the four-case score as evidence of a better model. The local adapter is technically runnable but not ready for the production extractor.
