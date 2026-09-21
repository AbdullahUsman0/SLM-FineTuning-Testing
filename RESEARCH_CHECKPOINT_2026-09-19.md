# Research checkpoint — 2026-09-19 (updated 2026-09-20)

This is the durable, citable-internal checkpoint for the forecasting assistant SLM project. Read it with `HANDOFF_AUDIT_2026-09-17.md` and the raw artifacts before drafting a paper. It records observations and evidence limits; it is not a publication-ready performance claim. Repository state inspected: `AbdullahUsman0/SLM-FineTuning-Testing` at `7cc08f39272038f4c84ef4d41e64f436ce778742`; dependent `int-abd-5/fpy` at `04d52c015d1e3ecdefe92b87116f209361509b4b`. Local audit and scripts currently have uncommitted changes, so preserve their eventual commit hashes too.

## One-paragraph memory

The original Qwen3.5-0.8B experiments progressed through v1–v3 and an incomplete v4; the exact v4 paraphrase cache, generated SFT data, and old adapters were not recovered from Git or the shared Drive. On a new 32 GB Core Ultra 5 125U laptop without CUDA, stock Qwen3.5-2B runs locally; LoRA training used Colab Tesla T4. A fresh Qwen3.5-2B rank-16 LoRA trained two epochs on the *v3* extraction-only SFT set (865 train / 184 validation) and was verified by adapter SHA-256. On the same 48 v3 **validation** scenarios, exact non-intent slot F1 was 0.0714 for stock 2B, 0.0938 for its LoRA, and 0.3409 for the `gpt-5-mini-2025-08-07` OpenAI production client (9, 12, and 75 correct of 234 gold non-intent slots). OpenAI had better extraction recall but also more forbidden-slot inference and poorer question-contract accuracy. The LoRA had three invalid correction-turn extraction outputs and is not ready for deployment. The earlier OpenAI 0.4068 figure came from a different historical cohort; use this new same-case report instead. Next: build fact-grounded, multi-slot and correction-heavy v5 data; compare base and LoRA under the same runtime; repeat the three-way evaluation on newly frozen v5 validation and untouched final test sets.

## Provenance and run inventory

| Corpus / run | What is established | What is missing or uncertain |
| --- | --- | --- |
| v1 | 200 scenarios; 266 SFT train / 57 validation records | Weights and raw per-case output absent. |
| v2 | 320 scenarios; 420 / 90 SFT records | Weights and raw per-case output absent. |
| historical v3 0.8B | 320 scenarios; 865 / 184 extraction-only SFT; reported training eval loss 0.00138258 and token accuracy 0.999691 | Old adapter and raw held-out output absent. Reported slot F1 0.0751 is historical prose, not reproduced here. |
| v4 0.8B | Deterministic builder regenerates 420 scenarios; handoff claims 2,991 SFT records and an interrupted epoch-1 run | Exact paraphrase cache, exact generated SFT records, checkpoint, and behavioral result absent. Do not claim a completed fourth training round. |
| new v3 2B LoRA | Qwen/Qwen3.5-2B; LoRA rank 16 / alpha 32; 2 epochs; LR 5e-5; max length 3072 with no truncation; effective batch 16; Colab Tesla T4; 865 / 184 SFT records | Need a fair full-precision base run and untouched final test. |

The new 2B best adapter is 67,332,688 bytes, SHA-256 `538da1675b195ce1ca119e7a8cbc84bcf21d1b2f6928412d0bea8587fa2eaf13`. Drive run: `MyDrive/FYP-model-runs/qwen35-2b-lora-v3-repro-20260918/`. The downloaded copy and `run-manifest.json` are in ignored `training-runs/qwen35-2b-lora-v3-repro-20260918/`. Training reported eval loss `0.0014435567427426577` and mean token accuracy `0.9996431084430736`; these do not measure extraction quality. Verify the Drive link and privacy before sharing it publicly.

The v3 manifest line-ending issue is only a raw-byte mismatch: tracked Git LF records and Windows CRLF files have identical logical JSONL content. The historical training manifest hashes match LF Git blobs. The local verifier fix accepts either representation after checking their SHA-256 hashes. The Colab preflight converted verified files to the manifest's CRLF form before training; record both hashes and this transformation in any methods section.

## Paired v3 validation result (48 scenarios, 90 calls)

Exact slot score matches `(slot_id, candidate_value, status)` tuples. Both reports used the same scenario list and component scorer. Stock 2B was Q4_K_M llama.cpp on laptop CPU; LoRA was full-precision Transformers PEFT on Colab GPU. This runtime difference limits causal interpretation, especially for latency.

| Metric | Stock Qwen3.5-2B | v3-trained 2B LoRA |
| --- | ---: | ---: |
| Exact slot micro F1, including intent | 0.2412 | 0.3032 |
| Exact **non-intent** slot F1 | 0.0714 | 0.0938 |
| Correct non-intent slots / 234 gold | 9 | 12 |
| Intent accuracy | 0.9608 | 0.9020 |
| Provider success | 1.0000 | 0.9667 |
| Invalid extraction JSON calls | 0 | 3 |
| Empty-update collapse | 0 | 0.0667 |
| Forbidden slot inference | 0 | 0 |
| Question contract accuracy | 0.8462 | 0.9231 |

Paired scenario-cluster bootstrap, 10,000 resamples, seed 42: aggregate F1 difference LoRA minus stock `+0.0620`, 95% percentile interval `[+0.0194, +0.1129]`; non-intent F1 difference `+0.0223`, interval `[-0.0352, +0.1013]`. The adapter matched only 12 of 234 gold non-intent slots, all from three complete scenarios; it matched none across the other 45 validation scenarios. It missed all 27 `file_format` gold values and most descriptions and source fields. This is the practical failure mode to address. The training split has 672 of 865 selected short, sentence, or non-answer examples, often yielding zero or one update; that distribution is a plausible contributor, not proven causation.

Raw reports: ignored `results/qwen35-2b-base-v3-validation.json` SHA-256 `4e87d44f12d16a8ba8f6db08ff721d2609a2bb1063e4d1e6d8128861b1124bf3` and `results/qwen35-2b-lora-v3-validation.json` SHA-256 `b682e45d796f71812ea8cd55f155a26ea0916f8cdcb0c89d1ba5637a1c090d30`. Preserve copies alongside paper artifacts. The v3 test split has already been used for a 10-scenario exploratory comparison, so create a new untouched final test set.

## Same-case OpenAI comparison — measured 2026-09-20

The project `OpenAIProductionProvider`, backed by the pinned `fpy` Responses client, successfully ran the historical `gpt-5-mini-2025-08-07` snapshot on **all 48 identical v3 validation scenarios / 90 calls**, using the same component scorer and gold labels. The official model page lists this snapshot as deprecated, but it accepted this run. The API key was supplied in the process only and was not saved in the report or repository. The prior handoff's OpenAI F1 `0.4068` and historical 0.8B F1 `0.0751` concern a different cohort and remain separate historical claims.

| Metric | Stock 2B | v3 2B LoRA | OpenAI GPT-5 mini snapshot |
| --- | ---: | ---: | ---: |
| Exact slot micro F1, including intent | 0.2412 | 0.3032 | **0.3875** |
| Exact **non-intent** slot F1 | 0.0714 | 0.0938 | **0.3409** |
| Correct non-intent slots / 234 gold | 9 | 12 | **75** |
| Non-intent precision / recall | 0.5000 / 0.0385 | 0.5455 / 0.0513 | 0.3641 / **0.3205** |
| Intent accuracy | **0.9608** | 0.9020 | 0.9020 |
| Provider success | 1.0000 | 0.9667 | 1.0000 |
| Forbidden-slot inference rate | **0** | **0** | 0.1795 |
| Question contract accuracy | 0.8462 | **0.9231** | 0.6410 |
| Invalid/provider-error calls | **0** | 3 | **0** |

The OpenAI non-intent F1 minus LoRA difference is `+0.2472`, with scenario-cluster paired bootstrap 95% percentile interval `[+0.1686, +0.3383]`; OpenAI minus stock is `+0.2695`, interval `[+0.1925, +0.3461]` (10,000 resamples, seed 42). These intervals describe uncertainty across the 48 validation scenarios, not generalization to a new cohort. OpenAI matched many more actual fields but also produced many more incorrect fields (131 non-intent false positives versus 10 for LoRA) and inferred forbidden slots in 17.95% of applicable calls. Its question contract passed 25/39 calls, versus 36/39 for LoRA. No API token-usage/cost data were retained by the existing production client. API latency and local/Colab latency are not hardware-matched. The OpenAI and SLM provider prompts and decoding also differ, so this is a **system-level comparison**, not an isolated model-only effect.

An in-progress v5 corpus audit (`v5-plan-and-inventory.md`, not yet independently reviewed here) also flags legacy v3 prompt/label omissions, cross-split input overlap, and ambiguous gold/forbidden-label conflicts. Those potential benchmark defects are another reason to treat all v3 scores as exploratory and to use the reviewed, newly frozen v5 set for paper claims.

The full OpenAI report is `research-checkpoints/2026-09-20-v3-validation/openai-gpt5mini-v3-validation.json`, SHA-256 `5c2ea3d4093f7123632f91e4d1d4c271f895f7c3b214924c96d7a91ce21ae501`. This trackable folder also contains byte-identical stock and LoRA reports plus the derived three-way summary; see its `README.md` for all hashes. The comparison script verified identical scenario order, call types, turns, and gold labels across all three reports. The OpenAI report was saved after each scenario by `scripts/evaluate-openai-resumable.py`; `scripts/compare-validation-reports.py` reproduces the non-intent metrics and paired intervals. Commit and push the evidence folder with the research checkpoint before depending on it across machines.

## Deployment and paper guardrails

- The local F16 LoRA GGUF (33,664,640 bytes, SHA-256 `e2b63e04851446ea3a733041d39deb7d0753244f548f3b97ea656c689f5ef7f5`) loads with the Q4_K_M base, but a four-scenario CPU smoke test is not a quality estimate. It had two invalid question JSON calls and materially different output from full-precision PEFT. Do not treat conversion fidelity as established.
- The laptop has integrated Intel Graphics and an Intel AI Boost NPU; no CUDA GPU. The current trainer requires CUDA. CPU is the reproducible local evaluation default; Intel Vulkan is optional and changed small-sample outputs. No validated NPU path has been established for this Qwen3.5-2B setup.
- Do not cite validation scores as final generalization, report token accuracy as task accuracy, or rank 2B against the historical OpenAI number. Disclose test-set exposure, backend/quantization confounds, missing artifacts, and CIs. Freeze a new scenario-disjoint final test before new training; evaluate it once after selection.
- As last inspected, `corpus-v5/v5-20260919-r1/build-status.json` says `complete_pending_human`: a candidate synthetic corpus exists, but required human label/overlap review is pending. `training/v5-colab-inventory.json` records a T4 and a failed Drive credential-propagation mount, with zero v5 training seconds. Do not cite v5 as a completed experiment until those gates and an actual run finish.

## Next reproducible milestone

Execute `training/ALIBABA_AGENT_V5_DATA_TRAIN_EVAL_PROMPT.md`. Deliver dataset provenance and leakage audit, versioned v5 train/validation/test hashes, same-runtime full-precision base and LoRA reports, a new same-case OpenAI report on v5, per-case error analysis, and a held-out final comparison with confidence intervals. Record repository commit, `fpy` commit, package versions, model revision, hardware, prompts, generation settings, seeds, exact adapters/checkpoints, SHA-256s, and any deviations in a run manifest. Save bulky artifacts to Drive and durable summary/manifests to the repository. Do not promote a model until non-intent extraction and correction behavior improve materially without JSON or safety regressions.
