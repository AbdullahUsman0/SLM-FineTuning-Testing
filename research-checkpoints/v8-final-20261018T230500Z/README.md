# Fixed v8 training and local development diagnostic

Completed the authorized fixed experiment. These are new inspected synthetic DEVELOPMENT results. No final-test labels or paid APIs were used; independent human annotation remains incomplete. There is no statistical significance test or established model winner.

| Model | Track | Exact F1 | Precision | Recall | JSON | Admitted | Extra flags | Median seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v8 | component | 0.9640 | 0.9651 | 0.9628 | 498/498 | 491/498 | 1 | 7.343 |
| v8 | rollout | 0.9640 | 0.9651 | 0.9628 | 498/498 | 491/498 | 1 | 7.353 |
| v6 | component | 0.4844 | 0.4987 | 0.4709 | 498/498 | 412/498 | 34 | 7.966 |
| v6 | rollout | 0.4867 | 0.5017 | 0.4725 | 498/498 | 401/498 | 35 | 7.727 |

Run: `D:\SLM\FYP-model-runs\qwen35-2b-lora-v8-final-20261018T230500Z`. GPU: NVIDIA RTX A4000. Completed 2,234/2,234 optimizer steps; 224 complete SHA-verified checkpoints retained. Seed 42; fresh rank-16 LoRA; two epochs; pinned stock base; effective batch 16; no truncation.

The isolated environment reuses the existing verified CUDA packages through a read-only site-packages path; only missing JSON Schema dependencies were installed in the new environment. Preserve both environments for resume. launch-config records the actual dedicated pinned fpy checkout; the historical trainer also logs a sibling fpy location as incidental repository metadata, which is not the dependency selected by this diagnostic.

comparison.json contains English/Roman Urdu/mixed and domain-family summaries, seeded-repair exclusion, and v8-minus-v6 differences with regression flags. granular-domain-breakdown.json adds all 20 individual domains. Per-arm failure-findings.json lists all structural, grounding/admission, extra/forbidden-slot, correction, unknown and accumulated-state failures. Extra flags include rejected/repeated emissions. Do not compare these scores directly with old cohorts or use training loss as extraction F1.

Adapter ZIP and checksum are local to the training PC Downloads, documented in adapter-package.json. They contain no base weights or optimizer state. They have not been uploaded as release assets. Raw development journals are compressed here with member SHA manifests. Final-step naming does not establish a best model.

Resume the SAME run with the supplied training/final_round/run.py start command after inspecting a failure; never edit global_step. The worker selects only a complete hash-verified checkpoint. This followup refuses duplicate supervision and preserves training failures for inspection.

## Publication recovery and adapter download

Both diagnostics completed on 23 October 2026 (v6 at approximately 00:03 PDT). The first automatic publication stopped at Git whitespace validation because the copied PowerShell test log retained trailing spaces. Only the publication text copy was normalized; PREPARATION_TESTS-original-utf16.txt.gz preserves every original log byte. PUBLICATION_RECOVERY.json records hashes and verification. No evaluation, labels, raw responses, training settings or checkpoints were changed or rerun.

The verified experimental final-step adapter is already available at https://github.com/AbdullahUsman0/SLM-FineTuning-Testing/releases/tag/v8-adapter-20261022 . The later local adapter ZIP mentioned above is a distinct package; see ADAPTER_RELEASE_NOTICE.md. The earlier v8-only snapshot remains intact and records its historical pending-v6 status; this parent comparison is now complete.

The matched development result favors v8 in point estimates, but this is inspected synthetic development, with incomplete independent human review and no final OpenAI comparison or significance test. Roman Urdu remains a weakness (v8 F1 0.7715). The v6 rollout has zero exactly complete states under this strict full-state criterion; this does not mean every v6 slot extraction failed.
