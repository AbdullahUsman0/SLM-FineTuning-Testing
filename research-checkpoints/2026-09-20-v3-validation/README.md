# V3 three-way validation evidence — 2026-09-20

These four JSON files are byte-for-byte copies of the local `results/` reports and derived summary. All 48 v3 validation scenarios, 90 component calls, scenario order, turn labels, and gold outputs matched across the three reports. `scripts/compare-validation-reports.py` reproduces the non-intent exact slot metrics and scenario-cluster paired bootstrap (10,000 draws; seed 42). The original 2B base and LoRA evaluations used different runtime/quantization setups; the OpenAI provider also uses different prompts and decoding. Interpret the table as a **system comparison on validation**, not a controlled model-only or final-test result.

| File | SHA-256 |
| --- | --- |
| `qwen35-2b-base-v3-validation.json` | `4e87d44f12d16a8ba8f6db08ff721d2609a2bb1063e4d1e6d8128861b1124bf3` |
| `qwen35-2b-lora-v3-validation.json` | `b682e45d796f71812ea8cd55f155a26ea0916f8cdcb0c89d1ba5637a1c090d30` |
| `openai-gpt5mini-v3-validation.json` | `5c2ea3d4093f7123632f91e4d1d4c271f895f7c3b214924c96d7a91ce21ae501` |
| `v3-validation-three-way-summary.json` | `79b53217d0606a5d509591593573014999e4572e92c5ff931e201ae32a5de026` |

OpenAI model ID: `gpt-5-mini-2025-08-07`, via the pinned `fpy` Responses client with `store=False`. All 48 scenarios completed with no provider errors. The report contains no API key. Existing production client code did not retain API token usage or billed cost. The original OpenAI historical `0.4068` and old 0.8B `0.0751` came from another cohort and are not merged into this comparison. See `RESEARCH_CHECKPOINT_2026-09-19.md` for the full context and paper caveats.
