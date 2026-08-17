# Corpus v2

Corpus v2 keeps the original 200 forecasting scenarios and adds 120
conversational-intent scenarios. It contains 320 scenarios split by domain:

- 224 train
- 48 validation
- 48 held-out test

The generated SFT set contains 510 examples: 289 extraction examples and 221
question examples. Test scenarios never enter SFT data.

New cases include short confirmations (`yes`, `yeah`, `haan`), short forecast
requests, Roman Urdu requests, explicit negative intent, prior-question context,
and state-progression expectations. All records use the compact SLM prompt
profile from `local_slm_lab/slm_prompts.py`.

The exact Qwen tokenizer reports zero records above the 3,072-token training
limit. The maximum extractor example is 2,955 tokens.

Regenerate deterministically with:

```powershell
python scripts\build-corpus-v2.py
```

