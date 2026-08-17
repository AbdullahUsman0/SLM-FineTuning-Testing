# Forecasting LLMREI 200 Corpus

This corpus is synthetic, deterministic, schema-grounded, and designed for the
forecasting requirements-elicitation experiment. It is not copied from user data.

The 200 scenarios contain ten categories with twenty domain variants each. Splits
are domain-disjoint: 140 train, 30 validation, and 30 test. Test scenarios are
never converted to supervised fine-tuning examples.

The SFT files contain two production-compatible tasks:

- `extract`: return the structured `ExtractorResult` using only current-message evidence.
- `ask`: return one focused `QuestionOutput` for the selected schema slot.

Do not tune prompts, train, or select checkpoints using `splits/test.jsonl`.
Use validation data for checkpoint selection and run the test split only for
final comparisons.

Rebuild with:

```powershell
python .\scripts\build-corpus.py
```

