# Corpus v3: focused structured extraction

Corpus v3 keeps the 320 reviewed forecasting scenarios and leakage-safe domain
splits from v2, but changes the training contract:

- 1,049 unique extraction-only prompt/completion examples
- 865 training examples and 184 validation examples
- 48 held-out scenarios never included in SFT
- 272 short answers covering every required slot
- 272 natural-sentence answers covering every required slot
- 272 non-answers that must not be invented or forced into a slot
- 233 unique reviewed base extraction turns

Question wording is deliberately excluded from SFT. Runtime questions come
from the versioned forecasting schema, so the small model has one job: return a
grounded structured extraction from the current user message and dialogue state.

Build and verify:

```bash
python scripts/build-corpus-v3.py
python scripts/verify-corpus-v3.py
python -m unittest tests.test_corpus_v3 tests.test_training_helpers -v
```

The training script uses explicit conversational `prompt` and `completion`
fields and sets `completion_only_loss=True`. This prevents system/user prompt
tokens from dominating the reported loss and token-accuracy measurements.
