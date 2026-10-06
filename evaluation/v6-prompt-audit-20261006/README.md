# V6 natural-label audit and frozen prompt comparison

Agent review covers all 32 existing natural conversations and their 96 turns. No labels were changed. Independent human review remains pending. baseline-slot-audit.csv separates content omissions, wrong values, normalization-only differences and unexpected emissions; invalid outputs remain invalid. See baseline-audit.json for counts and limitations.

protocol.json freezes one original versus one revised prompt comparison: 384 new calls on the same v6 adapter, 32 conversations, gold-context component and own-state rollout tracks. This is a development experiment, not an independent test of generalization. The original study is preserved. No training, paid APIs or sealed labels.
