# English-only v8/OpenAI prospective comparison preparation

Preparation capsule; **not results or certified gold**. Read LOCAL_CODEX_HANDOFF.md.

- 44 primary English conversations / 205 turns; 14 weather, 30 economics.
- 3 separate joint-target challenges / 18 turns.
- AI-drafted annotations for 205 primary turns, awaiting semantic checking and genuine gold review.
- Input/protocol snapshot and evidence screen; executable runner/model/prompt/gold freeze still required.
- No new inference, paid API authorization, training or sealed-label access.

Verify `PUBLICATION_MANIFEST.json`, the ZIP SHA-256 and internal INPUT_PROTOCOL_SNAPSHOT.json. Extract only into a new folder:

```text
python research-checkpoints/english-v8-openai-prospective-20261030/verify_extract.py --output runtime/english-followup-r1
```

The publication does not contain weights, secret keys or the main PC's revised paper. Existing paper drafts remain in place. The future main comparison is 410 local GPU calls and 410 hosted API calls after readiness/budget gates, not a six-strategy factorial study.
