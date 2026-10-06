# Focused v6 output-schema prompt test

One candidate appends only a final output-contract reminder to the previous revised prompt. Its completeness, evidence, correction, unknown-information and illustrative content remains byte-identical as a prefix. The unchanged 32 development conversations and labels are frozen in natural.jsonl.

384 new calls compare revised and schema_fixed in one session on gold-context component and own-state rollout tracks. The original prompt is retained as a historical reference; its latency is not treated as a same-session measurement. Revised-repeat equivalence will be checked. See protocol.json for hashes, success criteria and limits.

No training, parsing repair, retries, paid APIs or sealed final labels. Default chat behavior is not changed by this test.
