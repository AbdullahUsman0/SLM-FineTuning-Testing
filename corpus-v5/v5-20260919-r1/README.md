# v5 synthetic forecasting corpus

All source facts are fictional simulations, including health operations; no real patient,
person, institution or external API data is used. v1-v3 are excluded pending source,
label and leakage human audit; untrusted v4 is excluded. No old biased examples are mixed in.

## Version and schema contract

See schema.json (all 79 definitions), dependency.json (pinned fpy source hashes),
prompt-manifest.json and split-freeze.json. The latter is written before generation
or training. Candidate values are JSON-encoded text with strict canonical types;
free text keeps exact spelling, punctuation and case. Durations use periods/unit.
Date labels are ISO strings; the reducer normalizes them to datetime internally.
Evidence offsets are Python Unicode character offsets, half-open [start,end), not bytes.
All JSONL files are UTF-8 with LF bytes. Component records use corpus.py/component_eval.py
scenario/turn/gold_extraction shape; source_facts, context and evidence_spans add provenance.
SFT uses messages (system/user/assistant), v5 all-schema extractor prompts and existing
compact question prompts. Each source has three fact-preserving main-turn renderings
and one question; correction/conflict sources additionally contain a prior turn.
Variants are alternative renderings, NOT consecutive user utterances. Their contexts
are explicit fixtures. Prior confirmation is a separately declared user event.
component_eval reconstructs context values but does not restore established state.intent;
v5 SFT restores it. Selected short answers rely on the actual reducer/selector context.
Questions follow the existing question_request contract, which treats supplied context as
confirmed; their provenance is synthetic and still needs human review.

## Splits and procedural seal

All source turns/variants/questions stay in the same frozen split. The entire synthetic
health-operations domain and the reported-speech handover template family are final only.
Other domains cross splits at source level. Shared clauses/fact assembly remain across
splits: this is not a zero-leakage or linguistically independent benchmark. See the
measured representative fuzzy overlap report and pending pair queue.

splits/train.jsonl and splits/validation.jsonl are development component cases.
splits/smoke.jsonl is a deterministic SUBSET of validation, never an independent split.
sft/train.jsonl and sft/validation.jsonl are the only public training-format examples.
Final component cases, labels, facts and SFT are ONLY in the separately ignored sealed
root. Public manifests contain aggregate final statistics and hashes, not final labels.
This is procedural sealing, not encryption, cryptographic access control, independent
human testing, or a license to tune against final results. Do not open final labels for
training/debugging; verification hashes their bytes without parsing them.

## Review gate

Every case is machine_validated_pending_human; none is accepted_human. Read
review-decisions.jsonl for every machine acceptance reason and human-review requirement.
review-queue.jsonl lists ALL final IDs, uncertain/context-dependent cases and at least
one train case per domain/category stratum. Review every variant and prior context,
not only the first utterance: check naturalness, canonical semantics, polarity, exact
quoted wording, evidence offsets, non-invention, corrections, and question relevance.
All flagged cross-split pairs remain pending; reviewer must record accept/reject with
identity, time and substantive reasons in a separately versioned review ledger. A
final reviewer must not communicate final labels or case-level feedback to the trainer.
Do not claim the review gate passed until all required cases and pairs are resolved.
Rejected or revised frozen examples require a NEW corpus revision, never overwrite.

## Measured limitations

Category ratios refer to extraction examples, not question examples. Statistics report
actual update histograms, per-slot positives and absence opportunities, lengths and
counts. Absence opportunities are not factual false labels. No ambiguous/inferred or
dont_care labels are synthesized. authentication_reference has NO positive examples:
the pinned sanitizer redacts embedded secret:// text, so evidence would not survive
in the required prompt. Its schema definition and negative opportunities remain;
this deliberate coverage gap is preferable to ungrounded labels or bypassing redaction.
Complex object values are explicitly written as JSON inside natural clauses.
Some enum diversity is restricted to keep simulated facts
coherent. Deterministic English-only templates are not human paraphrases. Prompt-schema
length, boilerplate similarity and context-conditioned short reply repetition remain.
Fuzzy audit compares one representative per source, not all pairwise variants or old
corpora. No training, live model evaluation, network calls or human review occurs here.
