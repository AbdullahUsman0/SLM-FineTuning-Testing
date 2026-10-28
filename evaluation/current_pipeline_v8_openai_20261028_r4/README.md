# R4: fresh hosted rerun with native admission and actionable-update scoring

This is a new, corrective inspected-development protocol. It is not independent
validation or an unmodified-production-prompt study. Do not promise a higher
hosted score or replace the original r2/r3 artifacts.

The 20 frozen main conversations (101 turns) and two separate pilots (nine
turns) retain their exact case hashes, raw gold and approved state adjudication.
Both models use the same unchanged R3 common instructions/context/schema.
The base, v8 adapter, current fpy revision plus bundled fixes, and generation
limits are unchanged. Hosted calls use exactly gpt-5-mini-2025-08-07, 2,048 output
tokens, minimal reasoning, no retries and direct api.openai.com transport.

SLM responses are REUSED, not freshly generated. Before any new hosted call,
prepare.py verifies all 220 SLM pilot/main component/rollout input identities
and resulting pipeline states under the native parser against their original
records. Any drift prohibits reuse. The source hashes and proof are frozen.
OpenAI pilot/main responses are all fresh and use a new ledger namespace.

Both arms are rescored from raw responses using the native Pydantic candidate
decoder once, including its fallback for ordinary strings. The same native
parser performs application admission. Schema validity is separately reported;
it is not an extra whole-response gate. Duplicate slot entries are reported and
remain duplicate-preserving predictions, rather than invalidating the entire
call. Native ignored statuses unmentioned/dont_care are excluded from actionable
tuples for both models and counted separately. No synonym, paraphrase, numeric
normalization, value repair, recovery or gold-state assistance receives raw
model credit. Capped/refused/unparseable responses receive no true positives.

Primary metric: micro exact non-intent actionable slot/native-decoded-value/status
precision, recall and F1. This is not semantic accuracy or end-to-end success.
Secondary measures: JSON/schema/native admission, duplicate/ignored updates,
extra/forbidden actionable slots, corrections, unknown no-action versus literal
empty output, prior-state preservation, raw versus deterministic recovery,
exact complete state by turn/final conversation, extraction latency and usage.
Component uses gold prior; rollout uses own accepted state and only declared
synthetic interventions. Static assistant questions are not fed into later turns.

The user approved ONE additional dollar. The original shared ledger and all 440
earlier paid requests remain intact. The original authorization row stays at $1;
an append-only amendment records $2 total and $1.90 task dispatch ceiling. This
rerun also has its own $0.90 NEW-spending limit above the fixed $0.64702325
baseline. All pilots/failures/retries count; pending/uncertain reservations stop
dispatch. Amendment installation is explicit and guarded by the shared lock.

Use the recorded interpreter and offline-deps, CURRENT_PIPELINE_RUN, exact source
hashes and offline-tests.json. A detached supervisor reloads the Windows user
credential, clears conflicting provider overrides and verifies server metadata.
Never print credentials or include environment files, credentials, base weights,
adapter weights, optimizer states or caches in the results ZIP.

Report the new comparison as fresh hosted versus trajectory-verified reused
SLM on a common adapter. Do not mix new OpenAI metrics with old strict-scoring
SLM metrics. Do not claim independent human review, general superiority,
deployment readiness or an adaptive interview study. No training or sealed
labels are involved. The study ends after this bounded run and publication.
