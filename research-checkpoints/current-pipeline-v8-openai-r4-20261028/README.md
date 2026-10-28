# R4 native-admission actionable-update comparison

Completed matched corrective DEVELOPMENT comparison: 20 frozen conversations,
101 turns per model/track, 404 main scored calls. OpenAI main 202/202 calls are
FRESH. SLM main 202/202 responses are REUSED from R3 after proving exact identity
of all component/rollout inputs and application trajectories under native parsing.
Both models are rescored under the same NEW R4 rules. The separate nine-turn
pilot contributes 18 fresh OpenAI and 18 reused SLM calls and is excluded from
main metrics. Do not describe both arms as fresh inference.

The cohort retains one conversation per 20 domains: six weather, twelve economic
and two combined domains; sixteen English, two Roman Urdu and two mixed cases.
There are 257 gold actionable non-intent updates per track and positive support
for 39 of the 79 slot types. These are not all-domain/all-slot accuracy claims.

The common prompt/context and gold remain identical to R3. Both arms now use
the native Pydantic candidate decoder once, including its ordinary-string fallback.
Ignored statuses unmentioned/dont_care are excluded from actionable predictions
for both arms, with their counts and literal empty-list output reported separately.
Native admission, JSON/schema validity, duplicate entries, evidence/reducer
admission and resulting state are distinct. Duplicate actionable updates retain
multiset penalties. No paraphrase/synonym repair or numeric semantic normalization
was introduced. Exact slot/value/status F1 is not general semantic accuracy.

Raw scoring precedes reducer normalization and deterministic recovery. Component
uses supplied gold prior; rollout uses its own accepted state except the explicitly
declared synthetic interventions. Current fpy plus the verified bundled fixes is
used, with the unchanged common augmented extraction adapter. Static generated
questions are not fed into later scripted turns. This is not an unmodified native
provider-prompt or adaptive-interview benchmark. The oracle label audit still
records application recovery mismatches even when given gold extraction.

All frozen source/case/adapter hashes were checked. The 299-test suite passed;
the first test collection attempt lacked explicit fpy import roots and is preserved
beside the successful command/log. All 404 raw scores, inputs and pipeline states
were replayed offline after completion. All 220 fresh hosted SDK responses and
usage receipts match the durable ledger, and all 440 prior call rows and the
original authorization row remain unchanged. The additional-dollar approval is
an append-only amendment, not a spending reset. No automatic paid retries.

A versioned R4.1 continuation handles only a local static-question renderer
length validation error. R4 stopped after 132 paid calls; all were copied and
replayed with exactly identical inputs, under the SAME paid-request namespace,
then only the remaining 88 were dispatched. Native extraction/admission, scoring,
prompts, gold and decoding did not change. The original blocked run and source
are preserved; renderer failures are separately recorded and do not repair model
output. This is a fixed-script evaluation wrapper correction, not a production
application fix.

This is inspected, related synthetic development, not independent confirmation.
The 20 cases share earlier public templates; independent human gold/disagreement
review remains INCOMPLETE. No significance, equivalence, general superiority,
deployment readiness or paper-wide validation claim follows from these estimates.
Do not combine this hosted row with SLM scores from another scoring protocol.
No new training, model substitution, sealed/private-label access or paid judge.
No model weights, caches, optimizer state or credentials are included.

Result folder: ../current-pipeline-v8-openai-20261028T094502Z-r4-1-native

| Model | Track | Precision | Recall | Exact actionable F1 | Native admitted | Exact pipeline turns | Median seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| v8 | component | 0.976285 | 0.961089 | 0.968627 | 100/101 | 75/101 | 8.082 |
| v8 | rollout | 0.964427 | 0.949416 | 0.956863 | 100/101 | 52/101 | 8.656 |
| openai | component | 0.644518 | 0.754864 | 0.695341 | 101/101 | 54/101 | 2.669 |
| openai | rollout | 0.636364 | 0.762646 | 0.693805 | 101/101 | 6/101 | 2.553 |

R4 usage-rate cost estimate: $0.35448805; cumulative task usage estimate: $0.94693530; cumulative conservative ledger accounting: $1.0469865. Reserved/uncertain $0. Rates include reasoning once; cached input affects the estimate but never dispatch permission. This is not a provider invoice. Total authorization $2; task dispatch ceiling $1.90; this rerun's new-spend ceiling $0.90.

ZIP SHA-256: 4d4a33bd6ec4ac1bab28b47d37ee324a315177035826119e36a230cd71c143ae
