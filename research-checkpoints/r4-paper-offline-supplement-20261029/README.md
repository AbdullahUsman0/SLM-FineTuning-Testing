# R4 paper evidence supplement and current main-PC handoff

Read [LOCAL_CODEX_HANDOFF.md](LOCAL_CODEX_HANDOFF.md) first. The user authorized
publication of these completed artifacts on 29 October 2026. No new inference,
paid API calls, training, human annotation or paper-wide validation is implied.

The complete ZIP is byte-identical to the previously verified local supplement:

- File: [r4-paper-offline-supplement-20261028-r2.zip](r4-paper-offline-supplement-20261028-r2.zip)
- SHA-256: `a09f33e7c36bab427991e0ed30a5916a926e99035d1c661e4cf926425a4ed3f7`
- Verification: CRC, all 246 archive members, copy/extraction round trip;
  the shipped source layout reproduced 299 tests and 404 saved main results.
- Additional supplement controls: 10 tests passed.
- Nested original R4 evidence remains unchanged at SHA-256
  `4d4a33bd6ec4ac1bab28b47d37ee324a315177035826119e36a230cd71c143ae`.

The ZIP and `supplement/` preserve the sealed analysis. Their internal
"unpublished" flags and "separate authorization" statements describe the earlier
local preparation stage. This root handoff records the subsequent publication
authorization. PUBLICATION_VERIFICATION.json records the GitHub download check.

## What is included

| Material | Location |
|---|---|
| Completion and artifact index | [supplement/ANALYSIS_COMPLETION.json](supplement/ANALYSIS_COMPLETION.json), [index](supplement/deliverables/README.md) |
| Human-review kit, guide and proposal-only importer | [human-review](supplement/deliverables/human-review) |
| Gold-first reviewer-only packet | [01-gold-first-review-kit.zip](reviewer-kits/01-gold-first-review-kit.zip) |
| Blinded-response reviewer-only packet | [02-blinded-response-review-kit.zip](reviewer-kits/02-blinded-response-review-kit.zip) |
| All-slot coverage, subgroups, intervals and error traces | [analysis](supplement/deliverables/analysis) |
| Fairness and A-F evidence tiers | [fairness-and-methods](supplement/deliverables/fairness-and-methods) |
| Paper replacement paragraphs and six-page allocation | [paper](supplement/deliverables/paper) |
| Unexecuted independent-study protocol and blank forms | [future proposal](supplement/deliverables/future-independent-evaluation) |
| Reproduction source and test logs | [code](supplement/code), [logs](supplement/logs), [source snapshot](supplement/reproduction) |

## Important findings

| System | Component exact actionable F1 | Rollout exact actionable F1 |
|---|---:|---:|
| v8 | 0.9686274509803922 | 0.9568627450980393 |
| OpenAI | 0.6953405017921147 | 0.6938053097345133 |

The matched cohort has 20 conversations and 101 turns per model/track. Hosted
calls were fresh; local responses were verified reused and both were rescored
under common R4 rules. Complete final rollout state is 9/20 versus 1/20.
Component final snapshots are not successful whole conversations.

All 20 oracle mismatch calls have correct pre-recovery states, then differ after
trace-reported recovery: 14 output_granularity, four source_reference and one
each target_unit/source_mode. Local abstention on unknown rollout turns is 20/20,
but prior preservation is 13/20; hosted preservation is 15/20. These findings
separate structural validity, actionable extraction and application-state reliability.

Gold positively supports 39/78 non-intent slot types. Paired exploratory intervals
are POST-HOC and conditional on inspected synthetic cases. Eighteen cases share
one template variant; neither conversation nor script-family intervals establish
population independence or general superiority. Independent human review is pending.

## Reviewer distribution

The full bundle contains the coordinator mapping, model identities and scores.
Give reviewers only the staged ZIPs, gold first and A/B responses after that pass
is locked. Do not give them the full bundle, source archive or coordinator folder.
See [COORDINATOR_NOTICE.md](COORDINATOR_NOTICE.md). The public repository is not
an access-control mechanism for blinding; do not claim identities cannot be inferred.
