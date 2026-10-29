"""Paper-ready supplements and blank future protocol; no manuscript replacement."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
from datetime import datetime
import importlib.metadata
from common import OUT,E,ROOT,dump,write,csv_write,sha


def main():
    analysis=OUT/'analysis';paper=OUT/'paper';future=OUT/'future-independent-evaluation';paper.mkdir(exist_ok=True);future.mkdir(exist_ok=True)
    summary=json.loads((E/'main-summary.json').read_bytes());audit=json.loads((analysis/'oracle-20-mismatches.json').read_bytes());intervals=json.loads((analysis/'exploratory_paired_intervals.json').read_bytes())
    mechanism=['# Oracle recovery mechanism audit','', 'Scope: 101 frozen main component turns supplied with gold extraction and gold prior. These are deterministic application replays, not generated model responses or new end-to-end rollouts.','',f"All {audit['mismatch_calls']} published post-pipeline oracle mismatch keys reproduce. All 20 of those calls match expected state at the pre-recovery reducer snapshot. Thus their discrepancies arise in the later application path, not from missing/wrong oracle extraction.",'','| Changed slot | Mismatch calls |','|---|---:|']
    for sid,n in Counter(s for x in audit['findings'] for s in x['changed_slots']).items():mechanism.append(f'| {sid} | {n} |')
    explicit=all(set(x['changed_slots'])<=set(x['recovery_slot_ids']) for x in audit['findings'])
    mechanism+=['',f'All changed slots have explicit recorded recovery IDs: {explicit}.','', 'Under this fixed-script adapter, selected-slot recovery can assign the current message to a pending free-text slot even when the message states other requirements or explicitly withholds information. This is an observed failure mode of these fixtures; it is not an estimate of its frequency in a live interview with assistant-question context.','', 'The 14 output_granularity discrepancies, four source_reference discrepancies, one target_unit discrepancy and one source_mode discrepancy are retained with messages, expected/pre/post states and recovery IDs in oracle-20-mismatches.json. Full event traces are in oracle-traces.jsonl.gz. No engine patch or automatic label correction was made.','', 'The raw extraction/state separation matters: refusing to invent a model update does not prevent a later fallback from assigning an unrelated value. The seven rollout unknown-state changes after correct abstention in the local arm must not be described as model hallucinations without this trace distinction.','', 'A future fix should be assessed in a newly versioned application study, with reviewed question/answer context and declared intervention policy. Saved outputs replayed under different rollout contexts are not a fresh experiment.']
    (analysis/'ORACLE_MECHANISM.md').write_text('\n'.join(mechanism)+'\n',encoding='utf-8')
    table=['| System | Track | JSON | Schema / native | Precision | Recall | Exact actionable F1 | Exact pipeline turns |','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['v8','openai']:
        for track in ['component','rollout']:
            r=summary[arm][track];m=r['raw_exact'];table.append(f"| {arm} | {track} | {r['json_valid']}/101 | {r['schema_valid']}/101 / {r['raw_contract_admitted']}/101 | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {r['exact_pipeline_state_turns']}/101 |")
    table='\n'.join(table)
    text='''# Suggested replacement paragraphs for the existing paper

These are proposed insertions, not an edited or replacement manuscript. The latest
six-page paper resides on the main PC and was not available here. Preserve its
cohort history and references; apply only after checking that draft.

## Methods: separate three reliability layers

We distinguish structural validity, actionable requirement extraction and application
state reliability. The completed R4/R4.1 comparison uses 20 previously inspected
synthetic development conversations (101 turns), with one conversation per domain,
six weather/twelve economic/two combined domains and 16 English/two Roman Urdu/two
mixed conversations. Both systems use a common augmented instruction/schema
adapter and identical component inputs; rollout uses each system's accepted state,
subject only to declared synthetic interventions. Native Pydantic candidate parsing
is applied once to both systems. The primary exact slot/value/status multiset F1
excludes intent and non-actionable unmentioned/dont_care entries, which are reported
separately. Exact raw scoring precedes normalization and deterministic recovery.

The hosted arm made 202 fresh main calls and 18 separate pilot calls. The local
arm reused 202 main and 18 pilot responses only after verifying identical input
identities and native-parser state trajectories; both arms were rescored under
the same R4 rules. Local decoding was CUDA BF16 greedy generation with thinking
disabled and 1,024 new tokens; hosted Structured Outputs used the pinned
gpt-5-mini-2025-08-07 snapshot, minimal reasoning and 2,048 output tokens. An R4.1
wrapper continuation bounded an unscored static-question length error without
changing extraction inputs or redispatching completed paid calls. Questions were
not fed into subsequent scripted turns. This evaluates configured systems on fixed
scripts, not an unmodified production prompt or adaptive interview.

## Results: validity is not extraction or state success

All 404 main raw scores, inputs and pipeline states reproduce offline. Local exact
actionable F1 is 0.9686274509803922 component and 0.9568627450980393 rollout; hosted
F1 is 0.6953405017921147 and 0.6938053097345133. Hosted responses are JSON/schema/native
admitted on 101/101 calls per track, yet complete final rollout state is correct in
only 1/20 conversations, compared with 9/20 locally. Component final snapshots use
gold prior and must not be interpreted as successful whole conversations. Report
the configured latency descriptively, not as compute/energy efficiency.

## Error analysis: extraction and recovery can disagree

The local arm emits no actionable updates on all 20 rollout unknown turns, but its
application preserves prior state on only 13; the hosted arm preserves state on
15/20. For the hosted arm, 15 no-actionable outputs and 15 preserved states overlap
on 14 turns, so equal totals do not imply identical behavior. In the oracle replay,
all 20 post-pipeline mismatch calls have correct pre-recovery reducer states despite
gold current extraction and gold prior. Recorded recovery affects output_granularity
in 14 calls, source_reference in four, and target_unit/source_mode in one each.
These trace-backed application failures must not be assigned to raw model extraction.

## Coverage and exploratory uncertainty

Gold contains 257 actionable non-intent updates per track, covering 39 of 78
non-intent slot types. The supplement reports all 78 slots, unsupported-slot and
unknown-ID false positives, per-conversation counts, language/domain denominators
and declared-intervention sensitivity. Supported-slot macro F1 uses the fixed
39-slot positive-support universe and does not replace the primary inclusive
micro metric. The subset does not establish quality across every schema slot.

The post-hoc supplementary plan was saved before calculating new intervals.
Exploratory paired bootstrap resamples whole conversations and recomputes micro
F1 from summed counts, with a paired script-family sensitivity. Distinct entity
IDs are not evidence of independence: 18/20 cases share one template variant.
Intervals are conditional on inspected, related synthetic cases and do not establish
population superiority, equivalence or deployment readiness. No p-values are claimed.

## Scope and remaining validation

Independent human annotation remains incomplete. Gold-first and blinded paired
review forms are prepared, with decisions blank and a coordinator-only identity
mapping; preparation is not annotation. Existing A-F evidence spans different
models, cohorts and policies, rather than a completed two-model-by-six-method
factorial study. Historical model-size runs are not a controlled size experiment.
An independently authored, reviewed prospective cohort and any future inference
need separate approval. These findings support the paper's separation of structure,
actionable extraction and application-state reliability, with bounded development
claims and explicit failure mechanisms.
'''
    (paper/'REPLACEMENT_PARAGRAPHS.md').write_text(text+'\n\n'+table+'\n',encoding='utf-8')
    (paper/'MAIN_TABLE.md').write_text(table+'\n\nComplete final rollout state: local 9/20; hosted 1/20. Do not report component final snapshots as complete conversations.\n',encoding='utf-8')
    ci=['# POST-HOC exploratory paired intervals','','Paired differences are local minus hosted. Percentile 95% intervals use 10,000 paired resamples, PCG64 seed 20261028. These conditional estimates do not certify independence or population superiority.','','| Track | Resampling unit | Units | Difference | Conditional interval |','|---|---|---:|---:|---|']
    for r in intervals['intervals']:
        lo,hi=r['paired_difference_interval'];ci.append(f"| {r['track']} | {r['scheme']} | {r['units']} | {r['difference_point']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    (paper/'EXPLORATORY_INTERVAL_TABLE.md').write_text('\n'.join(ci)+'\n',encoding='utf-8')
    (paper/'SIX_PAGE_MAIN_SUPPLEMENT_SPLIT.md').write_text('''# Suggested allocation within the existing six-page IEEE draft

This is an editing guide, not a replacement manuscript or a claim to have built
the latest main-PC draft. Preserve its N/M/P/V8-D cohort distinctions.

| Main page | Suggested focus |
|---|---|
| 1 | Abstract, task and contribution: distinguish structure/extraction/state |
| 2 | Compact related work; forecasting schema and annotation/abstention policy |
| 3 | Common-adapter methods, fixed scripts, component/rollout, fresh/reused provenance |
| 4 | Main four-row results table; final rollout state and unknown-state contrast |
| 5 | One recovery-harm example; concise coverage and conditional uncertainty note |
| 6 | Validity limits, pending human review, conclusion and references as space permits |

Use the actual draft's existing layout to adjust these allocations. Avoid expanding
historical A-F or size-study tables into a new main factorial claim.

Supplement: S1 frozen sources/manifests/replay logs; S2 blank staged human-review
kit and coordinator-only mapping; S3 all 78 slots, support/macro definitions,
conversation/subgroup/intervention tables; S4 post-hoc plan and exploratory paired
intervals/draws; S5 taxonomy, 20 oracle traces and six examples; S6 fairness/A-F
evidence tiers and missing legacy receipts; S7 unexecuted independent-study proposal.

Keep full raw SDK responses in the sealed evidence archive, not the main PDF.
Only the coordinator should receive the full supplement containing the blinding key.
''',encoding='utf-8')
    (future/'PROTOCOL_PROPOSAL.md').write_text('''# Future independent evaluation — PROPOSAL ONLY, NOT EXECUTED

No requests, labels, participants or model outcomes have been collected here.
The forms are blank. Existing paid approvals do not authorize this study.

## Proposed design requiring human approval

Target 40 independently human-authored conversations, two per each of the 20
forecasting domains (12 weather, 24 economic, four combined conversations), with
three to five human-written turns each. Proposed language allocation: 28 English,
six Roman Urdu and six mixed, subject to recruiting feasibility and a new frozen
decision before inference. Do not present sparse per-domain results as precise.
At least five writers, at most eight conversations each, with writer/group IDs
retained for dependence analysis. These are proposed resources, not enrolled people.

Writers must not see inspected templates, R4 examples, model predictions or
training-generation scaffolds. They should compose their own requirements and
natural corrections/unknowns. Collect provenance/attestation and check public
training/development overlap without inspecting sealed labels. Lexical overlap
screening does not prove semantic independence. Exclude inspected/paraphrased
templates, copied benchmark requests, agent-authored/generated requests and any
request containing real credentials or personal/sensitive material. Log exclusions
and decide replacements before outcomes; do not select easier cases for either arm.

Gold-first annotation by two humans who did not develop the extraction models,
followed by a separate adjudicator. Record annotation independence, disagreements,
semantic variants versus exact label boundaries, evidence and application-state
expectations. Freeze labels, allowable semantic variants, scripts, schema, parser,
normalization/recovery policy, model/base/adapter revisions, common instructions,
decoding/output caps, metrics and failure handling BEFORE inference. Independent
review and explicit freeze approval are gates, not claims this proposal completes.

Use fresh local and hosted emissions on identical frozen scripts for a matched
study; component versus own-state rollout and raw versus pipeline measures remain
separate. A live adaptive-interview evaluation is a different protocol because
model questions lead to different user inputs. Do not call fixed scripts real-user
adaptive interviews. Reusing outputs after changed contexts is not fresh rollout.

Prespecify exact actionable F1, semantic acceptability (with reviewed variants),
JSON/schema/native admission, evidence/extra requirements, correction/unknown
handling, complete final-state success and latency. Choose primary endpoints and
practically meaningful criteria before outcomes. Plan paired writer/conversation
or defensible cluster resampling; justify exchangeability and avoid turn-level
independence or unsupported power claims. Retain all failed/capped/refused calls.

## Resources and approvals — none supplied by this package

Illustrative workload: 40 conversations x 3-5 turns = 120-200 turns. At two models
and two context tracks this is 480-800 extractor calls, with 240-400 hosted calls,
plus separately bounded pilots. This is arithmetic planning, not an approved run
or a promised provider cost. A new pricing/account/model-availability check,
explicit paid budget, durable spending guard and compute authorization are needed.

Illustrative human effort assumptions: 10 minutes/writer conversation is about
6.7 writer-hours; two annotations per 120-200 turns at two minutes each is 8-13.3
reviewer-hours, plus adjudication/coordination. These are unvalidated planning
assumptions; measure a small separately authorized human pilot before committing
resources. No participant was contacted and no recruitment or payment is approved.

Required decisions: supervisor/institutional ethics or exemption assessment where
applicable; consent/privacy/data-retention plan; writers/reviewers/adjudicator and
independence; cohort/language feasibility; exact versus semantic rubric; endpoint
and grouping plan; model/inference resources; new API budget; publication consent.
Do not recruit, collect data, run models, train or publish without those approvals.
''',encoding='utf-8')
    forms={'independent_request_collection.csv':['request_id','writer_pseudonym','domain','language','initial_request','human_written_followup_turns_json','provenance_attestation','no_inspected_template_attestation','no_sensitive_data_attestation','coordinator_exclusion_decision','exclusion_rationale'],
           'independent_gold_annotation.csv':['request_id','turn_id','reviewer_pseudonym','prior_context_json','expected_updates_json','expected_state_json','allowed_semantic_variants_json','evidence_spans_json','decision','rationale'],
           'independent_adjudication.csv':['request_id','turn_id','reviewer_1_pseudonym','reviewer_2_pseudonym','disagreement_json','adjudicator_pseudonym','decision','frozen_gold_json','rationale'],
           'approval_checklist.csv':['decision_area','approver_identity','approval_decision','approved_scope','approval_date','notes']}
    for name,fields in forms.items():csv_write(future/name,[],fields=fields)
    write(future/'STATUS.json',{'status':'prepared_unexecuted_proposal','participants_contacted':0,'human_requests_collected':0,'new_model_calls':0,'new_paid_approval':False,'independent_human_review_complete':False,'agent_authored_cases_called_independent':False})
    shutil.copyfile(ROOT/'scripts/import_reviews.py',OUT/'human-review/import_reviews.py')
    write(OUT/'TIMEZONE_ENVIRONMENT.json',{'timezone':'Asia/Karachi','available':True,'UTC_offset_on_2026_10_28':str(datetime(2026,10,28,tzinfo=ZoneInfo('Asia/Karachi')).utcoffset()),'tzdata_version':importlib.metadata.version('tzdata'),'environment_packages_modified':False,'bundled_modified_test_files_restored':True})
    print(dump({'paper_proposals_written':True,'oracle_recovery_ids_cover_all_mismatches':explicit,'future_forms_header_only':len(forms),'future_experiment_executed':False}))


if __name__=='__main__':main()
