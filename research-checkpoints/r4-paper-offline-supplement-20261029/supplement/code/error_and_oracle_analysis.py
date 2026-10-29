"""Trace-backed post-hoc taxonomy; no new labels or changed rollout contexts."""
from collections import Counter,defaultdict
from copy import deepcopy
import gzip
import json
from common import OUT,E,load,p,dump,write,csv_write,prior_states,mismatches,counts_to_prf,sha
from forecasting_assistant.application.state_reducer import apply_extraction,_canonical_evidence
from forecasting_assistant.application.normalization import normalize_value

TEMPORAL={'frequency','prediction_frequency','refresh_frequency','retraining_frequency','expected_history_length','forecast_horizon','lead_time'}
TAXONOMY={
    'missing_current_slot':'update','extra_current_actionable_slot':'update','admission_denied_readable_exact_matches':'update',
    'same_slot_value_wrong_status':'update','same_slot_exact_value_discrepancy':'update','cross_slot_value_match_candidate':'update','temporal_role_swap_candidate':'update','repeated_unchanged_prior_value_candidate':'update',
    'json_valid_failure':'call indicator','schema_valid_failure':'call indicator','raw_contract_admitted_failure':'call indicator','reducer_admitted_failure':'call indicator',
    'duplicate_nonintent_entries':'surplus update','ignored_status_entries_not_actionable_errors':'ignored update','correction_flag_discrepancy':'call indicator',
    'unsupported_evidence_native_canonicalizer':'update','nonverbatim_evidence_text':'update','raw_to_normalized_candidate_representation_change':'candidate update','normalizer_candidate_error':'candidate update',
    'trace_reported_recovery_slots':'trace slot event','post_reducer_pipeline_changed_slots':'slot transition','trace_reported_recovery_changed_slots':'slot transition','other_engine_changes_after_simple_reducer':'slot transition',
    'recovery_introduced_wrong_slot_state':'slot transition','recovery_corrected_slot_state':'slot transition','correct_reducer_state_harmed_by_recovery':'call indicator','exact_raw_extraction_with_recovery_harm':'call indicator',
    'inherited_rollout_prior_slot_discrepancy':'slot discrepancy','final_turn_state_slot_discrepancy':'slot discrepancy','state_status_discrepancy':'slot discrepancy','state_confirmation_flag_discrepancy':'slot discrepancy'}


def write_taxonomy_table(aggregate,out):
    output=[]
    for arm in ['v8','openai']:
        for track in ['component','rollout']:
            for category,unit in sorted(TAXONOMY.items()):
                item=aggregate.get((arm,track,category),{})
                output.append({'model':arm,'track':track,'category':category,'unique_affected_calls':item.get('calls',0),'call_denominator':101,'count_unit':unit,'individual_update_or_slot_count':item.get('individual_updates_or_slot_transitions',0) if unit!='call indicator' else ''})
    csv_write(out/'error_taxonomy_counts.csv',output)


def main():
    cases,rows,schema=load();out=OUT/'analysis';out.mkdir(exist_ok=True);turns={(c['scenario_id'],t['turn_id']):(c,t) for c in cases for t in c['turns']};priors=prior_states(cases,rows)
    detail=[];aggregate=defaultdict(lambda:Counter(calls=0,individual_updates_or_slot_transitions=0));recovery=[];unknown=[]
    for row in rows:
        key=tuple(row['key'][1:]);case,turn=turns[tuple(row['key'][3:])];prior=priors[key]['slots'];gold_prior={s:v for s,v in turn['context_state'].items() if v.get('value') is not None}
        obj=json.loads(row['raw_output']) if row['json_valid'] else {};expected=Counter(p.tuples(turn['gold_extraction']));pred=Counter(p.tuples(obj));readable_match=expected&pred
        missing=expected-readable_match;extra=pred-readable_match;gold_ids={t[0] for t in expected};pred_ids={t[0] for t in pred};flags=Counter();candidates=[]
        def mark(name,n):
            if n:flags[name]+=int(n)
        mark('missing_current_slot',sum(n for t,n in missing.items() if t[0] not in pred_ids))
        mark('extra_current_actionable_slot',row['extra_slot_flags'])
        mark('admission_denied_readable_exact_matches',sum(readable_match.values()) if not row['raw_contract_admitted'] else 0)
        for candidate,n in extra.items():
            same=[g for g in expected if g[0]==candidate[0]]
            if any(g[1]==candidate[1] and g[2]!=candidate[2] for g in same):mark('same_slot_value_wrong_status',n)
            if any(g[1]!=candidate[1] for g in same):mark('same_slot_exact_value_discrepancy',n)
            cross=[g for g in expected if g[0]!=candidate[0] and g[1:]==candidate[1:]]
            if cross:
                mark('cross_slot_value_match_candidate',n)
                temporal=[g for g in cross if g[0] in TEMPORAL and candidate[0] in TEMPORAL]
                if temporal:mark('temporal_role_swap_candidate',n)
                candidates.append({'emitted_tuple':candidate,'matching_other_gold_slots':[g[0] for g in cross],'human_confirmation_required':True})
            if candidate[0] not in gold_ids and candidate[0] in prior:
                if json.loads(candidate[1])==prior[candidate[0]]['value']:mark('repeated_unchanged_prior_value_candidate',n)
        for name in ['json_valid','schema_valid','raw_contract_admitted','reducer_admitted']:
            mark(name+'_failure',not row[name])
        ids=[u.get('slot_id') for u in obj.get('updates',[]) if isinstance(u,dict) and u.get('slot_id')!='intent']
        mark('duplicate_nonintent_entries',sum(max(0,n-1) for n in Counter(ids).values()))
        mark('ignored_status_entries_not_actionable_errors',row['ignored_status_updates'])
        if row['raw_contract_admitted']:
            parsed=p.admit(row['raw_output'],p.wire_schema())
            mark('correction_flag_discrepancy',parsed.correction_detected!=turn['gold_extraction']['correction_detected'])
            for update in parsed.updates:
                if update.slot_id=='intent' or update.status.value in {'unmentioned','dont_care'}:continue
                mark('unsupported_evidence_native_canonicalizer',_canonical_evidence(update,turn['message'],turn['message'].casefold()) is None)
                mark('nonverbatim_evidence_text',update.evidence_text not in turn['message'])
                try:
                    normalized=normalize_value(schema.get(update.slot_id),update.candidate_value)
                    mark('raw_to_normalized_candidate_representation_change',dump(normalized)!=dump(update.candidate_value))
                except Exception:mark('normalizer_candidate_error',1)
        reduced=row['reducer_state_before_recovery'];after=row['pipeline_state'];reference=row['expected_state']
        changed=mismatches(reduced,after);reported=set(row['provenance']['recovered_slot_ids']);explicit=[s for s in changed if s in reported]
        harm=[s for s in changed if reduced.get(s)==reference.get(s) and after.get(s)!=reference.get(s)]
        help_=[s for s in changed if reduced.get(s)!=reference.get(s) and after.get(s)==reference.get(s)]
        mark('trace_reported_recovery_slots',len(row['provenance']['recovered_slot_ids']))
        mark('post_reducer_pipeline_changed_slots',len(changed));mark('trace_reported_recovery_changed_slots',len(explicit))
        mark('other_engine_changes_after_simple_reducer',len(set(changed)-reported))
        mark('recovery_introduced_wrong_slot_state',len(set(harm)&reported));mark('recovery_corrected_slot_state',len(set(help_)&reported))
        before_bad=mismatches(reduced,reference);after_bad=mismatches(after,reference)
        if not before_bad and after_bad and set(harm)&reported:mark('correct_reducer_state_harmed_by_recovery',1)
        if row['confusion'][1:]==[0,0] and sum(expected.values())==row['confusion'][0] and set(harm)&reported:mark('exact_raw_extraction_with_recovery_harm',1)
        inherited=mismatches(prior,gold_prior) if row['key'][2]=='rollout' else []
        mark('inherited_rollout_prior_slot_discrepancy',len(inherited))
        mark('final_turn_state_slot_discrepancy',len(after_bad))
        for sid in after_bad:
            a=after.get(sid);g=reference.get(sid)
            if a is not None and g is not None and a['value']==g['value']:
                mark('state_status_discrepancy',a['status']!=g['status']);mark('state_confirmation_flag_discrepancy',a['confirmed_by_user']!=g['confirmed_by_user'])
        item={'key':row['key'],'category_counts':dict(flags),'raw_confusion':row['confusion'],'prior_state':prior,'gold_reference_prior':gold_prior,'pre_recovery_state':reduced,'post_recovery_state':after,'expected_state':reference,'recovery_changed_slots':explicit,'post_reducer_changed_slots':changed,'recovery_harmed_slots':[s for s in harm if s in reported],'recovery_helped_slots':[s for s in help_ if s in reported],'prior_discrepant_slots':inherited,'exact_after_discrepant_slots':after_bad,'slot_swap_review_candidates':candidates,'interpretation':'Overlapping diagnostic categories; exact discrepancies and cross-slot candidates need human semantic review; normalization candidates are not proof each update was committed.'}
        detail.append(item)
        for category,n in flags.items():aggregate[(row['key'][1],row['key'][2],category)].update(calls=1,individual_updates_or_slot_transitions=n)
        if changed:recovery.append({'key':row['key'],'changed_slots':changed,'reported_recovery_slots':sorted(reported),'harmed_slots':[s for s in harm if s in reported],'helped_slots':[s for s in help_ if s in reported],'before_discrepant_slots':before_bad,'after_discrepant_slots':after_bad})
    (out/'error-taxonomy-per-call.jsonl.gz').write_bytes(gzip.compress(('\n'.join(dump(x) for x in detail)+'\n').encode(),mtime=0))
    write_taxonomy_table(aggregate,out)
    write(out/'post_reducer_recovery_changes.json',recovery)
    for arm in ('v8','openai'):
        for track in ('component','rollout'):
            group=[r for r in rows if r['key'][1:3]==[arm,track] and r['unknown_turn']]
            unknown.append({'model':arm,'track':track,'unknown_turns':len(group),'raw_no_actionable_updates':sum(r['unknown_no_updates'] for r in group),'literal_wire_empty_nonintent_updates':sum(r['wire_empty_nonintent_updates'] for r in group),'prior_preserved':sum(r['unknown_preserved'] for r in group),'raw_no_actionable_and_preserved':sum(r['unknown_no_updates'] and r['unknown_preserved'] for r in group),'raw_no_actionable_but_state_changed':sum(r['unknown_no_updates'] and not r['unknown_preserved'] for r in group)})
    assert next(x for x in unknown if x['model']=='v8' and x['track']=='rollout')['raw_no_actionable_updates']==20
    assert next(x for x in unknown if x['model']=='v8' and x['track']=='rollout')['prior_preserved']==13
    assert next(x for x in unknown if x['model']=='openai' and x['track']=='rollout')['prior_preserved']==15
    csv_write(out/'unknown_extraction_vs_state.csv',unknown)
    # Oracle: reproduce gold-prior component diagnostics, not a new model rollout.
    oracle=[];oracle_traces=[]
    for case in cases:
        for turn in case['turns']:
            state=p.restore(turn,schema);raw=dump(turn['gold_extraction']);record={'raw_output':raw,'capped':False,'refused':False}
            gold=p.admit(raw,p.wire_schema());reduced=p.snapshot(apply_extraction(state,gold,schema,len(state.turns)+1,turn['message']))
            final,trace=p.run_pipeline(state,turn['message'],schema,record,p.wire_schema());after=p.snapshot(final);expected_state=turn['expected_state_after'];diff=mismatches(after,expected_state)
            if diff:
                before_diff=mismatches(reduced,expected_state)
                oracle.append({'scenario_id':case['scenario_id'],'turn_id':turn['turn_id'],'message':turn['message'],'changed_slots':diff,'pre_recovery_mismatches':before_diff,'recovery_slot_ids':trace['recovered_slot_ids'],'mismatch_exists_before_recovery':bool(before_diff),'pre_recovery_state':reduced,'post_recovery_state':after,'expected_state':expected_state,'interpretation':'Gold component extraction supplied; application discrepancy remains. No inference or label correction.'})
                oracle_traces.append({'scenario_id':case['scenario_id'],'turn_id':turn['turn_id'],'raw_gold_response':raw,'trace':trace})
    original=json.loads((E/'label-audit.json').read_bytes())['main']['pipeline_oracle_findings']
    assert len(oracle)==len(original)==20
    assert {(x['scenario_id'],x['turn_id']) for x in oracle}=={tuple(x['key']) for x in original}
    write(out/'oracle-20-mismatches.json',{'oracle_calls':101,'mismatch_calls':20,'original_keys_reproduced':True,'mismatch_calls_existing_before_recovery':sum(x['mismatch_exists_before_recovery'] for x in oracle),'recovery_only_mismatch_calls':sum(not x['mismatch_exists_before_recovery'] for x in oracle),'findings':oracle})
    (out/'oracle-traces.jsonl.gz').write_bytes(gzip.compress(('\n'.join(dump(x) for x in oracle_traces)+'\n').encode(),mtime=0))
    detail_by_key={tuple(x['key']):x for x in detail};used=set();examples=[]
    selectors=[('Correct current extraction harmed by recovery',lambda r,d:r['key'][1:3]==['v8','component'] and r['confusion'][0]>0 and r['confusion'][1:]==[0,0] and d['category_counts'].get('correct_reducer_state_harmed_by_recovery')),
      ('Unknown abstention does not ensure state preservation',lambda r,d:r['key'][1:3]==['v8','rollout'] and r['unknown_turn'] and r['unknown_no_updates'] and not r['unknown_preserved']),
      ('Ignored unknown markers are distinct from actionable facts',lambda r,d:r['key'][1:3]==['openai','component'] and r['unknown_turn'] and r['unknown_no_updates'] and r['ignored_status_updates'] and r['unknown_preserved']),
      ('Exact extraction discrepancy requiring human semantic review',lambda r,d:r['key'][1:3]==['openai','component'] and d['category_counts'].get('same_slot_exact_value_discrepancy')),
      ('Locally correct extraction inherits earlier rollout errors',lambda r,d:r['key'][1:3]==['openai','rollout'] and r['confusion'][0]>0 and r['confusion'][1:]==[0,0] and d['prior_discrepant_slots'] and not r['complete_pipeline_state']),
      ('Structural admission is separate from readable facts',lambda r,d:r['key'][1]=='v8' and not r['raw_contract_admitted'])]
    md=['# Six trace-backed worked examples','', 'Raw excerpts below are illustrative subsets, not repaired responses. Full raw text, priors, pre/post states and traces are in worked-examples.json. Taxonomy is post-hoc; semantic acceptability remains a human-review question.','']
    for title,predicate in selectors:
        found=next((r for r in rows if tuple(r['key']) not in used and predicate(r,detail_by_key[tuple(r['key'])])),None)
        assert found is not None,title
        used.add(tuple(found['key']));d=detail_by_key[tuple(found['key'])];case,turn=turns[tuple(found['key'][3:])]
        full={'title':title,'key':found['key'],'message':turn['message'],'raw_response':found['raw_output'],'raw_response_sha256':__import__('hashlib').sha256(found['raw_output'].encode()).hexdigest(),'gold_extraction':turn['gold_extraction'],'raw_confusion':found['confusion'],'actual_prior':priors[tuple(found['key'][1:])],**{k:d[k] for k in ['pre_recovery_state','post_recovery_state','expected_state','recovery_harmed_slots','recovery_helped_slots','prior_discrepant_slots','category_counts']},'provenance':found['provenance']}
        examples.append(full);obj=json.loads(found['raw_output']);updates=obj.get('updates',[])
        focus=d['post_reducer_changed_slots'] or [u['slot_id'] for u in updates[:2]]
        delta={s:{'before':d['pre_recovery_state'].get(s),'after':d['post_recovery_state'].get(s),'expected':d['expected_state'].get(s)} for s in focus[:2]}
        excerpt={'intent':obj.get('intent'),'correction_detected':obj.get('correction_detected'),'updates':updates[:2]}
        md+=['## '+str(len(examples))+'. '+title,'',f"Source: `{found['key'][3]}` / `{found['key'][4]}`; {found['key'][1]}, {found['key'][2]}. Raw TP/FP/FN: {found['confusion']}.",'', 'Message: '+turn['message'][:450], '', 'Raw excerpt: `'+dump(excerpt)+'`','', 'Pre/post-recovery comparison: `'+dump(delta)+'`','', 'Trace recovery slots: `'+dump(found['provenance']['recovered_slot_ids'])+'`. See the corresponding full machine-readable example for complete context and trace.','']
    write(out/'worked-examples.json',examples);(out/'WORKED_EXAMPLES.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    renderer=[{'key':r['key'],'failures':r['provenance'].get('static_question_renderer_failures',[])} for r in rows if r['provenance'].get('static_question_renderer_failures')]
    write(out/'renderer_failures.json',{'versioned_wrapper_fix':'R4.1 bounded only static-question length validation; inputs/scoring unchanged; question text removed before next input','affected_main_calls':len(renderer),'calls':renderer})
    summary={'posthoc':True,'taxonomy_categories_overlap':True,'counts_are_not_a_partition_of_FP_FN':True,'per_call_diagnostics':404,'oracle_mismatch_calls':20,'unknown_extraction_state':unknown,'worked_examples':6,'renderer_failure_calls':len(renderer),'independent_human_review_complete':False,'all_published_metrics_unchanged':True}
    write(out/'error_analysis_summary.json',summary);print(dump(summary))


if __name__=='__main__':main()
