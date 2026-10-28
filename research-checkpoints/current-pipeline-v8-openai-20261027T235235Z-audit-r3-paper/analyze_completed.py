"""Read-only supplementary reporting of prespecified endpoints after completion.

No inference, prompt changes, label changes, checkpoint choice or primary rescoring.
"""
from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import sys

RUN=Path(__file__).resolve().parent
CONFIG=json.loads((RUN/'supervisor-config.json').read_bytes())
sys.path.insert(0,CONFIG['code'])
from controls import atomic,digest,dumps,sha
import pipeline as p
p.configure(RUN/'fpy',Path(CONFIG['repo']))
from forecasting_assistant.application.state_reducer import _canonical_evidence


def main(result):
    receipt=json.loads((RUN/'main-completion.json').read_bytes())
    assert receipt['status']=='complete'
    protocol=json.loads((RUN/'protocol.json').read_bytes())
    assert receipt['protocol_sha256']==sha(RUN/'protocol.json')
    records=[json.loads(line) for line in (RUN/'main-scored-records.jsonl').read_bytes().splitlines()]
    journal=[json.loads(line) for line in (RUN/'calls.jsonl').read_bytes().splitlines()]
    cases=json.loads((RUN/'main-cases.json').read_bytes())
    turns={(c['scenario_id'],t['turn_id']):(c,t) for c in cases for t in c['turns']}
    assert len(records)==404 and len(turns)==101
    summary=json.loads((RUN/'main-summary.json').read_bytes())
    supplement={'scope':'Read-only reporting of correction flags, grounding, final-conversation state and usage; primary scores and admission unchanged','primary_summary_sha256':sha(RUN/'main-summary.json'),'script_sha256':sha(__file__),'independent_human_review_complete':False,'confidence_intervals_implemented':False,'models':{}}
    paper=[]
    for arm in ('v8','openai'):
        supplement['models'][arm]={}
        for track in ('component','rollout'):
            rows=[r for r in records if r['key'][1:3]==[arm,track]]
            assert len(rows)==101
            assert p.prf(rows)==summary[arm][track]['raw_exact']
            flags_correct=flags_false_positive=flags_false_negative=structurally_invalid=intent_correct=0
            rejection_reasons=Counter()
            unsupported=[];evidence_eligible=ignored_status=0;recovery_changes=[]
            for row in rows:
                case,turn=turns[tuple(row['key'][3:])]
                if row['raw_contract_admitted']:
                    obj=json.loads(row['raw_output'])
                    intent_correct+=obj['intent']==turn['gold_extraction']['intent']
                    flag=obj['correction_detected'];expected=turn['gold_extraction']['correction_detected']
                    flags_correct+=flag==expected
                    flags_false_positive+=flag and not expected
                    flags_false_negative+=expected and not flag
                    decoded=p.admit(row['raw_output'],protocol['response_schema'])
                    for update in decoded.updates:
                        if update.status.value in ('unmentioned','dont_care'):
                            ignored_status+=1;continue
                        evidence_eligible+=1
                        if _canonical_evidence(update,turn['message'],turn['message'].casefold()) is None:
                            unsupported.append({'key':row['key'],'slot_id':update.slot_id})
                else:
                    structurally_invalid+=1
                    if row.get('capped'):rejection_reasons['generation_cap']+=1
                    elif row.get('refused'):rejection_reasons['refusal']+=1
                    elif not row['json_valid']:rejection_reasons['outer_JSON_invalid_or_duplicate_field']+=1
                    elif not row['schema_valid']:rejection_reasons['declared_schema_invalid']+=1
                    else:
                        obj=json.loads(row['raw_output'])
                        slot_ids=[u['slot_id'] for u in obj.get('updates',[])]
                        if len(slot_ids)!=len(set(slot_ids)):rejection_reasons['duplicate_slot_ID']+=1
                        else:rejection_reasons['inner_candidate_JSON_or_Pydantic_contract']+=1
                before=row['reducer_state_before_recovery'];after=row['pipeline_state']
                changed=[sid for sid in set(before)|set(after) if before.get(sid)!=after.get(sid)]
                if changed:
                    recovery_changes.append({'key':row['key'],'changed_slots':sorted(changed),'reported_recovery_slots':row['provenance']['recovered_slot_ids']})
            by_key={tuple(r['key'][3:]):r for r in rows}
            final_correct=all_correct=0
            for case in cases:
                final=by_key[(case['scenario_id'],case['turns'][-1]['turn_id'])]
                final_correct+=final['complete_pipeline_state']
                all_correct+=all(by_key[(case['scenario_id'],t['turn_id'])]['complete_pipeline_state'] for t in case['turns'])
            detail={'calls':101,'raw_correction_flag_correct_all_calls':flags_correct,'raw_correction_flag_false_positive':flags_false_positive,'raw_correction_flag_false_negative':flags_false_negative,'structurally_invalid_calls_in_flag_denominator':structurally_invalid,'eligible_update_evidence_count_including_intent':evidence_eligible,'unsupported_evidence_update_count':len(unsupported),'unsupported_evidence_updates':unsupported,'ignored_unmentioned_or_dontcare_updates':ignored_status,'exact_final_pipeline_state_conversations':final_correct,'all_turns_exact_pipeline_state_conversations':all_correct,'conversation_denominator':20,'pipeline_vs_reducer_state_changes':recovery_changes}
            detail.update(raw_intent_correct_all_calls=intent_correct,raw_contract_rejection_reasons=dict(rejection_reasons))
            supplement['models'][arm][track]=detail
            m=summary[arm][track];exact=m['raw_exact']
            paper.append({'model':arm,'track':track,'calls':101,'raw_exact_precision':exact['precision'],'raw_exact_recall':exact['recall'],'raw_exact_F1':exact['f1'],'json_valid':m['json_valid'],'schema_valid':m['schema_valid'],'raw_contract_admitted':m['raw_contract_admitted'],'reducer_admitted_without_recovery':m['reducer_admitted_without_recovery'],'extra_slot_flags':m['extra_slot_flags'],'forbidden_slot_flags':m['forbidden_slot_flags'],'raw_correction_flag_correct_all_calls':flags_correct,'correct_raw_correction_updates':m['correct_raw_corrections'],'correction_calls':m['correction_calls'],'raw_unknown_omissions':m['raw_unknown_omissions'],'unknown_turns':m['unknown_turns'],'pipeline_unknown_preserved':m['pipeline_unknown_preserved'],'exact_pipeline_state_turns':m['exact_pipeline_state_turns'],'exact_final_pipeline_state_conversations':final_correct,'conversation_denominator':20,'recovery_calls':m['recovery_calls'],'median_extraction_seconds':m['median_seconds']})
    ledger_path=Path(protocol['shared_budget_ledger'])
    db=sqlite3.connect('file:'+ledger_path.as_posix()+'?mode=ro',uri=True)
    all_ledger=db.execute('SELECT key,bound,actual,state,response_sha,usage FROM calls').fetchall();db.close()
    ledger=[]
    for row in all_ledger:
        scoped_key=json.loads(row[0])
        if len(scoped_key)==6 and scoped_key[0]==RUN.name:
            ledger.append((dumps(scoped_key[1:]),*row[1:]))
    hosted=[r for r in journal if r['key'][1]=='openai']
    assert len(hosted)==len(ledger)==220
    assert {tuple(json.loads(row[0])) for row in ledger}=={tuple(r['key']) for r in hosted}
    assert all(row[3]=='reconciled' for row in ledger)
    usage_by_key={tuple(r['key']):r['usage'] for r in hosted};by_scope={}
    total=Decimal(0);guard=Decimal(0)
    for key,bound,actual,state,response_sha,stored_usage in ledger:
        key=json.loads(key);usage=json.loads(stored_usage)
        assert usage==usage_by_key[tuple(key)]
        response_path=RUN/'hosted-responses'/(digest(key)+'.json')
        assert sha(response_path)==response_sha
        input_tokens=usage['input_tokens'];output_tokens=usage['output_tokens'];cached=(usage.get('input_tokens_details') or {}).get('cached_tokens',0);reasoning=(usage.get('output_tokens_details') or {}).get('reasoning_tokens',0)
        assert 0<=cached<=input_tokens and 0<=reasoning<=output_tokens
        # Reasoning is included in output_tokens, not added to it again.
        estimate=(Decimal(input_tokens-cached)*Decimal('.25')+Decimal(cached)*Decimal('.025')+Decimal(output_tokens)*Decimal('2'))/Decimal(1_000_000)
        accounted=(Decimal(input_tokens)*Decimal('.25')+Decimal(output_tokens)*Decimal('2'))/Decimal(1_000_000)
        assert int(accounted*Decimal(1_000_000_000))==actual and actual<=bound
        total+=estimate;guard+=accounted
        scope='/'.join(key[:3]);row=by_scope.setdefault(scope,{'calls':0,'input_tokens':0,'cached_input_tokens':0,'output_tokens_including_reasoning':0,'reasoning_tokens':0,'usage_rate_estimate_USD':Decimal(0),'conservative_ledger_accounted_USD':Decimal(0)})
        row['calls']+=1;row['input_tokens']+=input_tokens;row['cached_input_tokens']+=cached;row['output_tokens_including_reasoning']+=output_tokens;row['reasoning_tokens']+=reasoning;row['usage_rate_estimate_USD']+=estimate;row['conservative_ledger_accounted_USD']+=accounted
    for row in by_scope.values():
        row['usage_rate_estimate_USD']=str(row['usage_rate_estimate_USD']);row['conservative_ledger_accounted_USD']=str(row['conservative_ledger_accounted_USD'])
    billing={'scope':'Entire authorized comparison: both pilot tracks and both main tracks; changing key did not reset budget','completed_hosted_requests':220,'usage_rate_estimate_USD':str(total),'conservative_ledger_accounted_USD':str(guard),'reserved_USD':0,'uncertain_USD':0,'user_ceiling_USD':1,'internal_dispatch_ceiling_USD':0.9,'reasoning_charged_once':True,'no_cache_discount_used_to_allow_dispatch':True,'usage_estimate_is_not_provider_invoice':True,'all_response_SHA_and_ledger_usage_reconciled':True,'by_scope':by_scope}
    baseline=json.loads((RUN/'budget-at-start.json').read_bytes())
    cumulative_guard=Decimal(sum(row[2] or 0 for row in all_ledger))/Decimal(1_000_000_000)
    cumulative_priced=Decimal(0)
    for row in all_ledger:
        assert row[3]=='reconciled'
        u=json.loads(row[5]);cache=(u.get('input_tokens_details') or {}).get('cached_tokens',0)
        cumulative_priced+=(Decimal(u['input_tokens']-cache)*Decimal('.25')+Decimal(cache)*Decimal('.025')+Decimal(u['output_tokens'])*Decimal('2'))/Decimal(1_000_000)
    assert cumulative_guard-Decimal(str(baseline['actual_USD']))==guard
    billing.update(current_revision_hosted_requests=len(ledger),cumulative_task_hosted_requests=len(all_ledger),cumulative_task_usage_rate_estimate_USD=str(cumulative_priced),cumulative_task_conservative_ledger_USD=str(cumulative_guard),prior_spending_carried_forward_USD=str(baseline['actual_USD']),revision_namespace=RUN.name)
    atomic(result/'supplementary-contract-audit.json',supplement)
    atomic(result/'billing-usage-analysis.json',billing)
    with (result/'PAPER_UPDATE_TABLE.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paper[0]));writer.writeheader();writer.writerows(paper)
    return {'billing':billing,'paper_rows':paper}


if __name__=='__main__':
    parser=__import__('argparse').ArgumentParser();parser.add_argument('--result',type=Path,required=True);args=parser.parse_args();print(dumps(main(args.result)))
