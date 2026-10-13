"""Independently replay all raw journals; completion requires all planned final calls."""
import argparse
import csv
from pathlib import Path
import runtime as rt
from evaluate import initialize,grade,check_binding
from journal import Journal
from metrics import summary,paired,bootstrap,failure_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','freeze','config'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();config=rt.read(args.config)
    rt.verify_sources('verify')
    cases,plans,schema,policy,identity,approval=initialize(args.run,args.freeze,config)
    fingerprint=rt.digest(identity);reports={};expected_all=set();file_hashes={}
    for arm in ('v6','v7','openai'):
        scored=Journal(args.run/'records'/arm,fingerprint);raws=Journal(args.run/'raw'/arm,fingerprint)
        for track in ('component','rollout'):
            records=[]
            for case in sorted(cases,key=lambda c:c['scenario_id']):
                state=rt.create_initial_state(schema)
                for call in plans[case['scenario_id']]:
                    call_id=f"{arm}-{track}-{case['scenario_id']}-{call['index']:02d}";expected_all.add(call_id)
                    stored=scored.load(call_id);raw=raws.load(call_id)
                    if stored is None or raw is None:raise ValueError('Missing expected completed final call')
                    prior=call['before'].model_copy(deep=True) if track=='component' else state.model_copy(deep=True)
                    context_hash=rt.digest({'message':call['turn']['message'],'state':rt.state_snapshot(prior)})
                    check_binding(raw,call_id,context_hash,rt.digest(call['gold']))
                    reproduced,after=grade(raw,case,call,track,arm,prior,policy,schema,
                        from_journal=stored['score_reconstructed_from_journal'])
                    if stored!=reproduced:raise ValueError('Stored score/prior/admission does not reproduce raw response')
                    if track=='rollout':state=after
                    records.append(stored)
                    path=args.run/'records'/arm/(call_id+'.json');file_hashes[path.relative_to(args.run).as_posix()]=rt.sha(path)
            reports[arm,track]=records
    actual={p.stem for p in (args.run/'records').rglob('*.json')}
    if actual!=expected_all:raise ValueError('Unexpected/duplicate final call files')
    ledger=rt.read(args.run/'api-budget-ledger.json')
    exposure=sum(a['actual_cost_usd'] if a.get('actual_cost_usd') is not None else a['reserved_usd'] for a in ledger['attempts'])
    if exposure>approval['cap_usd']+1e-12:raise ValueError('Approved expenditure bound exceeded')
    if ledger['identity']!=fingerprint:raise ValueError('Budget ledger from a different experiment')
    # Every hosted completed response must bind to the ledger before expenditure verification.
    raw_api=Journal(args.run/'raw/openai',fingerprint)
    for path in (args.run/'raw/openai').glob('*.json'):
        raw=raw_api.load(path.stem);attempt=ledger['attempts'][raw['attempt_id']-1]
        if attempt['call_id']!=raw['call_id'] or attempt['actual_cost_usd']!=raw['cost_usd']:
            raise ValueError('API response usage is not reconciled to the budget ledger')
    metrics={(arm,track):summary(records) for (arm,track),records in reports.items()}
    comparisons={}
    for left,right,role in [('v7','openai','primary'),('v6','openai','secondary'),('v7','v6','secondary')]:
        comparisons[left+'-minus-'+right]={'role':role,'tracks':{track:bootstrap(paired(reports[left,track],reports[right,track]))
            for track in ('component','rollout')}}
    out=args.run/'verified-results';out.mkdir(exist_ok=False)
    rt.atomic(out/'summary.json',{'metrics':{arm:{track:metrics[arm,track] for track in ('component','rollout')}
        for arm in ('v6','v7','openai')},'comparisons':comparisons,
        'known_total_API_cost_usd':sum(a['actual_cost_usd'] or 0 for a in ledger['attempts']),
        'API_exposure_upper_bound_usd':exposure,'approved_cap_usd':approval['cap_usd']},exclusive=True)
    with (out/'summary.csv').open('x',encoding='utf-8',newline='') as stream:
        fields=['model','track','calls','exact_F1','precision','recall','JSON_valid','schema_valid','admitted','extra_flags','latency_median_seconds']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for (arm,track),m in metrics.items():
            writer.writerow({'model':arm,'track':track,'calls':m['calls'],'exact_F1':m['exact_nonintent']['f1'],
                'precision':m['exact_nonintent']['precision'],'recall':m['exact_nonintent']['recall'],
                'JSON_valid':m['raw_JSON_valid']['numerator'],'schema_valid':m['declared_schema_valid']['numerator'],
                'admitted':m['primary_parser_admission']['numerator'],'extra_flags':m['extra_current_turn_slot_flags'],
                'latency_median_seconds':m['latency_original_generation_seconds']['median']})
    strata={}
    for (arm,track),records in reports.items():
        strata[arm+'-'+track]={field:{value:summary([r for r in records if r[field]==value])
            for value in sorted({r[field] for r in records})} for field in ('domain','language','category')}
    rt.atomic(out/'subgroup-exploratory.json',strata,exclusive=True)
    for (arm,track),records in reports.items():
        rt.atomic(out/(arm+'-'+track+'-failure-triage.json'),failure_rows(records),exclusive=True)
    interval=comparisons['v7-minus-openai']['tracks']['rollout']['ci95']
    conclusion=('V7 achieved higher primary F1 on this evaluated cohort.' if interval[0]>0 else
        'OpenAI achieved higher primary F1 on this evaluated cohort.' if interval[1]<0 else
        'The primary comparison did not establish a statistically significant difference; this is not equivalence.')
    paper=['# Verified final matched requirement-extraction results','',conclusion,'',
        '| Model | Component exact F1 | Rollout exact F1 |','| --- | ---: | ---: |']
    for arm in ('v6','v7','openai'):paper.append(f"| {arm} | {metrics[arm,'component']['exact_nonintent']['f1']:.4f} | {metrics[arm,'rollout']['exact_nonintent']['f1']:.4f} |")
    paper+=['','Primary interval: '+rt.dumps(comparisons['v7-minus-openai']['tracks']['rollout']),
        '', 'These are independently reviewed synthetic requirements, grouped by 12 authored template families, not independent human-authored requests. Exact requirement extraction is not forecasting accuracy. Full 79-slot schema; frozen adapters and common logical prompt; exact evidence; no repairs or semantic retries. Hosted strict-schema transport and reasoning differ from local greedy generation. Interleaved local blocks reduce but do not eliminate time/session and network confounds. Secondary comparisons and subgroups are exploratory. No noninferiority or equivalence claim. No additional training follows this experiment.']
    (out/'PAPER_RESULTS.md').write_text('\n'.join(paper)+'\n',encoding='utf-8')
    rt.verify_models(config);rt.verify_sources()
    for p in (args.run/'raw').rglob('*.json'):file_hashes[p.relative_to(args.run).as_posix()]=rt.sha(p)
    for p in out.rglob('*'):
        if p.is_file():file_hashes[p.relative_to(args.run).as_posix()]=rt.sha(p)
    completion={'status':'independently_verified_complete','fingerprint':fingerprint,'scored_calls':len(expected_all),
        'expected_scored_calls':identity['planned_calls_per_track']*6,'artifact_sha256':file_hashes,
        'all_failures_in_denominators':True,'all_contexts_and_rollout_priors_replayed':True,
        'known_API_cost_usd':sum(a['actual_cost_usd'] or 0 for a in ledger['attempts']),
        'conservative_API_exposure_usd':exposure,'approved_cap_usd':approval['cap_usd'],
        'previously_sealed_final_accessed':False,'training_started':False,'completed_at':rt.now()}
    rt.atomic(args.run/'completion.json',completion,exclusive=True)
    rt.atomic(args.run/'status.json',{'status':'independently_verified_complete','scored_calls':len(expected_all),'updated_at':rt.now()})
    print(conclusion,flush=True)


if __name__=='__main__':main()
