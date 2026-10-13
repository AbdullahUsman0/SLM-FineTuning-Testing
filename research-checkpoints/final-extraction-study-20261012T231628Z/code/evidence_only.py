"""Summarize historical public artifacts under their original policy; no model calls."""
from collections import Counter
import csv
import io
import json
from pathlib import Path
import zipfile
import runtime as rt


def main():
    rt.verify_sources()
    checkpoint=rt.CHECKPOINT;repo=checkpoint.parents[1]
    historical=repo/'research-checkpoints/policy-openai-slm-matched-20261008'
    pilot=repo/'research-checkpoints/v7-r8-pilot-20261009/results'
    old=rt.read(historical/'summary.json')
    verified=rt.read(checkpoint/'evidence/historical-matched-verification.json')
    assert verified['status']=='verified' and verified['scored_calls']==744
    rows=[]
    for row in old['summaries']:
        rows.append({'cohort_id':'historical-31-20261008','comparison_group':'same-31-same-original-policy',
            'arm':row['model'],'track':row['track'],'conversations':31,'calls':row['calls'],
            'exact_f1':row['f1'],'precision':row['precision'],'recall':row['recall'],
            'json_valid':row['JSON_valid'],'schema_valid':row['declared_schema_valid'],
            'parser_admitted':row['parser_admitted'],'median_latency_seconds':row['median_latency_seconds'],
            'extra_current_turn_flags':None,'independent_human_review_completed':False,
            'source_checkpoint':'policy-openai-slm-matched-20261008'})
    archive=pilot/'qwen35-2b-lora-v7-r8-pilot-20261010T002204Z-evaluation.zip'
    assert rt.sha(archive)=='a253dc06fe165cbb2c516903e3cd928344ed010407747f51a73a9b494a74bb7a'
    with (pilot/'summary.csv').open(encoding='utf-8',newline='') as stream:
        pilot_rows=list(csv.DictReader(stream))
    diagnostics={'scope':'Existing public pilot; original historical scoring only',
        'raw_archive_sha256':rt.sha(archive),'strata':[],'role_examples':[],'paid_API_calls':0,
        'final_cohort_calls':0,'sealed_final_accessed':False,'not_human_adjudication':True}
    ambiguity={'scope':'Original public pilot rollout; diagnostic only, no changed primary scores',
        'source_ZIP_sha256':rt.sha(archive),'arms':{}}
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        for row in pilot_rows:
            arm,track=row['arm'],row['track']
            report=json.loads(package.read(f'evaluation/{arm}/{track}-report.json'))
            records=report['records']
            measured=rt.score_records(records,[slot.slot_id for slot in rt.load_schema().slots])['nonintent']
            assert measured==report['metrics']['nonintent']
            assert len(records)==1040 and measured['f1']==float(row['strict_f1'])
            rows.append({'cohort_id':'v7-r8-synthetic-pilot-200','comparison_group':'same-200-same-pilot-policy',
                'arm':arm,'track':track,'conversations':200,'calls':1040,'exact_f1':measured['f1'],
                'precision':measured['precision'],'recall':measured['recall'],
                'json_valid':int(row['json_valid']),'schema_valid':int(row['schema_valid']),
                'parser_admitted':int(row['admitted']),'median_latency_seconds':float(row['latency_median_seconds']),
                'extra_current_turn_flags':int(row['unmentioned_slot_flags']),
                'independent_human_review_completed':False,'source_checkpoint':'v7-r8-pilot-20261009/results'})
            if arm not in ('v6','v7') or track!='rollout':continue
            ambiguous_rows=[r for r in records if r['category']=='ambiguous_then_clarified' and r['key'][-1]==1]
            ambiguous_bad=[]
            for record in ambiguous_rows:
                extra=[u for u in (record.get('emitted') or {}).get('updates',[])
                    if u['slot_id'] in {'frequency','prediction_frequency'}]
                if extra:ambiguous_bad.append({'key':record['key'],'message':record['context']['message'],
                    'extra_updates':extra,'parser_admitted':record['parser_admitted']})
            ambiguity['arms'][arm]={'calls':len(ambiguous_rows),'calls_with_extra_cadence_emissions':len(ambiguous_bad),
                'admitted_calls_with_extra_cadence_emissions':sum(x['parser_admitted'] for x in ambiguous_bad),
                'examples':ambiguous_bad}
            for field,values in [('language',['en','ur','mixed']),('category',['lifecycle_dense','cadence_contrast'])]:
                for value in values:
                    selected=[r for r in records if r[field]==value]
                    diagnostics['strata'].append({'arm':arm,'track':track,'field':field,'value':value,
                        'calls':len(selected),'exact':rt.score_records(selected,[s.slot_id for s in rt.load_schema().slots])['nonintent']})
            selected_examples={}
            for record in records:
                expected={u['slot_id'] for u in record['expected']['updates']}
                emitted=(record.get('emitted') or record.get('raw_value') or {}).get('updates',[])
                for update in emitted:
                    if not isinstance(update,dict):continue
                    slot=update.get('slot_id')
                    if slot in expected:continue
                    if slot=='expected_history_length' and 'frequency' in expected:kind='sampling_as_history'
                    elif slot=='source_mode' and 'refresh_frequency' in expected:kind='refresh_as_access_mode'
                    elif record['category']=='ambiguous_then_clarified' and not expected:kind='ambiguous_cadence_update'
                    else:continue
                    if kind not in selected_examples:
                        selected_examples[kind]={'arm':arm,'kind':kind,'key':record['key'],
                            'message':record['context']['message'],'expected_slot_ids':sorted(expected),
                            'extra_update':update,'parser_admitted':record['parser_admitted']}
            diagnostics['role_examples'].extend(selected_examples.values())
    old_ci=rt.read(historical/'openai-vs-v6.json')['comparisons']
    paired={'historical-31-v6-minus-OpenAI':{t:{k:x[k] for k in ('delta_right_minus_left','ci95','samples','seed','cluster_count')}
        for t,x in old_ci.items()},'pilot-200-v7-minus-v6':rt.read(pilot/'paired-comparisons.json')['v6-vs-v7']}
    rt.atomic(checkpoint/'summary.json',{'status':'verified_existing_evidence_only','new_scored_runs':0,
        'new_paid_API_calls':0,'new_paid_budget_usd':0,'rows':rows,'paired_comparisons':paired,
        'cross_cohort_comparison_allowed':False,'fresh_v7_vs_OpenAI_result_available':False,
        'limitation':'Different cohorts and prompts; independent final review not completed. No winner/equivalence claim.'},exclusive=True)
    target=checkpoint/'summary.csv'
    with target.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rt.atomic(checkpoint/'evidence/pilot-offline-diagnostics.json',diagnostics,exclusive=True)
    rt.atomic(checkpoint/'evidence/ambiguous-cadence-diagnostic.json',ambiguity,exclusive=True)
    print('Historical evidence summarized; all six pilot exact metrics and selected strata replayed under original scoring. No model/API calls.')


if __name__=='__main__':main()
