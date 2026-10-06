"""Separate unreadable output from omissions in a supplementary failure audit.

This descriptive refinement never changes a prediction, gold label or score.
It was added during the frozen two-prompt run; it is not a primary metric.
"""
from collections import Counter
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('audit',ROOT/'scripts/audit-v6-natural-failures.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)

def assess(records,schema):
    rows=[]
    for r in records:
        gold={u[0]:u for u in audit.tuples(r['expected']) if u[0]!='intent'}
        emitted={}
        for u in audit.tuples(r.get('emitted')):
            if u[0]!='intent':emitted.setdefault(u[0],[]).append(u)
        for slot in sorted(gold.keys()|emitted.keys()):
            g=gold.get(slot);predictions=emitted.get(slot,[])
            if r.get('raw_json_valid') is not True:
                category='unassessable_strict_json_failure'
            elif len(predictions)>1:
                category='duplicate_slot_emissions'
            elif g is not None and not predictions:
                category='absent_from_readable_emissions'
            elif g is None:
                prior=r['context']['state']['slots'].get(slot)
                category='repeated_prior_fact' if prior and prior['value'] is not None and predictions[0][1]==audit.dumps(prior['value']) else 'new_unmentioned_slot'
            elif g==predictions[0]:
                category='exact_readable_content'
            elif audit.normalized_tuple(g,schema)==audit.normalized_tuple(predictions[0],schema):
                category='normalization_only'
            elif g[1]==predictions[0][1]:
                category='wrong_status'
            else:
                category='different_value_or_type_not_adjudicated_semantically'
            rows.append({'arm':r['arm'],'track':r['track'],'domain':r['evaluation_domain'],
                         'scenario_id':r['scenario_id'],'turn':r['key'][2],'kind':r['test_kind'],'slot':slot,
                         'category':category,'json_valid':r.get('raw_json_valid'),'prediction_valid':r['prediction_valid'],
                         'expected_value':g[1] if g else None,'observed_values':json.dumps([p[1] for p in predictions]),
                         'message':r['context']['message']})
    return rows

def analyze(source,output):
    output.mkdir(parents=True,exist_ok=False);schema=audit.load_schema();rows=[];hashes={}
    for p in sorted(source.glob('*-natural_*.json.gz')):
        payload=p.read_bytes();hashes[p.name]=audit.sha256(p)
        report=json.loads(gzip.decompress(payload))
        assert report['status']=='complete'
        rows.extend(assess(report['records'],schema))
    assert rows
    with (output/'content-assessment.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries={}
    for track in sorted({r['track'] for r in rows}):
        summaries[track]={}
        for arm in sorted({r['arm'] for r in rows}):
            group=[r for r in rows if r['track']==track and r['arm']==arm]
            if not group:continue
            summaries[track][arm]={
                'categories':dict(Counter(r['category'] for r in group)),
                'readable_omissions_by_slot':dict(Counter(r['slot'] for r in group if r['category']=='absent_from_readable_emissions')),
                'categories_by_turn_kind':{kind:dict(Counter(r['category'] for r in group if r['kind']==kind)) for kind in ('initial','correction','unknown')}}
    result={'created_utc':audit.now(),'source_report_sha256':hashes,'code_sha256':audit.sha256(Path(__file__)),
            'method':'Supplementary post hoc structural failure audit. Invalid strict JSON makes content unassessable; repeated slot IDs are reported separately. Different free-text values are not independently adjudicated as semantic errors. No outputs are repaired, applied or rescored.',
            'tracks':summaries}
    audit.write_json(output/'content-assessment.json',result)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':analyze(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
