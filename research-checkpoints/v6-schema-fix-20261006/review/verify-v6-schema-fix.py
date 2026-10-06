"""Independent post-run artifact, matched-record and failure verification."""
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

REPO=Path(r'D:\SLM\SLM-FineTuning-Testing')
RUN=Path(r'D:\SLM\FYP-model-runs\v6-schema-fix-20261006')
SOURCE=RUN/'study-results'
OUT=RUN/'independent-review'
REL='research-checkpoints/v6-schema-fix-20261006/results'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

publication=read(RUN/'publication-status.json')
assert publication['status']=='published'
commit=publication['commit']
completion=read(SOURCE/'completion.json')
assert completion['status']=='complete' and completion['scored_calls']==384
assert completion['training_started'] is False and completion['sealed_final_accessed'] is False
OUT.mkdir(exist_ok=True)
assert not any(OUT.iterdir()), 'Do not overwrite an existing review'
hashes=read(SOURCE/'artifact-hashes.json');checks={}
for name,want in hashes.items():
    data=(SOURCE/name).read_bytes()
    assert sha(data)==want['sha256'] and len(data)==want['bytes'],name
    blob=subprocess.check_output(['git','show',commit+':'+REL+'/'+name],cwd=REPO)
    assert blob==data,name
    checks[name]=want
reports={}; repeat={};gold_checks={}
protocol=read(REPO/'evaluation/v6-schema-fix-20261006/protocol.json')
for track in ('natural_component','natural_rollout'):
    for arm in ('revised','schema_fixed'):
        name=arm+'-'+track
        raw=gzip.decompress((SOURCE/(name+'.json.gz')).read_bytes())
        assert raw==(RUN/(name+'.json')).read_bytes(),name
        report=json.loads(raw); records=report['records']
        assert report['status']=='complete' and len(records)==96
        assert report['fingerprint']==completion['fingerprint']
        keys=[tuple(r['key']) for r in records]; assert len(set(keys))==96
        assert collections.Counter(r['evaluation_domain'] for r in records)==dict.fromkeys(('weather','inflation','stocks','crypto'),24)
        assert all(r['retry_count'] in (None,0) and r['model_calls'] in (None,1) for r in records)
        reports[name]=records
    left=reports['revised-'+track];right=reports['schema_fixed-'+track]
    assert [r['key'] for r in left]==[r['key'] for r in right]
    assert all(l['expected']==r['expected'] and l['gold_sha256']==r['gold_sha256'] for l,r in zip(left,right))
    if track=='natural_component':assert all(l['context']==r['context'] for l,r in zip(left,right))
    gold_checks[track]={'matched_turn_keys':96,'identical_gold_records':96,'identical_component_contexts':96 if track=='natural_component' else None}
    for arm in ('original','revised'):
        reference=protocol['historical_references'][arm+'-'+track]
        payload=(REPO/reference['path']).read_bytes();assert sha(payload)==reference['sha256']
        old=json.loads(gzip.decompress(payload))['records']
        assert [r['key'] for r in old]==[r['key'] for r in left]
        assert all(r['expected']==l['expected'] for r,l in zip(old,left))
        if arm=='revised':
            counts={field:sum(o[field]==n[field] for o,n in zip(old,left)) for field in ('raw_output','context','expected','prediction_valid')}
            repeat[track]=counts

adapter=Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter')
adapter_checks={}
for name,want in protocol['v6_adapter_sha256'].items():
    actual=sha((adapter/name).read_bytes());assert actual==want
    adapter_checks[name]=actual

outer={'intent','intent_confidence','updates','correction_detected','unsupported_claims'}
fields={'slot_id','candidate_value','status','confidence','evidence_text'}
failures={}
for name,records in reports.items():
    counter=collections.Counter();details=[]
    for r in records:
        if r['prediction_valid']:continue
        issues=[]
        if not r['raw_json_valid']:issues.append('strict_json_failure')
        else:
            value=r['raw_value']
            if not isinstance(value,dict):issues.append('non_object')
            else:
                extra=set(value)-outer;missing=outer-set(value)
                if extra:issues.append('extra_outer_fields:'+','.join(sorted(extra)))
                if missing:issues.append('missing_outer_fields:'+','.join(sorted(missing)))
                updates=value.get('updates',[])
                if isinstance(updates,list):
                    extras=set();absences=set()
                    for u in updates:
                        if isinstance(u,dict):extras.update(set(u)-fields);absences.update(fields-set(u))
                        else:issues.append('non_object_update')
                    if extras:issues.append('extra_update_fields:'+','.join(sorted(extras)))
                    if absences:issues.append('missing_update_fields:'+','.join(sorted(absences)))
                else:issues.append('updates_not_list')
        if r['duplicate_slot_ids']:issues.append('duplicate_slot_ids')
        if not issues:issues.append('other_contract_failure_not_adjudicated')
        counter.update(issues)
        details.append({'key':r['key'],'domain':r['evaluation_domain'],'kind':r['test_kind'],'issues':issues,'completion_tokens':r['completion_tokens'],'hit_generation_limit':r['hit_generation_limit']})
    failures[name]={'invalid_calls':len(details),'issue_counts':dict(counter),'calls':details}
write('independent-verification.json',{'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'published_commit':commit,'verified_artifacts':checks,'lossless_reports':4,'total_records':384,'matching':gold_checks,'revised_control_repeat':repeat,'post_inference_adapter_sha256':adapter_checks,'code_sha256':sha(Path(__file__).read_bytes()),'record_metadata_limit':'Optional model_calls and retry_count are null in all 384 records; no measured retry-count claim from these fields. The pinned runner executes one extraction per turn without a repair/retry loop.','result':'All assertions passed; historical original latency is not a same-session comparison.'})
write('failure-review.json',{'method':'Post hoc structural classification of every schema-invalid call; categories can overlap. No repair, label change or rescoring. Semantic human review remains pending. Strict JSON failure includes duplicate JSON object keys.','reports':failures})
print(json.dumps({'control_repeat':repeat,'failures':{k:{x:v[x] for x in ('invalid_calls','issue_counts')} for k,v in failures.items()}},indent=2))
