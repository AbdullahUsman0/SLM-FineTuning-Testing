"""Post-run independent integrity and matched-control audit; no model loading."""
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

REPO=Path(r'D:\SLM\SLM-FineTuning-Testing')
RUN=Path(r'D:\SLM\FYP-model-runs\v6-constraints-20261007')
SOURCE=RUN/'study-results-final'
OUT=RUN/'independent-review'
REL='research-checkpoints/v6-constraints-20261007/results'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

publication=read(RUN/'publication-status.json');assert publication['status']=='published'
commit=publication['commit'];completion=read(SOURCE/'completion.json')
assert completion['status']=='complete' and completion['scored_calls']==384
assert completion['training_started'] is False and completion['sealed_final_accessed'] is False
OUT.mkdir(exist_ok=False)
hashes=read(SOURCE/'artifact-hashes.json');checks={}
for name,want in hashes.items():
    data=(SOURCE/name).read_bytes()
    assert sha(data)==want['sha256'] and len(data)==want['bytes'],name
    blob=subprocess.check_output(['git','show',commit+':'+REL+'/'+name],cwd=REPO)
    assert blob==data,name
    checks[name]=want
reports={};repeat={};matched={}
grammar_checks={}
protocol=read(REPO/'evaluation/v6-constraints-20261007/protocol.json')
assert protocol['prompt_sha256']['original']==protocol['prompt_sha256']['constrained']
assert protocol['compile_options']=={'strict_mode':True,'any_order':False,'any_whitespace':True}
assert sha((REPO/'evaluation/v6-constraints-20261007/extractor-result.schema.json').read_bytes())==protocol['output_schema_sha256']
import xgrammar as xgr
info=xgr.TokenizerInfo(['a','b','<eos>'],xgr.VocabType.RAW,stop_token_ids=[2])
compiled=xgr.GrammarCompiler(info,max_threads=1).compile_json_schema(
    read(REPO/'evaluation/v6-constraints-20261007/extractor-result.schema.json'),**protocol['compile_options'])
for track in ('natural_component','natural_rollout'):
    for arm in ('original','constrained'):
        name=arm+'-'+track
        raw=gzip.decompress((SOURCE/(name+'.json.gz')).read_bytes())
        assert raw==(RUN/(name+'.json')).read_bytes(),name
        report=json.loads(raw);records=report['records']
        assert report['status']=='complete' and len(records)==96
        assert report['fingerprint']==completion['fingerprint']
        assert len({tuple(r['key']) for r in records})==96
        assert collections.Counter(r['evaluation_domain'] for r in records)==dict.fromkeys(('weather','inflation','stocks','crypto'),24)
        assert all(r['retry_count']==0 and r['model_calls']==1 for r in records)
        assert all(r['prompt_sha256']==protocol['prompt_sha256'][arm] for r in records)
        assert all(r['constraint_backend']==('none' if arm=='original' else protocol['backend']) for r in records)
        if arm=='constrained':
            outcomes=collections.Counter()
            for r in records:
                raw_text=r['raw_output']
                if not isinstance(raw_text,str):outcomes['raw_output_unavailable']+=1;continue
                matcher=xgr.GrammarMatcher(compiled)
                accepted=matcher.accept_string(raw_text)
                outcomes['complete_grammar_valid' if accepted and matcher.is_completed() else 'valid_prefix_incomplete' if accepted else 'grammar_invalid']+=1
            grammar_checks[track]=dict(outcomes)
        reports[name]=records
    left=reports['original-'+track];right=reports['constrained-'+track]
    assert [r['key'] for r in left]==[r['key'] for r in right]
    assert all(l['expected']==r['expected'] and l['gold_sha256']==r['gold_sha256'] for l,r in zip(left,right))
    if track=='natural_component':assert all(l['context']==r['context'] for l,r in zip(left,right))
    matched[track]={'matched_keys':96,'identical_gold_records':96,'identical_component_contexts':96 if track=='natural_component' else None}
    reference=protocol['historical_references'][track]
    payload=(REPO/reference['path']).read_bytes();assert sha(payload)==reference['sha256']
    old=json.loads(gzip.decompress(payload))['records']
    assert [r['key'] for r in old]==[r['key'] for r in left]
    assert all(r['expected']==l['expected'] for r,l in zip(old,left))
    repeat[track]={field:sum(o[field]==n[field] for o,n in zip(old,left)) for field in ('raw_output','context','expected','prediction_valid')}

adapter=Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter');weights={}
for name,want in protocol['v6_adapter_sha256'].items():
    actual=sha((adapter/name).read_bytes());assert actual==want;weights[name]=actual

failures={}
outer={'intent','intent_confidence','updates','correction_detected','unsupported_claims'}
fields={'slot_id','candidate_value','status','confidence','evidence_text'}
for name,records in reports.items():
    rows=[];counts=collections.Counter()
    for r in records:
        if r['prediction_valid']:continue
        reasons=[]
        if not r['raw_json_valid']:reasons.append('strict_json_failure')
        else:
            value=r['raw_value']
            if not isinstance(value,dict):reasons.append('non_object')
            else:
                if set(value)!=outer:reasons.append('outer_keys_mismatch')
                for u in value.get('updates',[]):
                    if set(u)!=fields:reasons.append('update_keys_mismatch')
                    try:json.loads(u['candidate_value'])
                    except (ValueError,TypeError,KeyError):reasons.append('candidate_value_not_valid_json_text')
        if r['duplicate_slot_ids']:reasons.append('duplicate_slot_ids')
        if not reasons:reasons.append('other_application_contract_failure')
        reasons=sorted(set(reasons));counts.update(reasons)
        rows.append({'key':r['key'],'domain':r['evaluation_domain'],'kind':r['test_kind'],'reasons':reasons,'traces':r['traces'],'hit_generation_limit':r['hit_generation_limit']})
    failures[name]={'invalid_calls':len(rows),'reason_counts':dict(counts),'calls':rows}
write('independent-verification.json',{'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'published_commit':commit,'artifacts':checks,'lossless_reports':4,'total_records':384,'matching':matched,'control_repeat':repeat,'post_inference_adapter_sha256':weights,'same_prompt_both_arms':True,'every_call_one_generation_no_retries':True,'independent_grammar_checks':grammar_checks,'mask_backend_from_pinned_source':'torch_native','record_metadata_limit':'The common evaluator sanitizes away mask_steps and mask_backend trace extensions. Per-call backend and model-call counts are retained, and output language membership is independently checked; no token-by-token masking-trace claim.','code_sha256':sha(Path(__file__).read_bytes()),'result':'All integrity assertions passed. Grammar-check outcomes are reported separately and do not certify behavioral quality.'})
write('failure-review.json',{'method':'Post hoc structural classification of all application-invalid calls; no repair or label/score changes. Categories can overlap. Semantic human adjudication remains pending.','reports':failures})
print(json.dumps({'control_repeat':repeat,'failures':{k:{x:v[x] for x in ('invalid_calls','reason_counts')} for k,v in failures.items()}},indent=2))
