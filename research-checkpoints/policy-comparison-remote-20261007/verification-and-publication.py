"""Verify completed local-only handoff runs, compare them, package and publish."""
from __future__ import annotations
import collections
import csv
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import urllib.request

BUNDLE=Path(r'D:\SLM\policy-comparison-20261007')
REPO=Path(r'D:\SLM\SLM-FineTuning-Testing')
LOG=Path(r'D:\SLM\FYP-model-runs\reviewed-policy-comparison-20261007')
OUT=LOG/'verified-results'
CHECKOUT=Path(r'D:\SLM\SLM-reviewed-policy-publication-20261007')
REL='research-checkpoints/policy-comparison-remote-20261007'
BRANCH='experiment/v5-grounded-20260919'
ZIP_NAME='reviewed-policy-gpu-results-20261007.zip'
FOLDERS={'v6':'policy-v6-bf16-20261007','base':'policy-stock-bf16-20261007','v5':'policy-v5-bf16-20261007'}
ENV=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
sys.path.insert(0,str(BUNDLE));sys.path.insert(0,str(BUNDLE/'runtime/fpy-v6-pinned/src'))
from local_slm_lab.comparison_policy import ComparisonPolicy
from local_slm_lab.v5_eval import digest,score_records,sha256,build_call_plan,tuples,dumps,prf
from forecasting_assistant.domain.schema import load_schema

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def git(*args,cwd=REPO):return subprocess.check_output(['git',*args],cwd=cwd,env=ENV,text=True,stderr=subprocess.STDOUT).strip()
def status(value):
    temp=LOG/'publication-status.json.writing';write(temp,value);os.replace(temp,LOG/'publication-status.json')

def supplementary(records,kinds):
    """Descriptive diagnostics from saved records; primary metrics stay untouched."""
    invented=redundant=correction_values=correction_states=unknown_empty=unknown_retained=0
    local_horizon=local_horizon_preserved=0
    name_counts=[0,0,0];value_counts=[0,0,0]
    for r in records:
        gold={k:(v,s) for k,v,s in tuples(r['expected']) if k!='intent'}
        observed={k:(v,s) for k,v,s in tuples(r.get('emitted')) if k!='intent'}
        prior=r['context']['state']['slots']
        for key,(value,_) in observed.items():
            if key not in gold:
                if key in prior and prior[key]['value'] is not None and value==dumps(prior[key]['value']):redundant+=1
                else:invented+=1
        valid=bool(r['prediction_valid'])
        overlap=len(gold.keys() & observed.keys()) if valid else 0
        same=sum(k in observed and observed[k][0]==v[0] for k,v in gold.items()) if valid else 0
        for i,n in enumerate((overlap,len(observed)-overlap,len(gold)-overlap)):name_counts[i]+=n
        for i,n in enumerate((same,len(observed)-same,len(gold)-same)):value_counts[i]+=n
        if r['expected']['correction_detected']:
            flag=valid and r['predicted']['correction_detected'] is True
            correction_values+=bool(flag and {k:v[0] for k,v in gold.items()}=={k:v[0] for k,v in observed.items()})
            correction_states+=bool(flag and r['transition_correct'])
            after=r.get('predicted_after') or {}
            if flag and after.get('slots',{}).get('forecast_horizon')==r['gold_after']['slots']['forecast_horizon']:
                local_horizon+=1
                local_horizon_preserved+=bool(after.get('intent')==r['gold_after']['intent'] and all(
                    after['slots'].get(k)==v for k,v in prior.items() if k not in {'forecast_horizon','intent'}))
        if kinds[tuple(r['key'])]=='unknown':
            unknown_empty+=bool(valid and not observed)
            after=r.get('predicted_after') or {}
            unknown_retained+=bool(valid and after.get('slots')==prior and after.get('intent')==r['context']['state']['intent'])
    return {'new_unmentioned_slot_count':invented,'repeated_prior_slots':redundant,
      'correction_flag_values':correction_values,'full_state_after_correction':correction_states,
      'unknown_no_updates_valid':unknown_empty,'unknown_retains_prior_valid':unknown_retained,
      'slot_name_f1':prf(*name_counts)['f1'],'slot_value_f1_without_status':prf(*value_counts)['f1'],
      'local_horizon_correct':local_horizon,'local_horizon_preserves_other_prior':local_horizon_preserved}

def run():
    policy=ComparisonPolicy();schema=load_schema()
    handoff=read(BUNDLE/'handoff-manifest.json')
    inputs={}
    for name,want in handoff['files'].items():
        p=BUNDLE/name;assert sha256(p)==want['sha256'] and p.stat().st_size==want['bytes'],name
        inputs[name]=want
    assert handoff['policy_sha256']==policy.fingerprint
    cases=[json.loads(s) for s in (policy.directory/'primary-natural.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(cases)==31
    plan=build_call_plan(cases,schema);assert len(plan)==93
    kinds={tuple(c['key']):case['turns'][c['key'][2]-1]['test_kind'] for case in cases for c in build_call_plan([case],schema)}
    OUT.mkdir(exist_ok=False)
    reports={};files={};checks={};metrics_rows=[];domain_rows=[]
    spec=importlib.util.spec_from_file_location('policy_compare',BUNDLE/'scripts/compare-policy-comparison.py')
    compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    for arm,folder in FOLDERS.items():
        root=BUNDLE/'training-runs'/folder;completion=read(root/'completion.json');manifest=read(root/'run-manifest.json')
        assert completion['status']=='complete' and completion['sealed_final_accessed'] is False
        assert completion['budget'] is None and manifest['identity']['arm']==arm
        assert manifest['identity']['representation']=='CUDA BF16 PEFT'
        assert manifest['identity']['protocol']['base_model']=='Qwen/Qwen3.5-2B'
        assert manifest['identity']['protocol']['base_revision']=='15852e8c16360a2fea060d615a32b45270f8a8fc'
        assert manifest['identity']['generation']['max_tokens']==1024
        if arm=='base':assert manifest['identity']['adapter'] is None
        assert manifest['fingerprint']==digest(manifest['identity'])==completion['fingerprint']
        runtime=read(root/'runtime.json');assert runtime['dtype']=='bfloat16' and 'A4000' in runtime['gpu']
        assert not list((root/'jobs').glob('*.partial.json'))
        selected=[root/name for name in ('completion.json','run-manifest.json','runtime.json','component-report.json','rollout-report.json')]
        jobs=sorted((root/'jobs').glob('[cr]*-[0-9][0-9][0-9].json'));assert len(jobs)==62
        selected+=jobs
        for path in selected:
            name=path.relative_to(BUNDLE).as_posix()
            files[name]=path
        for track in ('component','rollout'):
            report=read(root/(track+'-report.json'));records=report['records']
            assert report['status']=='complete' and len(records)==93
            assert report['fingerprint']==manifest['fingerprint']
            assert report['arm']==arm and report['track']==track
            assert len({tuple(r['key']) for r in records})==93
            assert report['api_cost_usd']==0
            assert all(r['policy_fingerprint']==policy.fingerprint and r['usage_cost_usd']==0 for r in records)
            for gold,r in zip(plan,records):
                assert gold['key']==r['key'] and gold['expected']==r['expected']
                assert digest(r['context'])==r['context_sha256']
                assert r['context']['message']==gold['context']['message']
                if track=='component':assert r['context']==gold['context']
                admission=policy.parse(r['raw_output'],'extract',hit_generation_limit=(r['completion_tokens'] or 0)>=1024)
                assert admission.admitted==r['parser_admitted']
            job_records=[]
            for path in sorted((root/'jobs').glob(track+'-[0-9][0-9][0-9].json')):
                job=read(path);assert job['status']=='complete' and job['fingerprint']==manifest['fingerprint']
                assert digest(job['records'])==job['records_sha256'];job_records+=job['records']
            assert job_records==records
            assert score_records(records,[s.slot_id for s in schema.slots])==report['metrics']
            reports[(arm,track)]=report
            m=report['metrics'];extra=supplementary(records,kinds);lat=sorted(r['latency_ms'] for r in records)
            counts=collections.Counter((r.get('error') or {}).get('type') for r in records if not r['prediction_valid'])
            domain_counts=collections.Counter(r['scenario_id'].split('/')[-2] for r in records)
            checks[arm+'/'+track]={'calls':93,'unique_keys':93,'job_records_equal_report':True,'metrics_recomputed_exactly':True,'admission_reparsed_exactly':True,'domains':dict(domain_counts),'invalid_call_errors':dict(counts)}
            metrics_rows.append({'arm':arm,'track':track,'calls':93,'strict_f1':m['nonintent']['f1'],
                'slot_name_f1':extra['slot_name_f1'],'slot_value_f1_without_status':extra['slot_value_f1_without_status'],
                'json_valid':m['raw_json_valid']['numerator'],'declared_schema_valid':m['raw_schema_valid']['numerator'],
                'parser_admitted':sum(r['parser_admitted'] for r in records),'prediction_valid':sum(r['prediction_valid'] for r in records),
                'new_unmentioned_slots':extra['new_unmentioned_slot_count'],
                'correction_flag_values':extra['correction_flag_values'],
                'full_state_after_correction':extra['full_state_after_correction'],
                'local_horizon_correct':extra['local_horizon_correct'],'local_horizon_preserves_other_prior':extra['local_horizon_preserves_other_prior'],
                'unknown_no_updates_valid':extra['unknown_no_updates_valid'],
                'unknown_retains_prior_valid':extra['unknown_retains_prior_valid'],
                'latency_mean_s':m['latency_ms']['mean']/1000,'latency_median_s':m['latency_ms']['p50']/1000,
                'latency_p95_s':m['latency_ms']['p95']/1000})
            for domain in ('weather','inflation','stocks','crypto'):
                subset=[r for r in records if r['scenario_id'].split('/')[-2]==domain]
                dm=score_records(subset,[s.slot_id for s in schema.slots]);de=supplementary(subset,kinds)
                domain_rows.append({'arm':arm,'track':track,'domain':domain,'calls':len(subset),
                  'strict_f1':dm['nonintent']['f1'],'slot_name_f1':de['slot_name_f1'],
                  'slot_value_f1_without_status':de['slot_value_f1_without_status'],
                  'json_valid':dm['raw_json_valid']['numerator'],'declared_schema_valid':dm['raw_schema_valid']['numerator'],
                  'parser_admitted':sum(r['parser_admitted'] for r in subset),'new_unmentioned_slots':de['new_unmentioned_slot_count'],
                  'correction_flag_values':de['correction_flag_values'],'full_state_after_correction':de['full_state_after_correction'],
                  'local_horizon_correct':de['local_horizon_correct'],'local_horizon_preserves_other_prior':de['local_horizon_preserves_other_prior'],
                  'unknown_no_updates_valid':de['unknown_no_updates_valid'],'unknown_retains_prior_valid':de['unknown_retains_prior_valid'],
                  'latency_mean_s':sum(r['latency_ms'] for r in subset)/len(subset)/1000})
    comparisons={}
    for left,right in (('base','v6'),('v5','v6'),('base','v5')):
        value=compare.compare(BUNDLE/'training-runs'/FOLDERS[left],BUNDLE/'training-runs'/FOLDERS[right])
        name=left+'-vs-'+right+'.json';write(OUT/name,value);comparisons[name]=value
    adapter_checks={}
    for arm,adapter in [('v6',Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter')),
                        ('v5',Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v5-anydesk-20261001T141017Z\full\best-adapter'))]:
        actual={name:sha256(adapter/name) for name in ('adapter_model.safetensors','adapter_config.json')}
        assert actual==read(BUNDLE/'training-runs'/FOLDERS[arm]/'run-manifest.json')['identity']['adapter']
        adapter_checks[arm]=actual
    verification={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'handoff_git_commit':'7d1b5b53016570c2b67ae5e9ef677aa720fa0a5b',
        'handoff_zip_sha256':'b6667d9a2467a4f58724e67165a601e9420fe6d38d70d3a9939591135aa0bf99',
        'policy_sha256':policy.fingerprint,'verified_bundle_payload_files':73,'total_scored_calls':558,'reports':checks,
        'run_file_sha256':{name:sha256(p) for name,p in files.items()},'post_run_adapter_sha256':adapter_checks,
        'package_versions':{p:importlib.metadata.version(p) for p in ('pydantic','jsonschema','numpy','torch','transformers','peft')},
        'dependencies_isolated':True,'paid_apis_used':False,'training_started':False,'sealed_final_accessed':False,
        'verification_code_sha256':sha256(Path(__file__)),'status':'all assertions passed'}
    write(OUT/'independent-verification.json',verification)
    with (OUT/'summary.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics_rows[0]));w.writeheader();w.writerows(metrics_rows)
    with (OUT/'domain-summary.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(domain_rows[0]));w.writeheader();w.writerows(domain_rows)
    lines=['# Fresh reviewed-policy GPU comparison','',
      'All three local GPU arms completed: stock 2B, v5 and v6, with 31 reviewed development conversations / 93 calls per component and rollout track. Total: 558 scored extractions. The exact transferred ZIP was verified and executed without altering its frozen parser, prompts, labels or sources. No training, paid APIs, sealed final labels or historical response-cache reuse.','',
      'This is a new experiment under policy `'+policy.version+'` (`'+policy.fingerprint+'`). Ordinary candidate text and JSON-encoded candidate text are admitted according to the same pinned parser. One ambiguous weather conversation is excluded and later gold contexts changed after review. These scores cannot replace or be directly compared with the earlier 32-case historical studies.','',
      '| Track | Arm | Exact slot/value/status F1 | JSON / declared schema / admitted | Unmentioned slots | Raw correction flag+values | Full state after correction | Unknown valid/no updates | Mean / median / p95 seconds |',
      '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for track in ('component','rollout'):
        for arm in ('base','v5','v6'):
            r=next(r for r in metrics_rows if r['arm']==arm and r['track']==track)
            lines.append(f'| {track} | {arm} | {r["strict_f1"]:.4f} | {r["json_valid"]}/93 / {r["declared_schema_valid"]}/93 / {r["parser_admitted"]}/93 | {r["new_unmentioned_slots"]} | {r["correction_flag_values"]}/31 | {r["full_state_after_correction"]}/31 | {r["unknown_no_updates_valid"]}/31 | {r["latency_mean_s"]:.3f} / {r["latency_median_s"]:.3f} / {r["latency_p95_s"]:.3f} |')
    lines+=['','Raw typed corrections and matching the complete accumulated state are different measurements. Component uses reviewed gold prior state; rollout uses each arm’s own accepted prior extraction. Unknown abstention alone does not establish state completeness. Unmentioned-slot flags are relative to reviewed labels and remain subject to human semantic adjudication. summary.csv also separates slot-name F1 from slot-value F1 without status; domain-summary.csv gives every weather/inflation/stocks/crypto subgroup. These supplemental summaries were added during inference, do not alter the frozen primary scorer, and are descriptive diagnostics.','',
      '| Pair | Track | F1 difference, right minus left | 95% scenario-bootstrap interval |',
      '| --- | --- | ---: | --- |']
    for name,value in comparisons.items():
        for track,d in value['comparisons'].items():
            lo,hi=d['ci95'];lines.append(f'| {d["left_arm"]} → {d["right_arm"]} | {track} | {d["delta_right_minus_left"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines+=['','Intervals use 10,000 paired resamples of the 31 source-conversation clusters. They are exploratory, with no correction for multiple comparisons, and do not establish general superiority or unseen-user quality.','',
      'The runs were serial, v6 then stock then v5, rather than interleaved in one model session. All used the same RTX A4000, BF16, pinned base revision, offline cache, greedy decoding, four torch threads and one unscored warmup. Cross-run response-time differences can include hardware/session drift and changed output length; do not interpret them as a controlled latency advantage. Latency summaries retain the frozen scorer conventions, including nearest-rank p95. They measure synchronized extraction, exclude loading/warmup, and do not represent an entire live conversation or external data fetching.','',
      'The hosted OpenAI arm was not run on this PC and its raw reports are not in this archive. The main PC must use its fresh completed reports with compare-policy-comparison.py to verify policy, gold and component-context matching before making any GPU-versus-OpenAI comparison. Do not compare the historical v6 0.5503 score directly with the fresh hosted score.','',
      'This is agent-reviewed posthoc development material. Independent human adjudication and unseen-case validation remain pending. No model or production default was changed.','',
      'The results ZIP includes completion.json, run-manifest.json, runtime.json, both raw reports and all 62 completed jobs per arm. It excludes weights, training checkpoints, credentials, caches and dependency packages. See summary.csv, pairwise comparison JSON files and independent-verification.json for the supporting measurements and checks.','']
    lines+=['A rejected wire response can contain readable facts; low operational F1 must not be interpreted as absence of semantic knowledge. Failed/unparseable outputs are not certified free of invented requirements.','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    # Only explicit completed report files and verification enter this archive.
    zip_path=OUT/ZIP_NAME
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        extras=['independent-verification.json','README.md','summary.csv','domain-summary.csv',*comparisons.keys()]
        for name in extras:z.write(OUT/name,name)
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None and len(z.namelist())==len(files)+len(extras)
        for name,p in files.items():assert z.read(name)==p.read_bytes()
        for name in extras:assert z.read(name)==(OUT/name).read_bytes()
    checksum=sha256(zip_path)
    (OUT/(ZIP_NAME+'.sha256')).write_text(checksum+'  '+ZIP_NAME+'\n',encoding='ascii',newline='\n')
    shutil.copyfile(Path(__file__),OUT/'verification-and-publication.py')
    write(OUT/'artifact-hashes.json',{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file()})
    status({'status':'publishing_verified_completed_results','total_calls':558,'zip_sha256':checksum})
    git('fetch','origin',BRANCH);assert not CHECKOUT.exists()
    git('worktree','add','--detach',str(CHECKOUT),'origin/'+BRANCH)
    destination=CHECKOUT/REL;assert not destination.exists();shutil.copytree(OUT,destination)
    attrs=CHECKOUT/'.gitattributes';s=attrs.read_text();rule='/'+REL+'/** -text whitespace=trailing-space,space-before-tab,cr-at-eol'
    if rule not in s:attrs.write_text(s.rstrip()+'\n'+rule+'\n',encoding='utf-8',newline='\n')
    git('add','-f',REL,cwd=CHECKOUT);git('add','.gitattributes',cwd=CHECKOUT)
    git('diff','--cached','--check',cwd=CHECKOUT)
    git('commit','-m','Publish completed fresh reviewed-policy stock v5 and v6 GPU results',cwd=CHECKOUT)
    for p in OUT.iterdir():
        if p.is_file():
            blob=subprocess.check_output(['git','show','HEAD:'+REL+'/'+p.name],cwd=CHECKOUT,env=ENV)
            assert hashlib.sha256(blob).hexdigest()==sha256(p),p.name
    git('push','origin','HEAD:refs/heads/'+BRANCH,cwd=CHECKOUT)
    commit=git('rev-parse','HEAD',cwd=CHECKOUT)
    assert git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]==commit
    download_url='https://raw.githubusercontent.com/AbdullahUsman0/SLM-FineTuning-Testing/'+commit+'/'+REL+'/'+ZIP_NAME
    with urllib.request.urlopen(download_url,timeout=60) as response:
        downloaded=response.read()
    assert hashlib.sha256(downloaded).hexdigest()==checksum,'Published GitHub download checksum mismatch'
    # Preserve any existing Downloads artifact: use an exclusive copy.
    downloads=Path(os.environ['USERPROFILE'])/'Downloads'
    if downloads.is_dir():
        for name in (ZIP_NAME,ZIP_NAME+'.sha256'):
            destination=downloads/name
            if not destination.exists():
                with destination.open('xb') as f:f.write((OUT/name).read_bytes())
    status({'status':'published','commit':commit,'total_calls':558,'zip_sha256':checksum,
      'report_url':'https://github.com/AbdullahUsman0/SLM-FineTuning-Testing/blob/'+commit+'/'+REL+'/README.md',
      'download_url':download_url,'github_download_checksum_verified':True,
      'paid_apis_used':False,'training_started':False})
    print('Published completed verified GPU results:',commit,'ZIP SHA-256:',checksum,flush=True)

if __name__=='__main__':
    try:run()
    except Exception as error:
        status({'status':'verification_or_publication_failed','error_type':type(error).__name__,'local_results':str(OUT),'training_started':False,'paid_apis_used':False})
        print('Result packaging/publication stopped:',type(error).__name__,flush=True)
        raise
