"""Independently check fresh r8 reports, compare, package and publish results."""
from collections import Counter
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import zipfile
from evaluate_r8 import (BUNDLE, V6, V6_HASHES, advance, atomic, build_plans, digest, now,
                         sha, state_snapshot, verify_freeze, intervene, create_initial_state,
                         apply_extraction, score_records)
from local_slm_lab.v5_eval import confusion, tuples

BRANCH = 'experiment/v5-grounded-20260919'
REPO = Path(r'D:\SLM\SLM-FineTuning-Testing')

def git(folder, *args):
    result = subprocess.run(['git', '-C', str(folder), *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Git operation failed: '+args[0]+': '+result.stderr[-1000:])
    return result.stdout.strip()

def matched(left, right, track):
    indexes = [{tuple(r['key']): r for r in values} for values in (left, right)]
    if any(len(index)!=len(values) for index,values in zip(indexes,(left,right))) or indexes[0].keys()!=indexes[1].keys():
        raise ValueError('Missing or duplicate matched calls')
    pairs = []
    for key in sorted(indexes[0], key=str):
        a,b = (index[key] for index in indexes)
        for field in ('key','scenario_id','cluster_id','expected','gold_sha256','forbidden_slots','policy_fingerprint','behavior','state_intervention'):
            if a[field]!=b[field]:
                raise ValueError('Matched gold or policy differs: '+field)
        if a['context']['message']!=b['context']['message'] or (track=='component' and a['context']!=b['context']):
            raise ValueError('Matched user script or component contexts differ')
        pairs.append((a,b))
    return pairs

def bootstrap(pairs, samples=10000, seed=42):
    import numpy as np
    clusters = sorted({a['cluster_id'] for a,b in pairs})
    counts = np.zeros((len(clusters),2,3),dtype=np.int64)
    index = {name:i for i,name in enumerate(clusters)}
    for a,b in pairs:
        for arm,r in enumerate((a,b)):
            counts[index[a['cluster_id']],arm] += confusion(r,nonintent=True)
    def f1(value):
        tp,fp,fn = value[...,0],value[...,1],value[...,2]
        denominator = 2*tp+fp+fn
        return np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),where=denominator!=0)
    point = f1(counts.sum(axis=0)); rng=np.random.default_rng(seed); deltas=[]
    for start in range(0,samples,256):
        selected=rng.integers(0,len(clusters),size=(min(256,samples-start),len(clusters)))
        scores=f1(counts[selected].sum(axis=1)); deltas.extend(scores[:,1]-scores[:,0])
    return {'left_f1':float(point[0]),'right_f1':float(point[1]),'delta_right_minus_left':float(point[1]-point[0]),
            'ci95':[float(v) for v in np.quantile(deltas,[.025,.975])], 'samples':samples,'seed':seed,
            'cluster_count':len(clusters),'interval':'Paired entity-group percentile bootstrap; exact micro F1 recomputed.'}

def numeric_canonical(value):
    # Same JSON number, including 1 versus 1.0; strings and booleans keep type.
    if type(value) is float and value.is_integer():return int(value)
    if isinstance(value,list):return [numeric_canonical(v) for v in value]
    if isinstance(value,dict):return {k:numeric_canonical(v) for k,v in value.items()}
    return value

def diagnostic_f1(records,mode):
    tp=fp=fn=0
    for r in records:
        def items(value):
            result=[]
            for u in (value or {}).get('updates',[]):
                if u['slot_id']=='intent':continue
                if mode=='names':item=(u['slot_id'],)
                else:
                    number=json.dumps(numeric_canonical(u['candidate_value']),sort_keys=True,ensure_ascii=False,separators=(',',':'))
                    item=(u['slot_id'],number) if mode=='values' else (u['slot_id'],number,u['status'])
                result.append(item)
            return Counter(result)
        gold=items(r['expected']);emitted=items(r.get('emitted'))
        overlap=sum((gold&emitted).values()) if r['prediction_valid'] else 0
        tp+=overlap;fp+=sum(emitted.values())-overlap;fn+=sum(gold.values())-overlap
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0

def summary(records,arm,track):
    metric=score_records(records)
    unknown=[r for r in records if r['behavior']=='unknown_omission']
    corrections=[r for r in records if r['expected']['correction_detected']]
    after=lambda r:r.get('predicted_after',{}).get('slots')
    before=lambda r:r['context']['state']['slots']
    return {'arm':arm,'track':track,'calls':len(records),'strict_f1':metric['nonintent']['f1'],
            'slot_name_f1':diagnostic_f1(records,'names'),
            'numeric_canonical_slot_value_status_f1':diagnostic_f1(records,'values_and_status'),
            'numeric_canonical_slot_value_f1_without_status':diagnostic_f1(records,'values'),
            'json_valid':sum(r['raw_json_valid'] for r in records),'schema_valid':sum(r['raw_schema_valid'] for r in records),
            'admitted':sum(r['parser_admitted'] for r in records),'evidence_grounded_calls':sum(r['evidence_grounded'] for r in records),
            'unmentioned_slot_flags':metric['hallucinated_slot_count'],
            'correction_calls':len(corrections),'correction_flag_and_tuples_correct':sum(r['prediction_valid'] and r['predicted']['correction_detected'] and confusion(r,nonintent=True)[1:]==(0,0) for r in corrections),
            'complete_state_after_correction':sum(r['transition_correct'] for r in corrections),
            'unknown_calls':len(unknown),'unknown_valid_no_updates':sum(r['prediction_valid'] and not r['predicted']['updates'] for r in unknown),
            'unknown_valid_prior_retained':sum(r['prediction_valid'] and not r['predicted']['updates'] and after(r)==before(r) for r in unknown),
            'state_transition_correct_calls':sum(r['transition_correct'] for r in records),
            'latency_mean_seconds':metric['latency_ms']['mean']/1000,
            'latency_median_seconds':metric['latency_ms']['p50']/1000,
            'latency_p95_seconds':metric['latency_ms']['p95']/1000}

def independently_verify(run):
    schema,policy,cases,plans,protocol = verify_freeze(run)
    wanted = [call for c in cases for call in plans[c['scenario_id']]]
    completion=json.loads((run/'completion.json').read_bytes())
    spec=importlib.util.spec_from_file_location('r8_training_verifier',BUNDLE/'training/train_lora.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    train_manifest=trainer.load_manifest(run/'pilot')
    for checkpoint in (run/'pilot').glob('checkpoint-*'):
        trainer.verify_checkpoint(checkpoint,train_manifest['run_fingerprint'])
    if {name:sha(Path(completion['adapter'])/name) for name in V6_HASHES}!=completion['adapter_sha256']:
        raise ValueError('Pilot adapter hash mismatch')
    if {name:sha(V6/name) for name in V6_HASHES}!=V6_HASHES:
        raise ValueError('v6 adapter changed')
    reports={}
    for arm in ('stock','v6','v7'):
        folder=run/'evaluation'/arm
        done=json.loads((folder/'completion.json').read_bytes())
        identity=json.loads((folder/'run-manifest.json').read_bytes())
        if done['status']!='complete' or list((folder/'jobs').glob('*.partial.json')):
            raise ValueError('Evaluation is incomplete')
        expected_adapter=None if arm=='stock' else V6_HASHES if arm=='v6' else completion['adapter_sha256']
        if identity['identity']!={'arm':arm,'protocol_fingerprint':digest(protocol),'adapter_sha256':expected_adapter} or identity['fingerprint']!=digest(identity['identity']):
            raise ValueError('Evaluation model or protocol identity differs')
        for track in ('component','rollout'):
            report=json.loads((folder/f'{track}-report.json').read_bytes())
            records=report['records'];keys=[tuple(r['key']) for r in records]
            if len(keys)!=1040 or len(set(keys))!=1040 or set(keys)!={tuple(c['key']) for c in wanted}:
                raise ValueError('Evaluation call inventory differs')
            if report['protocol_fingerprint']!=digest(protocol) or report['fingerprint']!=identity['fingerprint'] or report['status']!='complete':
                raise ValueError('Evaluation report protocol differs')
            jobs=sorted((folder/'jobs').glob(f'{track}-[0-9][0-9][0-9].json'))
            joined=[]
            if len(jobs)!=200:raise ValueError('Missing completed jobs')
            for jobpath in jobs:
                job=json.loads(jobpath.read_bytes())
                if job['status']!='complete' or job['fingerprint']!=identity['fingerprint'] or job['records_sha256']!=digest(job['records']):
                    raise ValueError('Job records changed')
                joined.extend(job['records'])
            if joined!=records:raise ValueError('Raw report does not reproduce completed jobs')
            indexed={tuple(r['key']):r for r in records}
            for case in cases:
                state=create_initial_state(schema)
                for call in plans[case['scenario_id']]:
                    r=indexed[tuple(call['key'])]
                    if track=='rollout' and call['state_intervention']:intervene(state,call['state_intervention'])
                    prior=call['state'] if track=='component' else state
                    if r['context']!={'message':call['context']['message'],'state':state_snapshot(prior)}:
                        raise ValueError('Component or rollout prior did not reproduce')
                    if r['expected']!=call['expected'] or r['context_sha256']!=digest(r['context']) or r['cluster_id']!=call['cluster_id']:
                        raise ValueError('Expected labels/context/group changed')
                    trace=r['policy_traces'][-1]
                    admission=policy.parse(r['raw_output'],'extract',hit_generation_limit=trace.get('hit_generation_limit',False))
                    if (r['raw_json_valid'],r['raw_schema_valid'],r['parser_admitted'],r['predicted'])!=(admission.json_valid,admission.declared_schema_valid,admission.admitted,admission.predicted):
                        raise ValueError('Admission or decoded prediction differs')
                    if r['prediction_valid']!=(r['provider_success'] and admission.admitted):raise ValueError('Prediction validity differs')
                    if r['usage_cost_usd']!=0:raise ValueError('Unexpected API usage')
                    succeeded=False
                    if r['prediction_valid']:
                        try:
                            after=apply_extraction(prior,admission.to_model('extract'),schema,call['key'][2],call['context']['message'])
                            succeeded=True
                        except Exception as error:
                            if r.get('transition_error',{}).get('type')!=type(error).__name__:raise ValueError('Transition failure differs')
                        if succeeded and (r.get('predicted_after')!=state_snapshot(after) or r['transition_correct']!=(state_snapshot(after)==call['gold_after'])):
                            raise ValueError('Reducer output or correctness differs')
                    if track=='rollout':state=advance(state,r,policy,schema)
            if score_records(records,[s.slot_id for s in schema.slots])!=report['metrics']:
                raise ValueError('Primary metrics did not reproduce')
            reports[(arm,track)]=report
    return reports,protocol

def zip_checked(path,files):
    if path.exists():raise FileExistsError('Do not overwrite a prior artifact')
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,p in files.items():z.write(p,name)
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or set(z.namelist())!=set(files):raise ValueError('ZIP failed CRC/inventory check')
        for name,p in files.items():
            if sha(p)!=hashlib.sha256(z.read(name)).hexdigest():raise ValueError('ZIP member differs')
    checksum=sha(path)
    path.with_name(path.name+'.sha256').write_text(checksum+'  '+path.name+'\n',encoding='utf-8',newline='\n')
    return checksum

def finish(run):
    reports,protocol=independently_verify(run)
    out=run/'continuation-results';out.mkdir()
    rows=[summary(reports[(arm,track)]['records'],arm,track) for track in ('component','rollout') for arm in ('stock','v6','v7')]
    with (out/'summary.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    strata=[]
    for (arm,track),report in reports.items():
        for field in ('domain','language','category'):
            for value in sorted({r[field] for r in report['records']}):
                subset=[r for r in report['records'] if r[field]==value]
                strata.append({'field':field,'value':value,**summary(subset,arm,track)})
        subset=[r for r in report['records'] if r['category']!='explicit_state_repair']
        strata.append({'field':'intervention_scope','value':'exclude_synthetic_repairs',**summary(subset,arm,track)})
    atomic(out/'stratified-summary.json',strata)
    pairs={}
    for left,right in (('stock','v6'),('stock','v7'),('v6','v7')):
        pairs[left+'-vs-'+right]={track:bootstrap(matched(reports[(left,track)]['records'],reports[(right,track)]['records'],track)) for track in ('component','rollout')}
    atomic(out/'paired-comparisons.json',pairs)
    atomic(out/'independent-verification.json',{'status':'passed','scored_calls':6240,'protocol':protocol,
           'all_records_reparsed':True,'all_primary_metrics_reproduced':True,'rollout_priors_replayed':True,
           'checkpoint_and_adapter_hashes_verified':True,'paid_API_calls':0,'sealed_final_accessed':False})
    diff=pairs['v6-vs-v7']['rollout']
    byarm={r['arm']:r for r in rows if r['track']=='rollout'}
    candidate_gate=(diff['ci95'][0]>0 and byarm['v7']['admitted']>=byarm['v6']['admitted'] and byarm['v7']['unmentioned_slot_flags']<=byarm['v6']['unmentioned_slot_flags'])
    atomic(out/'next-round-decision.json',{'status':'awaiting_human_failure_review_and_explicit_next_round_decision',
           'predeclared_quantitative_candidate_gate_passed':candidate_gate,'v6_vs_v7_rollout':diff,
           'automatic_full_training_started':False,'model_quality_generalized_to_unseen_users':False,
           'limitations':['Synthetic template overlap; agent review is not human annotation approval.',
                         '20 shared entity groups, not 200 independent conversations.',
                         'Filename-inference validation sparse for Roman Urdu and absent for mixed.',
                         'Serial GPU runs do not establish a controlled latency advantage.']})
    lines=['# Fresh stock/v6/v7 r8 pilot comparison','','All 6,240 scored extractions completed: 200 identical validation conversations, 1,040 calls per track per arm. All used the v7 prompt and same frozen wire policy. No paid APIs or sealed final labels.','',
           '| Arm | Track | Exact slot/value/status F1 | JSON valid | Admitted | Unmentioned flags | Median seconds |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {r['arm']} | {r['track']} | {r['strict_f1']:.4f} | {r['json_valid']}/1040 | {r['admitted']}/1040 | {r['unmentioned_slot_flags']} | {r['latency_median_seconds']:.2f} |")
    lines+=['',f"v6 to v7 rollout F1 delta: {diff['delta_right_minus_left']:+.4f}; 95% entity-group bootstrap interval [{diff['ci95'][0]:+.4f}, {diff['ci95'][1]:+.4f}].",'',
            'The 20 source groups contain ten related conversations each. Intervals resample groups. These are synthetic development cases with shared templates; they do not establish unseen-user quality. Filename-language coverage is uneven. All failed outputs remain in denominators.', '',
            'Rollout uses accepted predicted state and the original declared synthetic repair interventions. The repair category and rollout excluding repairs are reported separately. Component priors match the actual validation SFT. Correction, unknown, evidence and full-state metrics are in summary.csv; subgroup metrics are in stratified-summary.json. Supplemental numeric-canonical diagnostics treat JSON numbers 1 and 1.0 equally while preserving strings and booleans; the frozen primary exact-JSON tuple scorer and its intervals remain unchanged.', '',
            'This evaluates extraction on fixed user scripts, not generated clarification questions or a live complete product conversation. The interactive V7 provider still requires JSON-encoded candidate strings; the shared comparison policy also admits ordinary candidate text. Benchmark admission therefore does not certify identical interactive-runtime admission.', '',
            'The final-step pilot adapter remains experimental. No larger training round starts automatically. Quantitative criteria, human failure review and an explicit next-round decision are required. Loss alone cannot select the adapter. Latency is serial-arm synchronized extraction, excluding warmup/loading; session drift and output length can affect comparisons.', '',
            'The ZIP includes immutable raw jobs, both raw reports and manifests for every arm, diagnostics, paired comparisons, training completion/provenance and the frozen continuation code. It excludes model weights, optimizer files, credentials and caches. A separate verified adapter package is placed in Downloads for local testing.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    for name in ('completion.json','inputs-manifest.json','user-training-authorization.json','evaluation-freeze.json','behavioral-protocol.json'):
        shutil.copyfile(run/name,out/name)
    training_receipts=out/'training-provenance';training_receipts.mkdir()
    for name in ('run-manifest.json','evaluation.json','events.jsonl'):
        shutil.copyfile(run/'pilot'/name,training_receipts/name)
    adapter=Path(json.loads((run/'completion.json').read_bytes())['adapter'])
    adapter_manifest={'adapter':{name:{'sha256':sha(adapter/name),'bytes':(adapter/name).stat().st_size} for name in V6_HASHES}}
    atomic(out/'adapter-files.json',adapter_manifest)
    adapter_readme=out/'ADAPTER_README.md'
    adapter_readme.write_text('Experimental final-step v7 r8 pilot adapter; behavioral quality on unseen users has not been established.\n\nBase: Qwen/Qwen3.5-2B, revision '+protocol['base_revision']+'. Dataset: v7-20261009-r8-dev, 200-conversation pilot.\n\nDependencies: existing CUDA torch 2.11.0+cu128, transformers 5.15.0, peft 0.20.0, safetensors 0.8.0; pinned r8 application/data dependencies. Extract into a new directory. Use the verified r8 code bundle.\n\nPowerShell (replace the adapter directory with the extracted path):\n```powershell\n$env:HF_HOME="D:\\SLM\\hf-cache"\n$env:SLM_FPY_SRC="D:\\SLM\\v7-approved-policy-20261009-r8-review\\runtime\\fpy-v6-pinned\\src"\n& D:\\SLM\\SLM-FineTuning-Testing\\.venv\\Scripts\\python.exe D:\\SLM\\v7-approved-policy-20261009-r8-review\\scripts\\chat-peft.py --variant custom --adapter .\\best-adapter --adapter-manifest .\\adapter-files.json --base-model Qwen/Qwen3.5-2B --revision '+protocol['base_revision']+' --prompt-version v7 --device cuda --dtype bfloat16 --max-new-tokens 1024\n```\n',encoding='utf-8',newline='\n')
    downloads=Path.home()/'Downloads'
    adapter_zip=run/(run.name+'-adapter.zip')
    zip_checked(adapter_zip,{'best-adapter/'+name:adapter/name for name in V6_HASHES} | {'completion.json':run/'completion.json','adapter-files.json':out/'adapter-files.json','README.md':adapter_readme})
    files={p.relative_to(run).as_posix():p for p in (run/'evaluation').rglob('*.json')}
    files.update({p.relative_to(out).as_posix():p for p in out.rglob('*') if p.is_file()})
    files.update({'continuation-code/'+p.name:p for p in (run/'continuation-code').glob('*.py')})
    archive=run/(run.name+'-evaluation.zip');checksum=zip_checked(archive,files)
    for p in (archive,archive.with_name(archive.name+'.sha256'),adapter_zip,adapter_zip.with_name(adapter_zip.name+'.sha256')):
        dest=downloads/p.name
        if dest.exists():
            if sha(dest)!=sha(p):raise FileExistsError('Existing Downloads file differs')
        else:
            with p.open('rb') as source,dest.open('xb') as target:shutil.copyfileobj(source,target)
    tree=Path(r'D:\SLM')/('SLM-v7-pilot-publication-'+run.name.rsplit('-',1)[-1])
    git(REPO,'fetch','origin',BRANCH)
    git(REPO,'worktree','add','--detach',str(tree),'origin/'+BRANCH)
    relative=Path('research-checkpoints/v7-r8-pilot-20261009/results')
    destination=tree/relative
    if destination.exists():raise FileExistsError('Do not overwrite published results')
    destination.mkdir(parents=True)
    for p in out.iterdir():
        if p.is_file():shutil.copyfile(p,destination/p.name)
    shutil.copytree(training_receipts,destination/'training-provenance')
    for p in (archive,archive.with_name(archive.name+'.sha256')):shutil.copyfile(p,destination/p.name)
    (destination/'.gitattributes').write_bytes(b'* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n')
    git(tree,'add','--',relative.as_posix());git(tree,'diff','--cached','--check')
    git(tree,'commit','-m','Publish matched stock v6 v7 r8 pilot evaluation')
    for attempt in range(3):
        try:
            git(tree,'push','origin','HEAD:'+BRANCH);break
        except RuntimeError:
            if attempt==2:raise
            git(tree,'fetch','origin',BRANCH);git(tree,'rebase','origin/'+BRANCH)
            time.sleep(5)
    commit=git(tree,'rev-parse','HEAD')
    url=f'https://raw.githubusercontent.com/AbdullahUsman0/SLM-FineTuning-Testing/{commit}/{relative.as_posix()}/{archive.name}'
    payload=urllib.request.urlopen(url,timeout=45).read()
    if hashlib.sha256(payload).hexdigest()!=checksum:raise ValueError('Published download checksum differs')
    return {'status':'complete_and_published','commit':commit,'download_url':url,'zip_sha256':checksum,
            'scored_calls':6240,'downloads_adapter_zip':str(downloads/adapter_zip.name),'full_training_started':False,
            'github_download_verified':True,'finished_at':now()}
