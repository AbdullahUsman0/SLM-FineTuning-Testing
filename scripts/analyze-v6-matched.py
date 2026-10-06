"""Audit and summarize the completed matched stock/v5/v6 development study."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import gzip
import itertools
import json
from pathlib import Path
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_slm_lab.v5_eval import confusion, dumps, prf, rate, score_records, sha256, tuples

ARMS=('base','v5','v6')
TRACKS=('natural_component','natural_rollout','frozen_component')

def extra_metrics(records):
    extraction=[r for r in records if r['task']=='extract']
    slot_counts=[0,0,0]; value_counts=[0,0,0]
    invented=redundant=wrong=evidence_bad=0
    unknown=[]; invented_calls=0
    for r in extraction:
        gold={k:(v,status) for k,v,status in tuples(r['expected']) if k!='intent'}
        observed={k:(v,status) for k,v,status in tuples(r.get('emitted')) if k!='intent'}
        valid=bool(r.get('prediction_valid'))
        overlap=len(set(gold)&set(observed)) if valid else 0
        for i,n in enumerate((overlap,len(observed)-overlap,len(gold)-overlap)):slot_counts[i]+=n
        same=sum(k in observed and observed[k][0]==v[0] for k,v in gold.items()) if valid else 0
        for i,n in enumerate((same,len(observed)-same,len(gold)-same)):value_counts[i]+=n
        prior=r['context']['state']['slots']
        new=0
        for key,(value,status) in observed.items():
            if key in gold:
                wrong+=value!=gold[key][0]
            elif key in prior and prior[key]['value'] is not None and value==dumps(prior[key]['value']):
                redundant+=1
            else:
                invented+=1;new+=1
        invented_calls+=new>0
        raw=r.get('raw_value') or {}
        for update in raw.get('updates',[]) if isinstance(raw,dict) and isinstance(raw.get('updates'),list) else []:
            if isinstance(update,dict):
                text=update.get('evidence_text')
                evidence_bad+=not(isinstance(text,str) and text.strip() and text.casefold() in r['context']['message'].casefold())
        if r.get('test_kind')=='unknown':
            before=r['context']['state']['slots']; after=r.get('predicted_after',{}).get('slots')
            retained=bool(after is not None and before==after)
            unknown.append({'valid':valid,'no_updates':valid and not observed,
                            'forbidden':bool(set(observed)&set(r['forbidden_slots'])),
                            'retains_prior_state':valid and retained})
    return {'slot_name_prf':prf(*slot_counts),'slot_value_prf_without_status':prf(*value_counts),
            'new_unmentioned_slot_count':invented,'new_unmentioned_slot_call_rate':rate(invented_calls,len(extraction)),
            'repeated_unchanged_prior_slot_count':redundant,'wrong_values_for_stated_slots':wrong,
            'unsupported_evidence_update_count':evidence_bad,
            'unknown_no_updates_and_valid':rate(sum(u['no_updates'] for u in unknown),len(unknown)),
            'unknown_retains_prior_state_and_valid':rate(sum(u['retains_prior_state'] for u in unknown),len(unknown)),
            'unknown_forbidden_observed':rate(sum(u['forbidden'] for u in unknown),len(unknown))}

def summarize(records):
    metrics=score_records(records)
    metrics.update(extra_metrics(records))
    metrics['extract_latency_ms']=score_records([r for r in records if r['task']=='extract'])['latency_ms']
    metrics['ask_latency_ms']=score_records([r for r in records if r['task']=='ask'])['latency_ms']
    return metrics

def bootstrap(left,right,samples=10000,seed=42):
    import numpy as np
    lm={tuple(r['key']):r for r in left}; rm={tuple(r['key']):r for r in right}
    assert lm.keys()==rm.keys()
    groups=defaultdict(lambda:[[0,0,0],[0,0,0]])
    for key,l in lm.items():
        r=rm[key];assert l['gold_sha256']==r['gold_sha256'] and l['evaluation_domain']==r['evaluation_domain']
        if l['track']!='natural_rollout': assert l['context_sha256']==r['context_sha256']
        if l['task']!='extract':continue
        for arm,row in enumerate((l,r)):
            for i,n in enumerate(confusion(row,nonintent=True)):groups[row['cluster_id']][arm][i]+=n
    counts=np.array(list(groups.values()),dtype=float);rng=np.random.default_rng(seed);deltas=[]
    for start in range(0,samples,500):
        ix=rng.integers(0,len(counts),size=(min(500,samples-start),len(counts)))
        sums=counts[ix].sum(axis=1);den=2*sums[:,:,0]+sums[:,:,1]+sums[:,:,2]
        f=np.divide(2*sums[:,:,0],den,out=np.zeros_like(den),where=den!=0)
        deltas.extend(f[:,1]-f[:,0])
    return {'samples':samples,'seed':seed,'cluster_count':len(groups),
            'unit':'frozen source entity group; authored conversation for natural tracks',
            'delta_f1_ci95':[float(x) for x in np.quantile(deltas,[.025,.975])],
            'rollout_contexts_allowed_to_differ':left[0]['track']=='natural_rollout'}

def analyze(run,output):
    completion=json.loads((run/'completion.json').read_text())
    assert completion['status']=='complete' and completion['scored_calls']==2469
    output.mkdir(parents=True,exist_ok=False)
    reports={};result={'completion':completion,'tracks':{},'paired_f1_bootstrap':{}}
    errors=[]; summary_rows=[]
    for track in TRACKS:
        result['tracks'][track]={}
        for arm in ARMS:
            path=run/f'{arm}-{track}.json';report=json.loads(path.read_text())
            assert report['status']=='complete' and report['fingerprint']==completion['fingerprint']
            records=report['records'];reports[(arm,track)]=records
            metrics={'all':summarize(records)}
            for domain in ('weather','inflation','stocks','crypto'):
                metrics[domain]=summarize([r for r in records if r['evaluation_domain']==domain])
            result['tracks'][track][arm]=metrics
            for domain,m in metrics.items():
                summary_rows.append({'track':track,'arm':arm,'domain':domain,'calls':m['calls']['total'],
                    'slot_value_status_f1':m['nonintent']['f1'],'slot_name_f1':m['slot_name_prf']['f1'],
                    'slot_value_f1':m['slot_value_prf_without_status']['f1'],
                    'json_valid':m['raw_json_valid']['rate'],'schema_valid':m['raw_schema_valid']['rate'],
                    'new_unmentioned_slots':m['new_unmentioned_slot_count'],
                    'wrong_stated_values':m['wrong_values_for_stated_slots'],
                    'corrections_exact':m['correction_tuple_accuracy']['rate'],
                    'unknown_no_updates_valid':m['unknown_no_updates_and_valid']['rate'],
                    'extract_latency_mean_ms':m['extract_latency_ms']['mean'],
                    'extract_latency_p50_ms':m['extract_latency_ms']['p50'],
                    'extract_latency_p95_ms':m['extract_latency_ms']['p95']})
            for r in records:
                if r['task']!='extract':continue
                tp,fp,fn=confusion(r,nonintent=True)
                if not r['prediction_valid'] or fp or fn or r.get('test_kind') in ('correction','unknown'):
                    errors.append({'arm':arm,'track':track,'domain':r['evaluation_domain'],'scenario_id':r['scenario_id'],
                        'turn':r['key'][2],'test_kind':r.get('test_kind'),'valid':r['prediction_valid'],
                        'tp':tp,'fp':fp,'fn':fn,'latency_ms':r['latency_ms'],'message':r['context']['message'],
                        'expected':json.dumps(r['expected'],ensure_ascii=False),'raw_output':r.get('raw_output')})
            payload=path.read_bytes()
            with (output/(path.name+'.gz')).open('xb') as stream:
                with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as compressed:compressed.write(payload)
            assert gzip.decompress((output/(path.name+'.gz')).read_bytes())==payload
        for left,right in itertools.combinations(ARMS,2):
            result['paired_f1_bootstrap'][f'{track}:{right}-{left}']=bootstrap(reports[(left,track)],reports[(right,track)])
    for filename,rows in [('summary.csv',summary_rows),('errors-and-boundary-cases.csv',errors)]:
        with (output/filename).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    # Preserve all source/weight/decoding attestations, without copying model weights.
    for name in ('run-manifest.json','completion.json'):
        (output/name).write_bytes((run/name).read_bytes())
    text=['# Matched stock 2B / v5 / v6 evaluation','',
          'All 2,469 scored calls completed on the same RTX A4000, BF16 CUDA, pinned Qwen3.5-2B revision, greedy decoding and 1,024-token cap. Stock adapters were disabled and checked against native stock output. Arms rotated per scenario; serial latency follows per-arm warmup. No JSON repairs, retries, training, paid APIs or sealed final labels.', '',
          'The frozen component cohort contains all 150 v6 validation scenarios in the requested domains (90 weather, 15 inflation, 15 stocks, 30 crypto). Its language is synthetic and template-based. The 32 natural conversations (eight per domain) were authored before inference, with initial facts, an explicit horizon correction, and unknown information. They are exploratory diagnostics, pending independent human review. Natural component calls use gold prior state; rollout calls use the model’s own state with a fixed user script. These are not live-user trials.', '',
          'F1 requires exact slot, JSON-decoded value and status. Free-text values follow the exact wording required by the prompt. Intent is scored separately. New unmentioned slot emissions are separated from repeated prior facts and wrong values of stated slots. Invalid outputs fail extraction and cannot certify unknown-information handling. JSON syntax validity and full schema validity are separate. Timing excludes model load; question timing is separate in analysis.json.', '',
          '| Track | Model | Exact F1 | JSON valid | Schema valid | New unmentioned slots | Corrections exact | Unknown valid/no updates | Extract median / p95 (s) |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    fmt=lambda value: '—' if value is None else f'{value:.3f}'
    for track in TRACKS:
        for arm in ARMS:
            m=result['tracks'][track][arm]['all'];l=m['extract_latency_ms']
            text.append(f"| {track} | {arm} | {fmt(m['nonintent']['f1'])} | {fmt(m['raw_json_valid']['rate'])} | {fmt(m['raw_schema_valid']['rate'])} | {m['new_unmentioned_slot_count']} | {fmt(m['correction_tuple_accuracy']['rate'])} | {fmt(m['unknown_no_updates_and_valid']['rate'])} | {fmt(l['p50']/1000)} / {fmt(l['p95']/1000)} |")
    text+=['','See summary.csv for every domain, analysis.json for full metrics and paired 10,000-resample cluster bootstrap intervals, and errors-and-boundary-cases.csv for raw predictions beside expected labels. The nine compressed reports preserve every model response and its context losslessly. Unequal domain sizes require per-domain interpretation; aggregate F1 is micro-weighted. Bootstrap intervals address sampling variability in this cohort, not annotation bias or real-user generalization.', '',
           'No new fine-tuning round has been started. Review the natural conversation failures and independent annotations before choosing whether to change the dataset, prompts, or training. Teacher-forced training loss is not used as evidence of behavioral success.','']
    (output/'README.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')
    hashes={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()}
    (output/'artifact-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('\n'.join(text),flush=True)

if __name__=='__main__':analyze(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
