"""Matched serial BF16 stock/v5/v6 evaluation; no repairs, retries or final access.

One pinned base with two independently attested adapters. Stock disables LoRA;
its output is checked against the native base before any scored call. Arm order
rotates per scenario to reduce timing drift. Durable scenario reports support
resume without changing the frozen inputs or discarding failed calls.
"""
from __future__ import annotations
import argparse
import asyncio
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_slm_lab.v5_eval import (
    build_call_plan, digest, evaluate, load_cases, load_schema, now,
    provenance, runtime_metadata, score_records, sha256, state_snapshot,
)
from local_slm_lab.peft_provider import PeftInferenceConfig, load_peft_adapter
from local_slm_lab.v5_provider import V5TransformersProvider
from forecasting_assistant.application.state_reducer import apply_extraction
from forecasting_assistant.domain.models import DialogueTurn, ExtractorResult
from forecasting_assistant.domain.schema import create_initial_state

V5=Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v5-anydesk-20261001T141017Z\full\best-adapter')
V6=Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter')
ADAPTER_HASHES={
    'v5':{'adapter_model.safetensors':'e97f5e9808713a1daaaa105b3251b96f2f273834eca0e4e6da53989c652d7080',
          'adapter_config.json':'fd6f87135e9a032d0bdbeed093a379d5f3871d4796888df27dcacb032be3ac50'},
    'v6':{'adapter_model.safetensors':'e7add7b8b02d24615ed89067c7761c3985a4a6afb7c84ca2a72e3b06fd84e74d',
          'adapter_config.json':'38f60de9678324e91fa882ed86c70d09e4a39b27ae28884374ddc3e6fece15ef'},
}

def atomic(path,value):
    temporary=path.with_name(path.name+'.writing')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)

def rollout_call(gold,state):
    call=dict(gold)
    call['state']=state.model_copy(deep=True)
    call['context']={'message':gold['context']['message'],'state':state_snapshot(state)}
    # Expected labels and final state remain the independently frozen gold.
    return call

def advance_rollout(state,record,schema,index):
    if record.get('prediction_valid'):
        try:
            state=apply_extraction(state,ExtractorResult.model_validate(record['predicted']),schema,index,record['context']['message'])
        except Exception:
            pass # Failed transitions preserve prior state; no fabricated recovery.
    state.turns.append(DialogueTurn(turn_number=len(state.turns)+1,user_message=record['context']['message']))
    return state

class SynchronizedProvider(V5TransformersProvider):
    async def extract(self,message,state):
        self._torch.cuda.synchronize()
        try:
            return await super().extract(message,state)
        finally:
            self._torch.cuda.synchronize()

    async def ask(self,request):
        self._torch.cuda.synchronize()
        try:
            return await super().ask(request)
        finally:
            self._torch.cuda.synchronize()

def run(args):
    study=args.study.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    # Explicitly locked entire GPU study; OS releases this lock after a crash.
    import msvcrt
    with (out.parent/'gpu-training.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1): lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        return run_locked(args,study,out)

def run_locked(args,study,out):
    manifest=json.loads((study/'study-manifest.json').read_text())
    assert manifest['version']=='v6-matched-20261005-1' and manifest['sealed_final_accessed'] is False
    cases={}; plans={}
    schema=load_schema()
    for name in ('frozen','natural'):
        path=study/(name+'.jsonl')
        assert sha256(path)==manifest['cases'][name]['sha256']
        cases[name],_=load_cases(path,split='validation')
        plans[name]=build_call_plan(cases[name],schema)
        assert digest([c['key'] for c in plans[name]])==manifest['cases'][name]['ordered_keys_sha256']
        assert digest([[c['key'],digest(c['context']),digest(c['expected'])] for c in plans[name]])==manifest['cases'][name]['context_gold_sha256']
    # The smoke file is an unscored subset of development validation scenarios.
    warmup_cases,_=load_cases(ROOT/'corpus-v5/v5-20260919-r1/splits/smoke.jsonl',split='validation')
    warmup_plan=build_call_plan(warmup_cases[:1],schema)[:1]
    for arm,path in [('v5',V5),('v6',V6)]:
        for name,expected in ADAPTER_HASHES[arm].items():
            if sha256(path/name)!=expected: raise ValueError(f'{arm} adapter hash mismatch')
    snapshot=Path(os.environ.get('HF_HOME',r'D:\SLM\hf-cache'))/'hub/models--Qwen--Qwen3.5-2B/snapshots'/manifest['base_revision']
    if sha256(snapshot/'model.safetensors-00001-of-00001.safetensors')!='aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1':
        raise ValueError('Pinned base-model weight hash mismatch')
    audit=provenance('base')
    identity={'study':sha256(study/'study-manifest.json'),'adapters':ADAPTER_HASHES,
              'audit':audit,'runner':sha256(Path(__file__)),'torch_threads':4,
              'schema':digest(schema.model_dump(mode='json')),'batch_size':1}
    fingerprint=digest(identity)
    record_manifest=out/'run-manifest.json'
    if record_manifest.exists():
        saved=json.loads(record_manifest.read_text())
        assert saved['fingerprint']==fingerprint,'Run code/inputs/runtime changed; cannot resume'
    else:
        atomic(record_manifest,{'fingerprint':fingerprint,'identity':identity,'started_at':now()})
    if (out/'completion.json').exists():
        return json.loads((out/'completion.json').read_text())
    import torch
    from peft import PeftModel
    torch.set_num_threads(4)
    config=PeftInferenceConfig(base_model=manifest['base_model'],revision=manifest['base_revision'],
        variant='base',device='cuda',dtype='bfloat16',max_new_tokens=1024)
    started=perf_counter(); provider=SynchronizedProvider(config,schema)
    native_load_ms=(perf_counter()-started)*1000
    native=asyncio.run(evaluate(provider,warmup_cases[:1],schema=schema,plan=warmup_plan))
    provider._model=load_peft_adapter(PeftModel,provider._model,V5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        keys=provider._model.load_adapter(str(V6),adapter_name='v6',is_trainable=False)
    if any('missing adapter keys' in str(w.message).lower() for w in caught):
        raise RuntimeError('v6 adapter keys did not load')
    if any('lora' in k.lower() and '.v6.' in k for k in getattr(keys,'missing_keys',[])):
        raise RuntimeError('Missing v6 LoRA weights')
    if any(p.device.type!='cuda' for p in provider._model.parameters()):
        raise RuntimeError('Matched study must not offload')
    warmups={}
    for arm in ('base','v5','v6'):
        if arm!='base': provider._model.set_adapter('default' if arm=='v5' else 'v6')
        provider._model.eval()
        with provider._model.disable_adapter() if arm=='base' else nullcontext():
            warmups[arm]=asyncio.run(evaluate(provider,warmup_cases[:1],schema=schema,plan=warmup_plan))
    assert native['records'][0]['raw_output']==warmups['base']['records'][0]['raw_output'], 'Disabled-adapter stock differs from native base'
    session={'started_at':now(),'native_load_ms':native_load_ms,'runtime':runtime_metadata(provider),
             'native_vs_disabled_stock_raw_output_equal':True,'warmups':warmups,
             'stock_implementation':'same pinned base, PEFT adapters disabled; native output equivalence checked',
             'torch_num_threads':torch.get_num_threads(),'inference_batch_size':1}
    session_dir=out/'sessions'; session_dir.mkdir(exist_ok=True)
    atomic(session_dir/(now().replace(':','-')+'.json'),session)
    print('READY: verified native/disabled stock equivalence; all arms warmed up.',flush=True)
    jobs=[]
    for track,source in [('natural_component','natural'),('natural_rollout','natural'),('frozen_component','frozen')]:
        for index,scenario in enumerate(cases[source]):
            jobs.append((track,source,index,scenario))
    orders=[('base','v5','v6'),('v5','v6','base'),('v6','base','v5')]
    completed_count=0; total_calls=3*(len(plans['frozen'])+2*len(plans['natural']))
    status={'status':'running','total_calls':total_calls,'total_scenario_arm_units':len(jobs)*3,
            'fingerprint':fingerprint,'started_at':now()}
    atomic(out/'status.json',status)
    for job_index,(track,source,index,scenario) in enumerate(jobs):
        sid=scenario['scenario_id']; selected=[c for c in plans[source] if c['scenario_id']==sid]
        for arm in orders[job_index%3]:
            folder=out/arm/track; folder.mkdir(parents=True,exist_ok=True)
            target=folder/f'{index:04d}.json'
            if target.exists():
                old=json.loads(target.read_text())
                assert old['status']=='complete' and old['fingerprint']==fingerprint and old['scenario_id']==sid
                completed_count+=len(old['records'])
                continue
            if arm!='base': provider._model.set_adapter('default' if arm=='v5' else 'v6')
            provider._model.eval()
            provider.config=replace(config,variant='base' if arm=='base' else 'custom',adapter_path=None if arm=='base' else (V5 if arm=='v5' else V6))
            progress={'status':'partial','fingerprint':fingerprint,'scenario_id':sid,'arm':arm,'track':track,'records':[]}
            journal=target.with_suffix('.partial.json')
            if journal.exists():
                interruption=folder/f'{index:04d}.interrupted-{now().replace(":","-")}.json'
                os.replace(journal,interruption)
            state=create_initial_state(schema)
            with provider._model.disable_adapter() if arm=='base' else nullcontext():
                for call_index,gold in enumerate(selected,1):
                    call=rollout_call(gold,state) if track=='natural_rollout' else gold
                    result=asyncio.run(evaluate(provider,[scenario],schema=schema,plan=[call]))
                    record=result['records'][0]
                    record.update(arm=arm,track=track,evaluation_domain=scenario['evaluation_domain'],
                                  test_kind=scenario['turns'][call_index-1].get('test_kind',scenario['category']) if record['task']=='extract' else 'question')
                    if track=='natural_rollout': state=advance_rollout(state,record,schema,call_index)
                    progress['records'].append(record); completed_count+=1
                    atomic(journal,progress)
                    status.update(completed_calls=completed_count,heartbeat_utc=now(),current_arm=arm,
                                  current_track=track,current_scenario=sid,last_latency_ms=record['latency_ms'])
                    atomic(out/'status.json',status)
            progress['status']='complete'; atomic(target,progress); journal.unlink()
            print(f'DONE {job_index+1}/{len(jobs)} {track} {arm} {sid} [{completed_count}/{total_calls} calls]',flush=True)
    assert completed_count==total_calls
    for arm in ('base','v5','v6'):
        for track,source in [('natural_component','natural'),('natural_rollout','natural'),('frozen_component','frozen')]:
            reports=[json.loads(p.read_text()) for p in sorted((out/arm/track).glob('[0-9][0-9][0-9][0-9].json'))]
            records=[r for report in reports for r in report['records']]
            combined={'status':'complete','fingerprint':fingerprint,'arm':arm,'track':track,
                      'records':records,'metrics':score_records(records,[s.slot_id for s in schema.slots]),
                      'metadata':{'study':manifest,'runtime':session['runtime'],'audit':audit}}
            atomic(out/f'{arm}-{track}.json',combined)
    completion={'status':'complete','finished_utc':now(),'fingerprint':fingerprint,'scored_calls':completed_count,
                'arms':['base','v5','v6'],'sealed_final_accessed':False,'training_started':False}
    atomic(out/'completion.json',completion); atomic(out/'status.json',completion)
    return completion

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'evaluation/v6-matched-20261005')
    parser.add_argument('--output',type=Path,required=True)
    try:
        print(json.dumps(run(parser.parse_args()),indent=2),flush=True)
    except BaseException as error:
        print('STOP:',type(error).__name__,str(error),flush=True)
        raise
