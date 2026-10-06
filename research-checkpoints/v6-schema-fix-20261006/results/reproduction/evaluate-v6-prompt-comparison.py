"""One frozen two-prompt comparison with unchanged v6 weights."""
import argparse
import asyncio
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_slm_lab.v5_eval import build_call_plan,digest,evaluate,load_cases,load_schema,now,provenance,runtime_metadata,sha256
from local_slm_lab.peft_provider import PeftInferenceConfig
from local_slm_lab.v5_prompts import build_v5_extractor_input,build_v5_extractor_instructions
from local_slm_lab.v6_prompt_revision import build_revised_instructions
from local_slm_lab.v6_schema_fix import build_schema_fixed_instructions
from forecasting_assistant.domain.models import ExtractorResult

SPEC=importlib.util.spec_from_file_location('matched_runner',ROOT/'scripts/evaluate-v6-matched.py')
matched=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(matched)

class PromptProvider(matched.SynchronizedProvider):
    instructions=build_v5_extractor_instructions()

    async def extract(self,message,state):
        self._torch.cuda.synchronize()
        try:
            return self._structured('extract',self.instructions,build_v5_extractor_input(message,state,self.schema),ExtractorResult)
        finally:
            self._torch.cuda.synchronize()

def run(args):
    study=args.study.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (out.parent/'gpu-training.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        return run_locked(study,out)

def run_locked(study,out):
    protocol=json.loads((study/'protocol.json').read_text())
    assert protocol['total_scored_calls']==384 and protocol['training_started'] is False and protocol['sealed_final_accessed'] is False
    assert sha256(study/'natural.jsonl')==protocol['natural_sha256']
    cases,_=load_cases(study/'natural.jsonl',split='validation');schema=load_schema();plan=build_call_plan(cases,schema)
    assert len(cases)==32 and len(plan)==96
    assert digest([c['key'] for c in plan])==protocol['ordered_keys_sha256']
    assert digest([[c['key'],digest(c['context']),digest(c['expected'])] for c in plan])==protocol['context_gold_sha256']
    builders={'original':build_v5_extractor_instructions,'revised':build_revised_instructions,
              'schema_fixed':build_schema_fixed_instructions}
    arms=tuple(protocol['arms']);assert len(arms)==2 and len(set(arms))==2
    prompts={arm:builders[arm]() for arm in arms}
    for arm,text in prompts.items():assert digest(text)==protocol['prompt_sha256'][arm]
    adapter=matched.V6
    for name,expected in protocol['v6_adapter_sha256'].items():assert sha256(adapter/name)==expected,name
    snapshot=Path(os.environ['HF_HOME'])/'hub/models--Qwen--Qwen3.5-2B/snapshots'/protocol['base_revision']
    assert sha256(snapshot/'model.safetensors-00001-of-00001.safetensors')=='aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1'
    sources=[Path(__file__),ROOT/'scripts/evaluate-v6-matched.py',ROOT/'scripts/audit-v6-natural-failures.py',
             ROOT/'scripts/analyze-v6-prompt-comparison.py',ROOT/'local_slm_lab/v6_prompt_revision.py',
             ROOT/'local_slm_lab/v6_schema_fix.py']
    identity={'protocol_sha256':sha256(study/'protocol.json'),'provenance':provenance('base'),
              'source_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in sources},
              'schema_sha256':digest(schema.model_dump(mode='json'))}
    fingerprint=digest(identity)
    manifest=out/'run-manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())['fingerprint']==fingerprint,'Cannot resume changed inputs/code/runtime'
    else:matched.atomic(manifest,{'fingerprint':fingerprint,'identity':identity,'started_utc':now()})
    if (out/'completion.json').exists():return json.loads((out/'completion.json').read_text())
    import torch
    torch.set_num_threads(4)
    config=PeftInferenceConfig(base_model=protocol['base_model'],revision=protocol['base_revision'],variant='custom',
                               adapter_path=adapter,device='cuda',dtype='bfloat16',max_new_tokens=1024)
    provider=PromptProvider(config,schema)
    assert all(p.device.type=='cuda' for p in provider._model.parameters())
    warmup_cases,_=load_cases(ROOT/'corpus-v5/v5-20260919-r1/splits/smoke.jsonl',split='validation')
    warmup_plan=build_call_plan(warmup_cases[:1],schema)[:1];warmups={}
    for arm,text in prompts.items():
        provider.instructions=text
        warmups[arm]=asyncio.run(evaluate(provider,warmup_cases[:1],schema=schema,plan=warmup_plan))
    sessions=out/'sessions';sessions.mkdir(exist_ok=True)
    matched.atomic(sessions/(now().replace(':','-')+'.json'),{'runtime':runtime_metadata(provider),'warmups':warmups,
                    'arms_share_identical_v6_weights':True,'torch_threads':torch.get_num_threads(),'batch_size':1})
    print('READY: verified frozen inputs, prompts, base and v6 hashes; both prompts warmed up.',flush=True)
    completed=0
    for p in out.glob('*/*/[0-9][0-9][0-9][0-9].json'):
        report=json.loads(p.read_text());assert report['fingerprint']==fingerprint and report['status']=='complete'
        completed+=len(report['records'])
    status={'status':'running','started_utc':now(),'fingerprint':fingerprint,'completed_calls':completed,'total_calls':384}
    matched.atomic(out/'status.json',status)
    for track in ('natural_component','natural_rollout'):
        for index,case in enumerate(cases):
            calls=[c for c in plan if c['scenario_id']==case['scenario_id']]
            for arm in (arms if index%2==0 else tuple(reversed(arms))):
                folder=out/arm/track;folder.mkdir(parents=True,exist_ok=True);target=folder/f'{index:04d}.json'
                if target.exists():continue
                provider.instructions=prompts[arm]
                state=matched.create_initial_state(schema)
                report={'status':'partial','fingerprint':fingerprint,'arm':arm,'track':track,'scenario_id':case['scenario_id'],'records':[]}
                journal=target.with_suffix('.partial.json')
                if journal.exists():os.replace(journal,folder/f'{index:04d}.interrupted-{now().replace(":","-")}.json')
                for turn,gold in enumerate(calls,1):
                    call=matched.rollout_call(gold,state) if track=='natural_rollout' else gold
                    row=asyncio.run(evaluate(provider,[case],schema=schema,plan=[call]))['records'][0]
                    row.update(arm=arm,track=track,evaluation_domain=case['evaluation_domain'],test_kind=case['turns'][turn-1]['test_kind'],
                               prompt_sha256=protocol['prompt_sha256'][arm],hit_generation_limit=(row.get('completion_tokens') or 0)>=1024)
                    if track=='natural_rollout':state=matched.advance_rollout(state,row,schema,turn)
                    report['records'].append(row);completed+=1;matched.atomic(journal,report)
                    status.update(completed_calls=completed,heartbeat_utc=now(),current_arm=arm,current_track=track,
                                  current_scenario=case['scenario_id'],last_latency_ms=row['latency_ms'])
                    matched.atomic(out/'status.json',status)
                report['status']='complete';matched.atomic(target,report);journal.unlink()
                print(f'DONE {track} {arm} {case["scenario_id"]} [{completed}/384 calls]',flush=True)
    assert completed==384
    for arm in prompts:
        for track in ('natural_component','natural_rollout'):
            reports=[json.loads(p.read_text()) for p in sorted((out/arm/track).glob('[0-9][0-9][0-9][0-9].json'))]
            assert len(reports)==32
            records=[r for report in reports for r in report['records']];assert len(records)==96
            matched.atomic(out/f'{arm}-{track}.json',{'status':'complete','fingerprint':fingerprint,'arm':arm,'track':track,'records':records})
    completion={'status':'complete','scored_calls':384,'finished_utc':now(),'fingerprint':fingerprint,'arms':list(arms),
                'sealed_final_accessed':False,'training_started':False,'paid_apis_used':False}
    matched.atomic(out/'completion.json',completion);matched.atomic(out/'status.json',completion)
    return completion

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'evaluation/v6-prompt-audit-20261006')
    parser.add_argument('--output',type=Path,required=True)
    try:print(json.dumps(run(parser.parse_args()),indent=2),flush=True)
    except BaseException as error:
        print('STOP:',type(error).__name__,str(error),flush=True)
        raise
