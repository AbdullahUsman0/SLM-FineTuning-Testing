"""Resumable local-only diagnostic. Never a final test or hosted-model comparison."""
from __future__ import annotations
import argparse,gzip,hashlib,json,os,statistics,sys,time
from collections import Counter,defaultdict
from pathlib import Path
from data_tools import ROOT,CORPUS,BASE_REVISION,sha,atomic,dumps,rows,use_fpy,snapshot,restore,intervene,admit,strict_json,verify

def digest(v):return hashlib.sha256(dumps(v).encode()).hexdigest()
def select_cases(cases):
    """100 development conversations: 5/domain, exactly 80/10/10 languages."""
    by_domain=defaultdict(list)
    for c in cases:by_domain[c['domain']].append(c)
    chosen=[]
    for i,(domain,items) in enumerate(sorted(by_domain.items())):
        ordered=sorted(items,key=lambda c:digest(['v8-development-diagnostic-1',c['scenario_id']]))
        focused=next(c for c in ordered if c['language']=='en' and c['origin']=='new_failure_targeted_controlled_synthetic')
        english=[c for c in ordered if c['language']=='en' and c['origin']!='new_failure_targeted_controlled_synthetic'][:3]
        non_english=next(c for c in ordered if c['language']==('ur' if i%2==0 else 'mixed') and c['origin']!='new_failure_targeted_controlled_synthetic')
        chosen.extend([focused,*english,non_english])
    assert len(chosen)==100 and Counter(c['language'] for c in chosen)=={'en':80,'ur':10,'mixed':10}
    return sorted(chosen,key=lambda c:c['scenario_id'])

def decoded_tuples(obj,wire=False):
    result=[]
    if not isinstance(obj,dict):return result
    for u in obj.get('updates',[]) if isinstance(obj.get('updates'),list) else []:
        if not isinstance(u,dict) or u.get('slot_id')=='intent' or not all(k in u for k in ('slot_id','candidate_value','status')):continue
        value=u['candidate_value']
        if wire and isinstance(value,str):
            try:value=strict_json(value)
            except ValueError:pass
        result.append((str(u['slot_id']),dumps(value),str(u['status'])))
    return result

def counts(gold,emitted,valid):
    gold=Counter(gold);pred=Counter(emitted);tp=sum((gold&pred).values()) if valid else 0
    return [tp,sum(pred.values())-tp,sum(gold.values())-tp]

def prf(items):
    tp,fp,fn=map(sum,zip(*items)) if items else (0,0,0)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def read_journal(path):
    if not path.exists():return {}
    raw=path.read_bytes();parts=raw.splitlines(keepends=True);valid=[];records={}
    for i,line in enumerate(parts):
        try:r=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if i!=len(parts)-1:raise ValueError('Corrupt interior journal record')
            # Keep the incomplete tail, then recover only complete durable records.
            backup=path.with_name(path.name+'.incomplete-tail-'+hashlib.sha256(line).hexdigest()[:12])
            if not backup.exists():backup.write_bytes(line)
            temporary=path.with_name(path.name+'.recovered');temporary.write_bytes(b''.join(valid));temporary.replace(path)
            break
        key=tuple(r['key'])
        if key in records:raise ValueError('Duplicate scored journal key')
        records[key]=r;valid.append(line if line.endswith(b'\n') else line+b'\n')
        if i==len(parts)-1 and not line.endswith(b'\n'):
            temporary=path.with_name(path.name+'.recovered');temporary.write_bytes(b''.join(valid));temporary.replace(path)
    return records

def append(path,item):
    with path.open('ab',buffering=0) as f:
        f.write((dumps(item)+'\n').encode());os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('--fpy',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();use_fpy(a.fpy);verification=verify(CORPUS)
    from forecasting_assistant.domain.schema import create_initial_state,load_schema
    from forecasting_assistant.domain.models import ExtractorResult,DialogueTurn
    from forecasting_assistant.application.state_reducer import apply_extraction
    from local_slm_lab.v5_prompts import build_v5_extractor_input
    from run import run_lock
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with run_lock(output/'evaluation.lock'):
        cases=select_cases(list(rows(CORPUS/'splits/validation.jsonl.gz')));schema=load_schema()
        instructions=(CORPUS/'extractor-instructions.txt').read_text(encoding='utf-8');wire=json.loads((CORPUS/'response-schema.json').read_bytes())
        freeze={'version':'v8-development-diagnostic-1','corpus_manifest_sha256':verification['manifest_sha256'],
            'adapter_files':{name:sha(a.adapter/name) for name in ['adapter_model.safetensors','adapter_config.json']},
            'runner_sha256':sha(__file__),'data_tools_sha256':sha(Path(__file__).with_name('data_tools.py')),
            'provider_sources':{name:sha(ROOT/'local_slm_lab'/name) for name in ['v5_provider.py','peft_provider.py','v5_prompts.py']},
            'cases_sha256':digest(cases),'scenario_ids':[c['scenario_id'] for c in cases],'base_revision':BASE_REVISION,
            'policy':'strict JSON schema; one candidate decode; exact current evidence; reject invalid reducer state; no repair',
            'generation':{'max_new_tokens':1024,'thinking':False,'sampling':False,'repetition_penalty':1.,'dtype':'bfloat16','seed':42},
            'scope':'inspected synthetic development diagnostic, not a new final test','human_review_complete':False,'paid_API_calls':0}
        freeze_path=output/'freeze.json'
        if freeze_path.exists():
            if json.loads(freeze_path.read_bytes())!=freeze:raise ValueError('Evaluation inputs/code/adapter changed; use a NEW output directory')
        else:atomic(freeze_path,freeze)
        journal=output/'calls.jsonl';records=read_journal(journal);provider=None;seen=set()
        for track in ('component','rollout'):
            for case in cases:
                state=create_initial_state(schema)
                for number,t in enumerate(case['turns'],1):
                    if track=='component':state=restore(t,schema)
                    else:intervene(state,t.get('state_intervention'))
                    key=(track,case['scenario_id'],t['turn_id']);seen.add(key)
                    before_nonintent={k:v for k,v in snapshot(state).items() if k!='intent'}
                    context=build_v5_extractor_input(t['message'],state,schema)
                    identity=digest([context,t['gold_extraction'],t['expected_state_after']])
                    if key in records:
                        r=records[key]
                        if r['context_gold_sha256']!=identity:raise ValueError('Resume context/gold mismatch')
                    else:
                        if provider is None:
                            import torch
                            from local_slm_lab.peft_provider import PeftInferenceConfig
                            from local_slm_lab.v5_provider import V5TransformersProvider
                            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
                            torch.set_num_threads(4)
                            provider=V5TransformersProvider(PeftInferenceConfig(base_model='Qwen/Qwen3.5-2B',revision=BASE_REVISION,
                                variant='custom',adapter_path=a.adapter,device='cuda',dtype='bfloat16',max_new_tokens=1024),schema)
                            if not provider._model.active_adapters:raise ValueError('Adapter inactive')
                            if getattr(provider._model.config,'_commit_hash',None)!=BASE_REVISION:raise ValueError('Loaded revision drift')
                            atomic(output/'runtime.json',{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                                'model_class':type(provider._model).__name__,'active_adapters':provider._model.active_adapters,
                                'parameter_devices':sorted({str(x.device) for x in provider._model.parameters()}),
                                'chat_template_sha256':provider.chat_template_sha256})
                            # One separate unscored fixture to initialize kernels; no benchmark labels used.
                            warm=build_v5_extractor_input('Forecast air temperature for the next seven days.',create_initial_state(schema),schema)
                            provider._generate_raw(instructions,warm)
                        torch=provider._torch;torch.cuda.synchronize();start=time.perf_counter()
                        raw,prompt_tokens,completion_tokens,limit=provider._generate_raw(instructions,context)
                        torch.cuda.synchronize();elapsed=time.perf_counter()-start
                        r={'key':list(key),'context_gold_sha256':identity,'raw_output':raw,'prompt_tokens':prompt_tokens,
                           'completion_tokens':completion_tokens,'hit_generation_limit':limit,'latency_seconds':elapsed,
                           'domain':case['domain'],'domain_family':case['domain_family'],'language':case['language'],'category':case['category']}
                        append(journal,r);records[key]=r
                        print(dumps({'completed_calls':len(records),'track':track,'scenario':case['scenario_id'],'turn':t['turn_id']}),flush=True)
                    gold=ExtractorResult.model_validate(t['gold_extraction']).model_dump(mode='json');valid=False;json_valid=False;schema_valid=False;obj=None;error=None
                    try:
                        obj=strict_json(r['raw_output']);json_valid=True
                        from jsonschema import Draft202012Validator
                        Draft202012Validator(wire).validate(obj);schema_valid=True
                        if r['hit_generation_limit']:raise ValueError('Generation limit')
                        predicted=admit(r['raw_output'],t['message'],wire)
                        after=apply_extraction(state,predicted,schema,number,t['message'])
                        if any(s.status.value=='invalid' for s in after.slots.values()):raise ValueError('Invalid reduced slot')
                        state=after;valid=True
                    except Exception as exc:error=type(exc).__name__
                    emitted=decoded_tuples(obj,wire=True);g=decoded_tuples(gold)
                    r.update(confusion=counts(g,emitted,valid),admitted=valid,json_valid=json_valid,schema_valid=schema_valid,error_type=error,
                        extra_slot_flags=sum(1 for x in emitted if x[0] not in {gold_tuple[0] for gold_tuple in g}),
                        forbidden_slot_flags=sum(1 for x in emitted if x[0] in t['forbidden_slots']),
                        complete_state=snapshot(state)==t['expected_state_after'],gold_correction=t['gold_extraction']['correction_detected'],
                        expected_updates=len(g),source_group=case['entity_group'],state_intervention=bool(t.get('state_intervention')),
                        intent_correct=valid and obj['intent']==t['gold_extraction']['intent'],
                        unknown_turn=t['behavior'] in {'unknown_omission','focused_unknown_preserve','unknown_without_defaults'},
                        unknown_no_updates=valid and not emitted,
                        unknown_preserved_nonintent={k:v for k,v in snapshot(state).items() if k!='intent'}==before_nonintent,
                        correct_correction=valid and bool(obj['correction_detected']) and Counter(g)==Counter(emitted))
                    state.turns.append(DialogueTurn(turn_number=number,user_message=t['message']))
        if set(records)!=seen:raise ValueError('Journal contains calls outside this frozen cohort')
        def summarize(items):
            corrections=[r for r in items if r['gold_correction']]
            unknown=[r for r in items if r['unknown_turn']]
            return {'calls':len(items),'nonintent_exact':prf([r['confusion'] for r in items]),'json_valid':sum(r['json_valid'] for r in items),
                'schema_valid':sum(r['schema_valid'] for r in items),'admitted':sum(r['admitted'] for r in items),
                'extra_slot_flags_including_rejected':sum(r['extra_slot_flags'] for r in items),'forbidden_slot_flags_including_rejected':sum(r['forbidden_slot_flags'] for r in items),
                'complete_state_turns':sum(r['complete_state'] for r in items),'correction_calls':len(corrections),
                'correct_correction_calls':sum(r['correct_correction'] for r in corrections),'complete_state_after_correction':sum(r['complete_state'] for r in corrections),
                'intent_correct_calls':sum(r['intent_correct'] for r in items),'unknown_turns':len(unknown),
                'unknown_successful_empty_calls':sum(r['unknown_no_updates'] for r in unknown),
                'unknown_preserved_nonintent_state':sum(r['unknown_preserved_nonintent'] for r in unknown),
                'median_extraction_seconds':statistics.median(r['latency_seconds'] for r in items)}
        summary={'status':'complete','scope':freeze['scope'],'model_winner_established':False,'human_review_complete':False,'paid_API_calls':0,
            'freeze_sha256':sha(freeze_path),'journal_sha256':sha(journal),'conversations':len(cases),'tracks':{}}
        for track in ('component','rollout'):
            items=[r for r in records.values() if r['key'][0]==track]
            summary['tracks'][track]=summarize(items)
            summary['tracks'][track]['excluding_seeded_interventions']=summarize([r for r in items if r['category']!='explicit_state_repair'])
            summary['tracks'][track]['languages']={v:summarize([r for r in items if r['language']==v]) for v in ('en','ur','mixed')}
            summary['tracks'][track]['domains']={v:summarize([r for r in items if r['domain_family']==v]) for v in ('weather','economics','combined')}
        atomic(output/'summary.json',summary)
        with gzip.open(output/'scored-records.jsonl.gz','wt',encoding='utf-8') as f:
            for key in sorted(records):f.write(dumps(records[key])+'\n')
        print(dumps(summary))

if __name__=='__main__':main()
