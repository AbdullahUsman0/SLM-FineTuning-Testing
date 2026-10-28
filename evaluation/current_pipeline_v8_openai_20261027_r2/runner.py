"""Recoverable local/hosted runner. Default execution stops on any unmet gate."""
from __future__ import annotations
import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from controls import BudgetStop, Ledger, MODEL, append, atomic, digest, dumps, freeze, lock, read_journal, request_input_bound, sha
import pipeline as p


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_protocol(run, repo):
    protocol=json.loads((run/'protocol.json').read_bytes())
    for name,expected in protocol['runner_source_sha256'].items():
        if sha(Path(__file__).parent/name)!=expected:
            raise ValueError('Runner source changed: '+name)
    for name,expected in protocol['fpy_source_sha256'].items():
        if sha(run/'fpy'/name)!=expected:
            raise ValueError('fpy source changed: '+name)
    for name,expected in protocol['provider_source_sha256'].items():
        if sha(repo/name)!=expected:
            raise ValueError('Provider source changed: '+name)
    for filename,field in [('adjudication-authorization.json','adjudication_authorization_sha256'),('adjudication-changes.json','adjudication_changes_sha256')]:
        if sha(run/filename)!=protocol[field]:
            raise ValueError('Adjudication source changed')
    for name,expected in protocol['adapter_sha256'].items():
        if sha(Path(protocol['adapter'])/name)!=expected:
            raise ValueError('Adapter changed')
    for stage in ('main','pilot'):
        if digest(json.loads((run/(stage+'-cases.json')).read_bytes()))!=protocol[stage+'_case_sha256']:
            raise ValueError('Frozen case drift')
    p.configure(run/'fpy',repo)
    if p.wire_schema()!=protocol['response_schema']:
        raise ValueError('Schema drift')
    return protocol


class Hosted:
    def __init__(self, run, protocol, ledger):
        from openai import OpenAI
        self.run=run;self.protocol=protocol;self.ledger=ledger
        self.client=OpenAI(api_key=os.environ.get('OPENAI_API_KEY'),max_retries=0,timeout=90)
        model=self.client.models.retrieve(MODEL)
        if model.id!=MODEL:
            raise RuntimeError('Requested hosted snapshot unavailable; substitution prohibited')

    def generate(self, key, context):
        request={'model':MODEL,'instructions':self.protocol['instructions'],'input':context,'text':{'format':{'type':'json_schema','name':'ExtractorResult','strict':True,'schema':self.protocol['response_schema']}},'max_output_tokens':2048,'reasoning':{'effort':'minimal'},'store':False}
        dispatch=dumps(key);context_hash=digest(request)
        directory=self.run/'hosted-responses';directory.mkdir(exist_ok=True)
        response_path=directory/(digest(key)+'.json')
        # A crash after response fsync but before journal append replays stored bytes,
        # reconciles the SAME reservation, and never dispatches a second request.
        if response_path.exists():
            saved=json.loads(response_path.read_bytes())
            if saved['request_sha256']!=context_hash:
                raise ValueError('Saved hosted request context drift')
            response=saved['response']
            self.ledger.reconcile(dispatch,context_hash,response.get('usage') or {},sha(response_path))
            if response.get('model')!=MODEL:
                raise RuntimeError('Saved hosted model mismatch; substitution prohibited')
            return saved['record']
        freeze(directory/(digest(key)+'.request.json'),{'key':key,'request':request,'request_sha256':context_hash,'input_token_upper_bound':request_input_bound(request),'bound_method':'UTF-8 request bytes + 16384 overhead tokens; no cache discount; application guard, not provider billing guarantee'})
        self.ledger.reserve(dispatch,context_hash,request_input_bound(request))
        start=time.perf_counter()
        try:
            response=self.client.responses.create(**request)
        except BaseException as exc:
            self.ledger.uncertain(dispatch)
            append(self.run/'infrastructure-errors.jsonl',{'key':key,'at_UTC':utc(),'error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None),'budget':'reservation retained; no automatic retry'})
            raise BudgetStop('Hosted request failed or uncertain; reservation retained') from None
        elapsed=time.perf_counter()-start
        body=response.model_dump(mode='json')
        refused=any(part.get('type')=='refusal' for output in body.get('output',[]) for part in output.get('content',[]))
        record={'raw_output':response.output_text,'latency_seconds':elapsed,'capped':body.get('status')=='incomplete','refused':refused,'response_id':body.get('id'),'usage':body.get('usage'),'response_model':body.get('model')}
        atomic(response_path,{'key':key,'request_sha256':context_hash,'response':body,'record':record})
        self.ledger.reconcile(dispatch,context_hash,body.get('usage') or {},sha(response_path))
        atomic(self.run/'budget-status.json',self.ledger.summary())
        if body.get('model')!=MODEL:
            raise RuntimeError('Hosted response model mismatch; response preserved, no substitution')
        return record


class Local:
    def __init__(self,run,protocol,schema):
        import torch
        from local_slm_lab.peft_provider import PeftInferenceConfig
        from local_slm_lab.v5_provider import V5TransformersProvider
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 unavailable')
        torch.set_num_threads(4)
        self.provider=V5TransformersProvider(PeftInferenceConfig(base_model=protocol['base_model'],revision=protocol['base_revision'],variant='custom',adapter_path=Path(protocol['adapter']),device='cuda',dtype='bfloat16',max_new_tokens=1024),schema)
        model=self.provider._model
        if not model.active_adapters or getattr(model.config,'_commit_hash',None)!=protocol['base_revision']:
            raise RuntimeError('Local model revision/active adapter mismatch')
        devices=sorted({str(param.device) for param in model.parameters()})
        if devices!=['cuda:0']:
            raise RuntimeError('All model parameters must remain on CUDA')
        runtime={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'active_adapters':model.active_adapters,'model_class':type(model).__name__,'base_model_class':type(model.get_base_model()).__name__,'base_revision':getattr(model.config,'_commit_hash',None),'devices':devices,'dtype':str(next(model.parameters()).dtype),'chat_template_sha256':self.provider.chat_template_sha256}
        if runtime['base_model_class']!='Qwen3_5ForConditionalGeneration':
            raise RuntimeError('Wrong base architecture')
        atomic(run/'local-runtime.json',runtime)
        self.torch=torch;self.instructions=protocol['instructions']
        from forecasting_assistant.domain.schema import create_initial_state
        fixture='Forecast air temperature for the next seven days.'
        context=p.extraction_context(create_initial_state(schema),fixture,schema)
        smoke=self.generate(['unscored_local_warmup'],context)
        atomic(run/('local-smoke-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),{'scope':'unscored separate fixture, not a pilot/main case','fixture':fixture,'current_input':context,'record':smoke,'runtime':runtime})

    def generate(self,key,context):
        self.torch.cuda.synchronize();start=time.perf_counter()
        raw,prompt_tokens,completion_tokens,capped=self.provider._generate_raw(self.instructions,context)
        self.torch.cuda.synchronize()
        return {'raw_output':raw,'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,'capped':capped,'refused':False,'latency_seconds':time.perf_counter()-start}


def summarize(rows):
    def group(items):
        corrections=[r for r in items if r['gold_correction']]
        unknown=[r for r in items if r['unknown_turn']]
        return {'calls':len(items),'raw_exact':p.prf(items),'json_valid':sum(r['json_valid'] for r in items),'schema_valid':sum(r['schema_valid'] for r in items),'raw_contract_admitted':sum(r['raw_contract_admitted'] for r in items),'reducer_admitted_without_recovery':sum(r['reducer_admitted'] for r in items),'exact_pipeline_state_turns':sum(r['complete_pipeline_state'] for r in items),'recovery_calls':sum(bool(r['provenance']['recovered_slot_ids']) for r in items),'extra_slot_flags':sum(r['extra_slot_flags'] for r in items),'forbidden_slot_flags':sum(r['forbidden_slot_flags'] for r in items),'duplicate_slot_calls':sum(r['duplicate_slots'] for r in items),'correction_calls':len(corrections),'correct_raw_corrections':sum(r['correct_correction'] for r in corrections),'complete_pipeline_state_after_correction':sum(r['complete_pipeline_state'] for r in corrections),'unknown_turns':len(unknown),'raw_unknown_omissions':sum(r['unknown_no_updates'] for r in unknown),'pipeline_unknown_preserved':sum(r['unknown_preserved'] for r in unknown),'median_seconds':statistics.median(r['latency_seconds'] for r in items) if items else None}
    result={}
    for arm in ('v8','openai'):
        result[arm]={}
        for track in ('component','rollout'):
            items=[r for r in rows if r['key'][1:3]==[arm,track]]
            result[arm][track]=group(items)
            for field in ('domain','language'):
                result[arm][track][field]={value:group([r for r in items if r[field]==value]) for value in sorted({r[field] for r in items})}
            result[arm][track]['excluding_intervention_category']=group([r for r in items if r['category']!='explicit_state_repair'])
    return result


def evaluate(run,repo,stage,arms):
    protocol=validate_protocol(run,repo)
    if not protocol['gold_contract_gate_passed']:
        raise RuntimeError('Gold/state contract review gate failed; inspect annotation-review-queue.json')
    tests=json.loads((run/'offline-tests.json').read_bytes())
    if not tests.get('passed') or tests.get('runner_source_sha256')!=protocol['runner_source_sha256']:
        raise RuntimeError('Offline tests missing or source-mismatched')
    if stage=='main':
        pilot=json.loads((run/'pilot-completion.json').read_bytes())
        if pilot.get('status')!='complete' or not pilot.get('main_affordability_passed'):
            raise RuntimeError('Hosted pilot/affordability gate not passed')
    from forecasting_assistant.domain.schema import create_initial_state,load_schema
    from forecasting_assistant.prompts.extractor import build_extractor_input
    from forecasting_assistant.application.state_reducer import apply_extraction
    schema=load_schema();cases=json.loads((run/(stage+'-cases.json')).read_bytes());journal=run/'calls.jsonl'
    records=read_journal(journal);ledger=Ledger(protocol['shared_budget_ledger']);scored=[]
    for arm in arms:
        generator=None
        for track in ('component','rollout'):
            for case in cases:
                state=create_initial_state(schema)
                for turn in case['turns']:
                    if track=='component':
                        state=p.restore(turn,schema)
                    else:
                        p.intervention(state,turn.get('state_intervention'))
                    for prior in state.turns:
                        prior.assistant_message=None
                    before={k:v for k,v in p.snapshot(state).items() if k!='intent'}
                    context=p.extraction_context(state,turn['message'],schema)
                    identity=digest([protocol['instructions'],context,turn['gold_extraction'],turn['expected_state_after']])
                    key=[stage,arm,track,case['scenario_id'],turn['turn_id']]
                    if tuple(key) in records:
                        record=records[tuple(key)]
                        if record['identity']!=identity:
                            raise ValueError('Resume context/gold drift')
                    else:
                        if generator is None:
                            generator=Local(run,protocol,schema) if arm=='v8' else Hosted(run,protocol,ledger)
                        record=generator.generate(key,context)
                        record.update(key=key,identity=identity,current_input=context,instructions_sha256=digest(protocol['instructions']))
                        append(journal,record);records[tuple(key)]=record
                    raw_metrics=p.raw_score(record,turn,protocol['response_schema'])
                    reducer_admitted=False;reducer_error=None
                    try:
                        model_result=p.admit(record['raw_output'],protocol['response_schema'],record.get('capped',False),record.get('refused',False))
                        reduced=apply_extraction(state,model_result,schema,len(state.turns)+1,turn['message'])
                        reducer_admitted=True
                        reduced_state=p.snapshot(reduced)
                    except Exception as exc:
                        reducer_error=type(exc).__name__;reduced_state=p.snapshot(state)
                    state,provenance=p.run_pipeline(state,turn['message'],schema,record,protocol['response_schema'])
                    item={**record,**raw_metrics,'domain':case['domain'],'language':case['language'],'category':case['category'],'gold_correction':turn['gold_extraction']['correction_detected'],'unknown_turn':turn['behavior'] in {'unknown_omission','focused_unknown_preserve','unknown_without_defaults'},'unknown_preserved':before=={k:v for k,v in p.snapshot(state).items() if k!='intent'},'reducer_admitted':reducer_admitted,'reducer_error_type':reducer_error,'reducer_state_before_recovery':reduced_state,'pipeline_state':p.snapshot(state),'expected_state':turn['expected_state_after'],'complete_pipeline_state':p.snapshot(state)==turn['expected_state_after'],'provenance':provenance}
                    scored.append(item)
                    atomic(run/'status.json',{'status':'running','stage':stage,'arm':arm,'track':track,'completed_journal_calls':len(records),'at_UTC':utc(),'pid':os.getpid()})
                    print(dumps({'key':key,'completed_calls':len(records)}),flush=True)
        if generator is not None and arm=='v8':
            del generator
            import gc;gc.collect()
            import torch;torch.cuda.empty_cache()
    with (run/(stage+'-scored-records.jsonl')).open('w',encoding='utf-8',newline='\n') as stream:
        for row in scored:stream.write(dumps(row)+'\n')
    atomic(run/(stage+'-summary.json'),summarize(scored))
    expected=4*sum(len(c['turns']) for c in cases)
    complete=len(scored)==expected and set(arms)=={'v8','openai'}
    budget=ledger.summary();atomic(run/'budget-status.json',budget)
    receipt={'status':'complete' if complete else 'partial_local_only','stage':stage,'completed_calls':len(scored),'planned_calls':expected,'paid_budget':budget,'protocol_sha256':sha(run/'protocol.json'),'new_final_test_access':False,'independent_human_review_complete':False,'model_winner_established':False}
    if stage=='pilot' and complete:
        hosted=[r for r in scored if r['key'][1]=='openai']
        main_turns=protocol['planned_main_calls_per_arm_track']
        # Pilot projection is an estimate, NOT permission to bypass per-dispatch bounds.
        projected=budget['actual_USD']+2*main_turns*(budget['actual_USD']/len(hosted))*1.5
        receipt.update(projected_total_USD_with_50pct_margin=projected,main_affordability_passed=projected<0.9 and budget['reserved_or_uncertain_USD']==0.)
    atomic(run/(stage+'-completion.json'),receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--stage',choices=['pilot','main'],required=True);parser.add_argument('--arms',nargs='+',choices=['v8','openai'],default=['v8','openai']);args=parser.parse_args()
    os.environ['HF_HOME']=r'D:\SLM\hf-cache';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    with lock(args.run/'worker.lock'),lock(Path(r'D:\SLM\FYP-experiment-control\current-pipeline-v8-openai-20261023\dispatch.lock')):
        awake=ctypes.windll.kernel32.SetThreadExecutionState
        awake(0x80000001)
        try:
            receipt=evaluate(args.run,args.repo,args.stage,args.arms)
            atomic(args.run/'status.json',{'status':receipt['status'],'stage':args.stage,'at_UTC':utc(),'pid':os.getpid()})
        except BaseException as exc:
            protocol=json.loads((args.run/'protocol.json').read_bytes())
            ledger=Ledger(protocol['shared_budget_ledger'])
            atomic(args.run/'budget-status.json',ledger.summary())
            atomic(args.run/'status.json',{'status':'blocked_preserved','stage':args.stage,'error_type':type(exc).__name__,'reason':str(exc)[:300] if not hasattr(exc,'request') else 'Provider prerequisite failed; see safe error type and credentials gate','at_UTC':utc(),'pid':os.getpid(),'no_automatic_retry':True})
            raise SystemExit(2) from None
        finally:
            awake(0x80000000)


if __name__=='__main__':main()
