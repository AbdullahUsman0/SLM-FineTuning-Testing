"""Final-only matched evaluator. Review/registration gate precedes any model loading."""
import argparse
import ctypes
from pathlib import Path
import time
import uuid
import runtime as rt
from registration import verify_final
from shared_policy import SharedPolicy
from journal import Journal,Budget
from providers import Local,Hosted,worst_attempt_cost,OPENAI_MODEL


def plans(cases,schema,policy):
    result={}
    for case in cases:
        state=rt.create_initial_state(schema);calls=[]
        for index,turn in enumerate(case['turns'],1):
            before=state.model_copy(deep=True)
            admission=policy.parse(rt.dumps(turn['gold_extraction']),turn['message'])
            state,accepted,error=policy.advance(state,admission,schema,index,turn['message'])
            if not accepted:raise ValueError('Reviewed gold rejected by frozen policy')
            calls.append({'index':index,'turn':turn,'before':before,'gold':admission.predicted,
                          'gold_after':rt.semantic_state(state)})
            rt.append_user(state,turn['message'])
        result[case['scenario_id']]=calls
    return result


def check_binding(payload,call_id,context_hash,gold_hash):
    if any(payload.get(key)!=value for key,value in {'call_id':call_id,'context_sha256':context_hash,'gold_sha256':gold_hash}.items()):
        raise ValueError('Cached response changed context, labels or call identity')


def grade(raw,case,call,track,arm,prior,policy,schema,*,from_journal=False):
    if arm=='openai' and (raw.get('response') or {}).get('model')!=OPENAI_MODEL:
        raise ValueError('Hosted response came from a different model snapshot')
    message=call['turn']['message']
    admission=policy.parse(raw['raw_output'],message,response_status=raw['response_status'],
        refused=raw['refused'],hit_generation_limit=raw['hit_generation_limit'])
    after,accepted,error=policy.advance(prior,admission,schema,call['index'],message)
    try:raw_value=rt.strict_json(raw['raw_output'])
    except (ValueError,TypeError):raw_value=None
    emitted=policy.emitted(raw['raw_output'],admission)
    prior_metadata=prior.model_dump(mode='json')['slots']
    after_metadata=after.model_dump(mode='json')['slots']
    identifiers=[u['slot_id'] for u in emitted['updates']]
    snapshot=rt.state_snapshot(prior)
    record={'task':'extract','key':[case['scenario_id'],'extract',call['index']],
        'call_id':raw['call_id'],'scenario_id':case['scenario_id'],'cluster_id':case['source_group'],
        'domain':case['domain'],'language':case['language'],'category':case['template_family'],
        'coverage':call['turn']['coverage'],'arm':arm,'track':track,
        'context':{'message':message,'state':snapshot},'context_sha256':raw['context_sha256'],
        'expected':call['gold'],'gold_sha256':raw['gold_sha256'],'raw_output':raw['raw_output'],
        'raw_value':raw_value,'emitted':emitted,'predicted':admission.predicted,
        'raw_json_valid':admission.structural.json_valid,'raw_schema_valid':admission.structural.declared_schema_valid,
        'structural_parser_admitted':admission.structural.admitted,'parser_admitted':admission.admitted,
        'evidence_grounded':admission.evidence_valid,'prediction_valid':accepted,'provider_success':True,
        'admission_error':admission.error_code,'application_error':error,'duplicate_slot_ids':len(identifiers)!=len(set(identifiers)),
        'predicted_after':rt.semantic_state(after),'gold_after':call['gold_after'],
        'transition_correct':accepted and rt.semantic_state(after)==call['gold_after'],
        'prior_semantic_state':rt.semantic_state(prior),'latency_ms':raw['generation_elapsed_seconds']*1000,
        'raw_slot_metadata_preserved':prior_metadata==after_metadata,
        'latency_measurement_origin':'original_generation_including_network_for_hosted; never journal-replay CPU time',
        'score_reconstructed_from_journal':from_journal,'generation_session':raw['generation_session'],
        'prompt_tokens':raw.get('prompt_tokens'),'completion_tokens':raw.get('completion_tokens'),
        'reasoning_tokens':raw.get('reasoning_tokens',0),'usage_cost_usd':raw.get('cost_usd'),
        'hit_generation_limit':raw['hit_generation_limit'],'refused':raw['refused'],
        'policy_version':policy.version,'model_calls':1,'retry_count':raw.get('retry_count',0)}
    rt.append_user(after,message)
    return record,after


def generate_once(provider,arm,message,state,schema,policy,call_id,raw_journal,budget,session,context_hash,gold_hash):
    if arm!='openai':
        payload=provider.generate(message,state,schema)
        payload.update(call_id=call_id,context_sha256=context_hash,gold_sha256=gold_hash,generation_session=session)
        raw_journal.save(call_id,payload)
        return payload
    while True:
        attempt_id=budget.reserve(call_id)
        try:
            payload=provider.generate(message,state,schema,policy)
        except Exception as error:
            safe={'type':type(error).__name__,'http_status':getattr(error,'status_code',None)}
            budget.complete(attempt_id,None,status='transport_outcome_uncertain',error=safe)
            # Completed model responses are never retried. Only retry transport errors once.
            retryable=safe['http_status'] in {429,500,502,503,504} or safe['type'] in {'APITimeoutError','APIConnectionError'}
            previous=sum(a['call_id']==call_id for a in budget.data['attempts'])
            if not retryable or previous>=2:raise RuntimeError('Hosted transport stopped; inspect sanitized budget ledger') from None
            time.sleep(5)
            continue
        payload.update(call_id=call_id,context_sha256=context_hash,gold_sha256=gold_hash,
            generation_session=session,attempt_id=attempt_id,
            retry_count=sum(a['call_id']==call_id for a in budget.data['attempts'])-1)
        # The completed response is durable BEFORE the reservation is reconciled.
        raw_journal.save(call_id,payload)
        budget.complete(attempt_id,payload['cost_usd'])
        if (payload.get('response') or {}).get('model')!=OPENAI_MODEL:
            raise RuntimeError('Hosted model identity differs; response journaled and expenditure preserved')
        return payload


def initialize(run,freeze,config):
    reviewed=verify_final(freeze)
    rt.verify_models(config)
    private=Path(reviewed['private']);cases=[rt.strict_json(line) for line in (private/'candidate/conversations.jsonl').read_text(encoding='utf-8').splitlines()]
    policy=SharedPolicy();schema=rt.load_schema();call_plans=plans(cases,schema,policy)
    planned=sum(len(c['turns']) for c in cases)
    if len(cases)!=240:raise ValueError('Fixed final cohort must contain 240 conversations')
    budget=rt.read(private/'review/budget-approval.json')
    identity={'final_freeze_fingerprint':reviewed['fingerprint'],'policy_version':policy.version,
        'cohort_sha256':rt.sha(private/'candidate/conversations.jsonl'),'planned_calls_per_track':planned,
        'approved_budget_sha256':rt.sha(private/'review/budget-approval.json')}
    run.mkdir(parents=True,exist_ok=True)
    manifest_path=run/'run-manifest.json'
    if manifest_path.exists():
        manifest=rt.read(manifest_path)
        identity['run_id']=manifest['run_id']
        if manifest['identity']!=identity or manifest['fingerprint']!=rt.digest(identity):raise ValueError('Run identity differs; no resume')
    else:
        identity['run_id']=str(uuid.uuid4())
        rt.atomic(manifest_path,{'run_id':identity['run_id'],'created_at':rt.now(),'identity':identity,
            'fingerprint':rt.digest(identity),'final_review_verified':True,'source_commit':rt.read(rt.CHECKPOINT/'workspace-intake.json')['source_commit']},exclusive=True)
    return cases,call_plans,schema,policy,identity,budget


def run_block(run,freeze,config,arm,group):
    cases,call_plans,schema,policy,identity,approval=initialize(run,freeze,config)
    cases=sorted([c for c in cases if c['source_group']==group],key=lambda c:c['scenario_id'])
    if len(cases)!=20:raise ValueError('Registered template group must contain 20 conversations')
    identity_hash=rt.digest(identity)
    records=Journal(run/'records'/arm,identity_hash);raws=Journal(run/'raw'/arm,identity_hash)
    block=run/'blocks'/arm/(group+'.json')
    if block.exists():
        marker=rt.read(block)
        if marker['identity']!=identity_hash:raise ValueError('Block identity mismatch')
        for call_id,checksum in marker['record_sha256'].items():
            if rt.digest(records.load(call_id))!=checksum:raise ValueError('Completed block checksum mismatch')
        return
    session=str(uuid.uuid4());provider=None
    protocol=rt.read(rt.CHECKPOINT/'experiment-protocol.json')
    budget=None
    if arm=='openai':
        budget=Budget(run/'api-budget-ledger.json',identity=identity_hash,cap_usd=approval['cap_usd'],
            attempt_bound=worst_attempt_cost(),max_attempts=protocol['hosted']['maximum_attempts'],max_attempts_per_call=2)
    record_hashes={}
    for track in ('component','rollout'):
        for case in cases:
            state=rt.create_initial_state(schema)
            for call in call_plans[case['scenario_id']]:
                if (run/'stop.request').exists():raise InterruptedError('Stop requested at call boundary')
                prior=call['before'].model_copy(deep=True) if track=='component' else state.model_copy(deep=True)
                call_id=f"{arm}-{track}-{case['scenario_id']}-{call['index']:02d}"
                context_hash=rt.digest({'message':call['turn']['message'],'state':rt.state_snapshot(prior)})
                gold_hash=rt.digest(call['gold'])
                existing=records.load(call_id)
                if existing is not None:
                    check_binding(existing,call_id,context_hash,gold_hash)
                    raw=raws.load(call_id)
                    if raw is None:raise ValueError('Completed scored call has no durable raw response')
                    check_binding(raw,call_id,context_hash,gold_hash)
                    replay,state_after=grade(raw,case,call,track,arm,prior,policy,schema,
                        from_journal=existing['score_reconstructed_from_journal'])
                    if replay!=existing:raise ValueError('Cached scored response/state cannot be reproduced exactly')
                    if budget is not None:budget.reconcile(raw['attempt_id'],raw['cost_usd'])
                else:
                    raw=raws.load(call_id);recovered=raw is not None
                    if raw is not None:
                        check_binding(raw,call_id,context_hash,gold_hash)
                        if budget is not None:budget.reconcile(raw['attempt_id'],raw['cost_usd'])
                    else:
                        if provider is None:
                            provider=Hosted() if arm=='openai' else Local(arm,config)
                            rt.atomic(run/'sessions'/(arm+'-'+session+'.json'),{'arm':arm,'group':group,'session':session,
                                'runtime':provider.inventory,'started_at':rt.now()},exclusive=True)
                            # Unscored warmup is separate, bounded and never enters quality denominators.
                            warm_id='warmup-'+arm+'-'+(identity_hash if arm=='openai' else session)
                            if raws.load(warm_id) is None:
                                warm=generate_once(provider,arm,'Forecast smoke-fixture soil moisture for 3 hours.',
                                    rt.create_initial_state(schema),schema,policy,warm_id,raws,budget,session,'unscored','unscored')
                                rt.atomic(run/'warmups'/(warm_id+'.json'),warm,exclusive=True)
                        raw=generate_once(provider,arm,call['turn']['message'],prior,schema,policy,call_id,raws,budget,session,context_hash,gold_hash)
                    replay,state_after=grade(raw,case,call,track,arm,prior,policy,schema,from_journal=recovered)
                    records.save(call_id,replay)
                if track=='rollout':state=state_after
                record_hashes[call_id]=rt.digest(replay)
                rt.atomic(run/'status.json',{'status':'running','phase':'evaluation','arm':arm,'track':track,'group':group,
                    'last_call_id':call_id,'completed_scored_calls':sum(1 for p in (run/'records').rglob('*.json')),
                    'planned_scored_calls':identity['planned_calls_per_track']*6,'updated_at':rt.now(),
                    'budget_exposure_upper_bound_usd':None if budget is None else budget.exposure})
    rt.verify_models(config)
    rt.atomic(block,{'status':'complete','identity':identity_hash,'arm':arm,'group':group,
        'record_sha256':record_hashes,'finished_at':rt.now()},exclusive=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--arm',choices=['v6','v7','openai'],required=True)
    parser.add_argument('--group',required=True)
    args=parser.parse_args();config=rt.read(args.config)
    # Review is checked before taking a lock or importing torch/OpenAI.
    verify_final(args.freeze)
    with rt.lock(args.run/'run.lock'):
        awake=ctypes.windll.kernel32.SetThreadExecutionState
        if not awake(0x80000001):raise RuntimeError('Unable to request system awake state')
        try:run_block(args.run,args.freeze,config,args.arm,args.group)
        except Exception as error:
            rt.atomic(args.run/'status.json',{'status':'stopped','phase':'evaluation','error_type':type(error).__name__,
                'HTTP_status':getattr(error,'status_code',None),'updated_at':rt.now(),'training_started':False})
            print('Evaluation stopped: '+type(error).__name__+'; inspect sanitized status and ledger.',flush=True)
            raise SystemExit(1)
        finally:awake(0x80000000)


if __name__=='__main__':main()
