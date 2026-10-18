"""Shared checks for the final experimental development corpus; no final-test access."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from datetime import date, datetime
import gzip, hashlib, json, re, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
VERSION='v8-final-20261018-r1'
CORPUS=ROOT/'corpus-v8'/VERSION
BASE_REVISION='15852e8c16360a2fea060d615a32b45270f8a8fc'
FPY_REVISION='04d52c015d1e3ecdefe92b87116f209361509b4b'
PROMPT_VERSION='v8-grounded-role-isolation-1'

def dumps(v):
    return json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,
        default=lambda x:x.isoformat() if isinstance(x,(date,datetime)) else str(x))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    # Preserve the published v8 Windows-authored JSON bytes on Linux as well.
    with tmp.open('w',encoding='utf-8',newline='\r\n') as f:f.write(dumps(v)+'\n')
    tmp.replace(p)

def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def strict_json(s):
    def unique(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('Duplicate JSON key')
            d[k]=v
        return d
    def constant(v):raise ValueError('Nonfinite number')
    value=json.loads(s,object_pairs_hook=unique,parse_constant=constant)
    json.dumps(value,allow_nan=False)
    return value

def use_fpy(path):
    """Select a dedicated pinned checkout, never reset the user's working fpy."""
    p=Path(path).resolve()
    head=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    if head!=FPY_REVISION:raise ValueError('A dedicated fpy checkout at '+FPY_REVISION+' is required')
    if subprocess.check_output(['git','-C',str(p),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Pinned fpy has tracked modifications')
    sys.path.insert(0,str(p/'src'));sys.path.insert(1,str(ROOT))
    import forecasting_assistant.domain.models as models
    if not Path(models.__file__).resolve().is_relative_to(p/'src'):raise ValueError('Wrong fpy already imported')
    return p

def snapshot(state):
    return {s:{'value':json.loads(dumps(x.value)),'status':x.status.value,'confirmed_by_user':x.confirmed_by_user}
            for s,x in state.slots.items() if x.value is not None}

def restore(turn,schema):
    from forecasting_assistant.domain.models import SlotState,Intent,DialogueTurn
    from forecasting_assistant.domain.schema import create_initial_state
    state=create_initial_state(schema);state.intent=Intent(turn['established_intent'])
    for sid,item in turn['context_state'].items():
        state.slots[sid]=SlotState(slot_id=sid,**item,confidence=1.,evidence_text='Recorded prior synthetic fact')
    for i,message in enumerate(turn['prior_user_messages'],1):state.turns.append(DialogueTurn(turn_number=i,user_message=message))
    return state

def admit(raw,message,schema_file):
    """Strict outer wire, one candidate-value decode, exact current-message evidence."""
    from jsonschema import Draft202012Validator
    from forecasting_assistant.domain.models import ExtractorResult
    obj=strict_json(raw)
    Draft202012Validator(schema_file).validate(obj)
    seen=set()
    for u in obj['updates']:
        if u['slot_id'] in seen:raise ValueError('Duplicate slot update')
        seen.add(u['slot_id'])
        if not u['evidence_text'].strip() or u['evidence_text'] not in message:raise ValueError('Evidence is not an exact current span')
        strict_json(u['candidate_value'])
    return ExtractorResult.model_validate(obj)

def intervene(state,change):
    from forecasting_assistant.domain.models import SlotState
    if not change:return
    for x in change['mutations']:
        if x['kind']=='omitted_prior_value':state.slots[x['slot']]=SlotState(slot_id=x['slot'])
        elif x['kind']=='wrong_prior_value':
            state.slots[x['slot']].value=deepcopy(x['value']);state.slots[x['slot']].evidence_text='Declared synthetic context intervention'
        else:raise ValueError('Unknown context intervention')

def verify(corpus=CORPUS,semantic=False):
    corpus=Path(corpus)
    lock=json.loads((corpus/'manifest.sha256.json').read_bytes())
    if sha(corpus/'manifest.json')!=lock['manifest_sha256']:raise ValueError('Manifest checksum mismatch')
    manifest=json.loads((corpus/'manifest.json').read_bytes())
    for name,rec in manifest['files'].items():
        p=(corpus/name).resolve()
        if not p.is_relative_to(corpus.resolve()) or sha(p)!=rec['sha256']:raise ValueError('File checksum mismatch: '+name)
    if semantic:
        from forecasting_assistant.domain.schema import load_schema,create_initial_state
        from forecasting_assistant.domain.models import DialogueTurn
        from forecasting_assistant.application.state_reducer import apply_extraction
        from local_slm_lab.v5_prompts import build_v5_extractor_input
        schema=load_schema()
    wire=json.loads((corpus/'response-schema.json').read_bytes())
    instructions=(corpus/'extractor-instructions.txt').read_text(encoding='utf-8')
    counts={};groups={};keys={};input_hashes={};positive=Counter();grounded=0
    for split in ('train','validation'):
        count=Counter();seen=set();groups[split]=set();keys[split]=set();input_hashes[split]=set()
        samples=iter(rows(corpus/'sft'/f'{split}.jsonl.gz'))
        for case in rows(corpus/'splits'/f'{split}.jsonl.gz'):
            sid=case['scenario_id']
            if sid in seen or case['split']!=split:raise ValueError('Duplicate/wrong scenario split')
            seen.add(sid);count['scenarios']+=1;count['family:'+case['domain_family']]+=1;count['language:'+case['language']]+=1
            groups[split].add(case['entity_group'])
            if semantic:state=create_initial_state(schema)
            for number,turn in enumerate(case['turns'],1):
                e=next(samples);key=(sid,turn['turn_id']);keys[split].add(key)
                if e['metadata']['scenario_id']!=sid or e['metadata']['turn_id']!=turn['turn_id']:raise ValueError('SFT/scenario mismatch')
                if len(e['messages'])!=3 or [m['role'] for m in e['messages']]!=['system','user','assistant']:raise ValueError('Wrong messages')
                if e['messages'][0]['content']!=instructions:raise ValueError('Prompt drift')
                if strict_json(e['messages'][2]['content'])!=turn['gold_extraction']:raise ValueError('Gold drift')
                user=strict_json(e['messages'][1]['content'])
                if user['current_message']!=turn['message']:raise ValueError('Message drift')
                if len(user['slot_definitions'])!=79:raise ValueError('Schema cap regression')
                fingerprint=hashlib.sha256(dumps(e['messages'][:2]).encode()).hexdigest()
                if fingerprint in input_hashes[split]:raise ValueError('Duplicate model input')
                input_hashes[split].add(fingerprint)
                ids=[]
                for update in turn['gold_extraction']['updates']:
                    ids.append(update['slot_id']);positive[update['slot_id']]+=1;grounded+=1
                    if not update['evidence_text'].strip() or update['evidence_text'] not in turn['message']:raise ValueError('Ungrounded gold')
                    strict_json(update['candidate_value'])
                if len(ids)!=len(set(ids)) or set(ids)&set(turn['forbidden_slots']):raise ValueError('Forbidden/duplicate gold update')
                if semantic:
                    intervene(state,turn.get('state_intervention'))
                    if snapshot(state)!=turn['context_state'] or [t.user_message for t in state.turns]!=turn['prior_user_messages']:raise ValueError('Prior state drift '+str(key))
                    if build_v5_extractor_input(turn['message'],state,schema)!=e['messages'][1]['content']:raise ValueError('Prompt context drift '+str(key))
                    admitted=admit(e['messages'][2]['content'],turn['message'],wire)
                    state=apply_extraction(state,admitted,schema,number,turn['message'])
                    state.turns.append(DialogueTurn(turn_number=number,user_message=turn['message']))
                    if snapshot(state)!=turn['expected_state_after']:raise ValueError('Gold replay drift '+str(key))
                    if any(s.status.value=='invalid' for s in state.slots.values()):raise ValueError('Invalid gold slot')
                count['examples']+=1
        if next(samples,None) is not None:raise ValueError('Excess SFT examples')
        if count['scenarios']!=manifest['splits'][split]['scenarios'] or count['examples']!=manifest['splits'][split]['examples']:raise ValueError('Count mismatch')
        total=count['scenarios']
        for family,pct in [('weather',30),('economics',60),('combined',10)]:
            if count['family:'+family]*100!=total*pct:raise ValueError('Domain ratio drift')
        for language,pct in [('en',80),('ur',10),('mixed',10)]:
            if count['language:'+language]*100!=total*pct:raise ValueError('Language ratio drift')
        counts[split]=dict(count)
    if groups['train']&groups['validation'] or keys['train']&keys['validation'] or input_hashes['train']&input_hashes['validation']:raise ValueError('Cross-split overlap')
    return {'status':'pass','semantic_reducer_replay':semantic,'splits':counts,'positive_slot_count':len(positive),
            'positive_slots':dict(positive),'exact_grounded_updates':grounded,'cross_split_entity_groups':0,
            'cross_split_exact_model_inputs':0,'human_review_complete':False,'final_test':False,
            'manifest_sha256':sha(corpus/'manifest.json')}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,default=CORPUS);p.add_argument('--fpy',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.fpy:use_fpy(a.fpy)
    result=verify(a.corpus,semantic=bool(a.fpy))
    if a.output:atomic(a.output,result)
    print(dumps(result))
