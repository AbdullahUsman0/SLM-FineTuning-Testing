"""Build v8 from byte-verified public r8 development inputs, plus targeted contrasts."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import argparse,gzip,hashlib,json,sys,zipfile
from pathlib import Path
from data_tools import ROOT,CORPUS,VERSION,PROMPT_VERSION,FPY_REVISION,BASE_REVISION,dumps,sha,rows,use_fpy,snapshot,restore,verify,atomic

SOURCE=ROOT/'corpus-v7/v7-20261009-r8-dev'
PARENT_MANIFEST='4e59ba3eafa689325f94b108719da02b955e5c4f5de5571f208053e4c905da72'
PARENT_ZIP_SHA256='9f8424895c4f7591b65a5f1aaa7e87104f8035671c272177cd8b6a5bb8bba22b'
EXTRA=(
 '\nFinal development policy v8-grounded-role-isolation-1: observations/sampling belong only to frequency; '
 'forecast issuance belongs to prediction_frequency; fetching new source observations belongs to refresh_frequency; '
 'model refitting belongs to retraining_frequency. Historical coverage belongs to expected_history_length only '
 'when an explicit amount of past history is stated. Never infer historical coverage from a cadence or future horizon. '
 'Refreshing a source does not change its source_mode. A rejected suggestion or an unknown answer supplies no value. '
 'An explicit correction updates only the specified fields. Leave unrelated accepted slots unchanged. '
 'Copy evidence exactly, including its original whitespace; do not rewrite it. '
 'These are extraction instructions, not instructions to generate a numerical forecast.'
)
CATEGORIES=('cadence_role_isolation','refresh_without_access_change','unknown_without_defaults','correction_preserves_state')

def write_gz(path,items):
    path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
    with path.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as z:
        for item in items:
            raw=(dumps(item)+'\n').encode();h.update(raw);z.write(raw);n+=1
    return {'sha256':sha(path),'raw_sha256':h.hexdigest(),'records':n,'bytes':path.stat().st_size}

def turn(case,n,state,message,facts,behavior,correction=False,forbidden=()):
    from forecasting_assistant.domain.models import ExtractorResult,DialogueTurn
    from forecasting_assistant.application.state_reducer import apply_extraction
    from forecasting_assistant.domain.schema import load_schema
    updates=[]
    for slot,value,evidence,*status in facts:
        assert evidence in message
        updates.append({'slot_id':slot,'candidate_value':dumps(value),'status':status[0] if status else 'provided','confidence':1.,'evidence_text':evidence})
    gold={'intent':'create_forecast','intent_confidence':1.,'correction_detected':correction,'unsupported_claims':[],'updates':updates}
    before=snapshot(state)
    fact_ledger=[];spans=[]
    for u in updates:
        fid=f"{case['scenario_id']}/t{n}/{u['slot_id']}";start=message.index(u['evidence_text'])
        fact_ledger.append({'fact_id':fid,'slot_id':u['slot_id'],'value':json.loads(u['candidate_value']),'status':u['status']})
        spans.append({'fact_id':fid,'slot_id':u['slot_id'],'start':start,'end':start+len(u['evidence_text']),'text':u['evidence_text']})
    t={'turn_id':f't{n}','parent_id':case['scenario_id'] if n==1 else case['scenario_id']+f'/t{n-1}',
       'message':message,'gold_extraction':gold,'context_state':before,'context_slots':{k:v['value'] for k,v in before.items()},
       'established_intent':state.intent.value,'prior_user_messages':[x.user_message for x in state.turns],
       'behavior':behavior,'forbidden_slots':list(forbidden),'state_intervention':None,'language':case['language'],
       'reviewer_status':'machine_validated_pending_human','facts':fact_ledger,'evidence_spans':spans}
    state=apply_extraction(state,ExtractorResult.model_validate(gold),load_schema(),n,message)
    state.turns.append(DialogueTurn(turn_number=n,user_message=message));t['expected_state_after']=snapshot(state)
    return t,state

def focused_case(domain,profile,split,index,language):
    from forecasting_assistant.domain.schema import create_initial_state,load_schema
    category=CATEGORIES[index//10] if split=='train' else CATEGORIES[index%4]
    sid=f'{VERSION}/focused/{domain}/{split}-{index:03d}'
    group=f'{VERSION}/{domain}/{split}/focused-family-{index//10:02d}'
    case={'scenario_id':sid,'split':split,'category':category,'domain':domain,'domain_family':profile['domain_family'],
      'language':language,'entity_group':group,'contrast_group':group,'cluster_id':group,'corpus_version':VERSION,
      'prompt_version':PROMPT_VERSION,'schema_version':'1.0.0','origin':'new_failure_targeted_controlled_synthetic',
      'reviewer_status':'machine_validated_pending_human','state_interventions_are_synthetic':False}
    target=profile['gold_final_slots']['target_description'].rsplit(' for ',1)[0]
    target=f'{target} for fictional {split} station {domain} {index+7100}'
    unit=profile['gold_final_slots']['target_unit'];h=3+index%17;updated=h+5
    ur=language=='ur';mixed=language=='mixed';variant=index%5
    head=(f'Mujhe {target} ka forecast chahiye.' if ur else
          [f'I want a forecast of {target}.',f'Please predict {target}.',f'The quantity I need forecast is {target}.',f'Can you forecast {target}?',f'We need predictions of {target}.'][variant] if split=='train' else f'My forecasting brief concerns {target}.')
    horizon=(f'Forecast horizon {h} din hai.' if ur else f'Predict the next {h} days.')
    if mixed:horizon=f'Horizon aglay {h} din ka rakhein.'
    unit_clause=f'Target ki unit {unit} hai.' if ur else f'The measurement unit is {unit}.'
    mode='api' if index%2 else 'upload'
    mode_clause=('Input API ke zariye lein.' if mode=='api' else 'Main data upload karunga.') if ur else ('I will supply the input through an API.' if mode=='api' else 'I will upload the input data.')
    message=' '.join([head,horizon,unit_clause,mode_clause])
    facts=[('target_description',target,head),('problem_statement',target,head),('forecast_horizon',{'periods':h,'unit':'day'},horizon),('target_unit',unit,unit_clause),('source_mode',mode,mode_clause)]
    state=create_initial_state(load_schema());turns=[]
    t,state=turn(case,1,state,message,facts,'focused_initial');turns.append(t)
    n=2+index%4;cad=1+index%3;safe_reference=None
    if category=='cadence_role_isolation':
        if ur:
            a=f'Observed records har {n} ghantay collect hotay hain.';b=f'Naya forecast har {cad} din issue karein.';c=f'Source ka data har {n+3} ghantay dobara fetch karein.'
        elif mixed:
            a=f'Observations har {n} hours record hoti hain.';b=f'Issue a new forecast every {cad} days.';c=f'Source ko har {n+3} ghantay refresh karein.'
        elif split=='train':
            a=f'One observation is recorded every {n} hours.';b=f'Issue a new forecast every {cad} days.';c=f'Fetch source updates every {n+3} hours.'
        else:
            a=f'The spacing between observations is {n} hours.';b=f'The forecast issuance interval is {cad} days.';c=f'Source ingestion repeats at {n+3}-hour intervals.'
        message=' '.join([a,b,c]);facts=[('frequency',{'periods':n,'unit':'hour'},a),('prediction_frequency',{'periods':cad,'unit':'day'},b),('refresh_frequency',{'periods':n+3,'unit':'hour'},c)]
        forbidden=['expected_history_length','history_start','history_end','source_mode','retraining_frequency']
        if index%2==0:
            years=2+index%5
            past=(f'Past observations ki history {years} saal ki hai.' if ur or mixed else
                  f'The available past observations cover {years} years.' if split=='train' else
                  f'There are {years} years of historical observations available.')
            message+=' '+past;facts.append(('expected_history_length',{'periods':years,'unit':'year'},past));forbidden.remove('expected_history_length')
    elif category=='refresh_without_access_change':
        message=(f'Source records har {n} ghantay refresh karein. Access ka pehle wala tareeqa wahi rahega.' if ur else
                 f'Source ko har {n} hours refresh karein; keep the existing access method.' if mixed else
                 f'Fetch fresh observations from the source every {n} hours; keep the existing access method.' if split=='train' else
                 f'The source refresh interval is {n} hours. Its access method has not changed.')
        evidence=message.split(';')[0] if ';' in message else message.split('. ')[0]+'.'
        facts=[('refresh_frequency',{'periods':n,'unit':'hour'},evidence)]
        forbidden=['source_mode','frequency','prediction_frequency','expected_history_length']
        if mode=='api':
            # The pinned application's redactor permits a complete secret://
            # reference, but scrubs it when embedded inside a longer sentence.
            # Supply this invented reference as its own later user turn.
            safe_reference=f'secret://synthetic-{split}/{domain}/{index}'
    elif category=='unknown_without_defaults':
        message=('Timezone aur past history ki length abhi nahi pata. UTC ya aik saal khud se mat bharein.' if ur else
                 'Timezone aur historical coverage nahi pata; do not assume UTC or one year.' if mixed else
                 'The timezone and historical coverage have not been decided. Do not assume UTC or one year.' if split=='train' else
                 'I cannot yet specify a timezone or how much past history is available. Keep both unspecified.')
        facts=[];forbidden=['timezone','expected_history_length','history_start','history_end']
    else:
        message=('Abhi tak ki requirements theek hain. Unhein repeat ya change na karein.' if ur else
                 'Existing requirements theek hain; do not repeat their slot updates.' if mixed else
                 'Keep the requirements already recorded. I have no new values to add.' if split=='train' else
                 'Nothing in the recorded brief has changed; there are no additional requirements in this message.')
        facts=[];forbidden=['target_description','problem_statement','forecast_horizon','source_mode','target_unit']
    t,state=turn(case,2,state,message,facts,category,forbidden=forbidden);turns.append(t)
    message=(f'Sirf pehla horizon badal kar {updated} din kar dein. Baqi requirements na badlein.' if ur else
             f'Only change the horizon to {updated} days; baqi settings wahi rakhein.' if mixed else
             f'Replace the earlier horizon with {updated} days. Leave everything else unchanged.' if split=='train' else
             f'Correction to the existing brief: the future horizon must be {updated} days; preserve all other fields.')
    t,state=turn(case,3,state,message,[('forecast_horizon',{'periods':updated,'unit':'day'},message)],'focused_horizon_correction',True,
                 ['frequency','refresh_frequency','source_mode','target_description','problem_statement','target_unit','expected_history_length']);turns.append(t)
    has_auth=state.slots['authentication_reference'].value is not None
    message=('Seasonal cycle lengths aur access reference nahi pata. Abhi unko fill na karein.' if ur else
             'Seasonal periods aur access reference unknown hain; please do not fill them.' if mixed else
             'I do not know the seasonal cycle lengths or the authorized access reference. Leave them unspecified.' if split=='train' else
             'No seasonal cycle lengths or authentication reference have been supplied, so I cannot give those values yet.')
    if has_auth:
        message=('Seasonal cycle lengths aur interval coverage levels nahi pata. Pehle di hui requirements na badlein.' if ur or mixed else
                 'Seasonal cycle lengths and interval coverage levels are unknown. Keep the existing requirements unchanged.')
    t,state=turn(case,4,state,message,[],'focused_unknown_preserve',forbidden=['seasonal_periods','authentication_reference','known_seasonality']);turns.append(t)
    if safe_reference:
        t,state=turn(case,5,state,safe_reference,[('authentication_reference',safe_reference,safe_reference)],'safe_stored_reference',
                     forbidden=['source_mode','source_reference','frequency','forecast_horizon']);turns.append(t)
    case.update(turns=turns,gold_final_slots={k:v['value'] for k,v in snapshot(state).items()},gold_final_statuses={k:v['status'] for k,v in snapshot(state).items()},gold_intent=state.intent.value)
    return case

def sft(case,instructions,schema):
    from local_slm_lab.v5_prompts import build_v5_extractor_input
    for t in case['turns']:
        yield {'messages':[{'role':'system','content':instructions},{'role':'user','content':build_v5_extractor_input(t['message'],restore(t,schema),schema)},
                 {'role':'assistant','content':dumps(t['gold_extraction'])}],
              'metadata':{k:case[k] for k in ('scenario_id','split','category','domain','domain_family','language','entity_group','contrast_group','reviewer_status')}
               |{'corpus_version':VERSION,'task':'extract','turn_id':t['turn_id'],'behavior':t['behavior'],'state_intervention':t.get('state_intervention'),'source_corpus_version':case.get('source_corpus_version')}}

def build(output=CORPUS):
    global SOURCE
    from forecasting_assistant.domain.schema import load_schema
    output=Path(output)
    if output.exists():raise FileExistsError('Never overwrite a frozen corpus; choose a new version')
    schema_source=ROOT/'evaluation/comparison-policy-20261006/extractor-result.schema.json'
    if not SOURCE.exists() or not schema_source.exists():
        bundle=ROOT/'research-checkpoints/v7-approved-policy-20261009/v7-approved-policy-development-20261009-r8.zip'
        if sha(bundle)!=PARENT_ZIP_SHA256:raise ValueError('Published r8 parent bundle checksum mismatch')
        destination=ROOT/'runtime/v8-parent-inputs-20261018'
        prefix='corpus-v7/v7-20261009-r8-dev/'
        # Read only public development members; never extract final assignments/labels.
        members=[prefix+n for n in ['manifest.json','schema.json','splits/train.jsonl.gz','splits/validation.jsonl.gz','sft/train.jsonl.gz','sft/validation.jsonl.gz']]
        members+=['evaluation/comparison-policy-20261006/extractor-result.schema.json']
        with zipfile.ZipFile(bundle) as z:
            for name in members:
                p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);content=z.read(name)
                if p.exists() and p.read_bytes()!=content:raise ValueError('Cached parent input changed')
                if not p.exists():p.write_bytes(content)
        SOURCE=destination/prefix;schema_source=destination/members[-1]
    if sha(SOURCE/'manifest.json')!=PARENT_MANIFEST:raise ValueError('Published r8 manifest identity mismatch')
    original_manifest=json.loads((SOURCE/'manifest.json').read_bytes())
    sources={}
    for name in ['splits/train.jsonl.gz','splits/validation.jsonl.gz','sft/train.jsonl.gz','sft/validation.jsonl.gz']:
        if sha(SOURCE/name)!=original_manifest['files'][name]['sha256']:raise ValueError('r8 source checksum mismatch')
        sources[name]={'sha256':sha(SOURCE/name),'bytes':(SOURCE/name).stat().st_size}
    base={s:list(rows(SOURCE/'splits'/f'{s}.jsonl.gz')) for s in ('train','validation')}
    first=next(rows(SOURCE/'sft/train.jsonl.gz'))
    instructions=first['messages'][0]['content']+EXTRA
    output.mkdir(parents=True)
    (output/'extractor-instructions.txt').write_text(instructions,encoding='utf-8',newline='\n')
    # Carry the published closed wire schema without depending on untracked comparison code.
    (output/'response-schema.json').write_bytes(schema_source.read_bytes())
    (output/'schema.json').write_bytes((SOURCE/'schema.json').read_bytes())
    profiles={c['domain']:c for c in base['train']}
    schema=load_schema();files={};split_counts={};review=[]
    for split in ('train','validation'):
        cases=[]
        for c in base[split]:
            c=deepcopy(c);c['source_corpus_version']=c['corpus_version'];c['corpus_version']=VERSION;c['prompt_version']=PROMPT_VERSION;cases.append(c)
        for domain,profile in sorted(profiles.items()):
            count=40 if split=='train' else 10
            for block in range(count//10):
                order=sorted(range(10),key=lambda j:hashlib.sha256(dumps([VERSION,domain,split,block,'language',j]).encode()).hexdigest())
                language={j:'en' if pos<8 else 'ur' if pos==8 else 'mixed' for pos,j in enumerate(order)}
                for j in range(10):cases.append(focused_case(domain,profile,split,block*10+j,language[j]))
        cases.sort(key=lambda c:c['scenario_id'])
        files[f'splits/{split}.jsonl.gz']=write_gz(output/'splits'/f'{split}.jsonl.gz',cases)
        files[f'sft/{split}.jsonl.gz']=write_gz(output/'sft'/f'{split}.jsonl.gz',(e for c in cases for e in sft(c,instructions,schema)))
        split_counts[split]={'scenarios':len(cases),'examples':sum(len(c['turns']) for c in cases),'base_r8_scenarios':len(base[split]),'new_focused_scenarios':len(cases)-len(base[split])}
        # Review material is development-only, deliberately stratified; it is not an approval ledger.
        for domain in sorted(profiles):
            selected=[c for c in cases if c['domain']==domain and c['origin']=='new_failure_targeted_controlled_synthetic']
            review.extend(selected[:4])
    files['review-sample.jsonl.gz']=write_gz(output/'review-sample.jsonl.gz',review)
    for name in ['schema.json','response-schema.json','extractor-instructions.txt']:files[name]={'sha256':sha(output/name),'bytes':(output/name).stat().st_size}
    manifest={'corpus_version':VERSION,'schema_version':'1.0.0','prompt_version':PROMPT_VERSION,'base_model':'Qwen/Qwen3.5-2B','base_revision':BASE_REVISION,
      'fpy_revision':FPY_REVISION,'source_r8_manifest_sha256':sha(SOURCE/'manifest.json'),'source_files':sources,'splits':split_counts,'files':files,
      'construction':'Full public r8 development trajectories with unchanged gold; new system guidance plus 1000 targeted four/five-turn conversations; secret://synthetic references are invented identifiers, not credentials',
      'training_authorized':True,'authorization_file':'training/final_round/TRAINING_AUTHORIZATION.json',
      'human_review_complete':False,'status':'experimental_synthetic_development_machine_checks_required',
      'domain_scenario_ratio':{'weather':.3,'economics':.6,'combined':.1},'language_scenario_ratio':{'en':.8,'ur':.1,'mixed':.1},
      'final_test_included':False,'final_labels_accessed':False,'limitations':['Controlled synthetic wording and shared templates','Previous validation was inspected during development','Language ratios concern conversations, not tokens or examples','Independent human annotation is still pending','No claim of zero semantic leakage or a model winner']}
    atomic(output/'manifest.json',manifest);atomic(output/'manifest.sha256.json',{'manifest_sha256':sha(output/'manifest.json')})
    result=verify(output,semantic=True);atomic(output/'validation-report.json',result)
    print(dumps({'manifest_sha256':result['manifest_sha256'],'splits':split_counts,'status':result['status']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fpy',type=Path,required=True);p.add_argument('--output',type=Path,default=CORPUS)
    a=p.parse_args();use_fpy(a.fpy);build(a.output)
