"""Freeze a prediction-independent development selection and audit old gold."""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from controls import Ledger, atomic, digest, dumps, freeze, sha
import pipeline as p

VERSION = 'current-pipeline-dev20-20261023-r1'
FPY = '9bc2bd18c495514efa79a4371be1d706b9fc3824'
BASE = '15852e8c16360a2fea060d615a32b45270f8a8fc'
SHARED_LEDGER = Path(r'D:\SLM\FYP-experiment-control\current-pipeline-v8-openai-20261023\budget.sqlite3')
WEIGHTS = {'adapter_model.safetensors':'c83e0f1a458125970eb70ff360ebead02618f2f48b5bbb59e2ca91a5b7c4c0df','adapter_config.json':'a1ce14418c5c22375ccfd773338ef1b09117ec9b9e72db034c71ade886b4664c'}


def adjudicate(cases):
    """User-approved application-state status only; raw labels/selection untouched."""
    from copy import deepcopy
    revised=deepcopy(cases);changes=[]
    for case in revised:
        if case['scenario_id']!='v7-20261009-r8-dev/labor_employment/0178':continue
        for turn in case['turns']:
            for field in ('context_state','expected_state_after'):
                entry=turn[field].get('target_column')
                if entry and entry['value']=='true' and entry['status']=='provided':
                    entry['status']='invalid'
                    changes.append({'case':case['scenario_id'],'turn':turn['turn_id'],'field':field,'slot':'target_column','before_status':'provided','after_status':'invalid'})
        if case.get('gold_final_statuses',{}).get('target_column')=='provided':
            case['gold_final_statuses']['target_column']='invalid'
    return revised,changes


def select(cases):
    groups = {}
    for case in cases:
        groups.setdefault(case['domain'], []).append(case)
    domains = sorted(groups)
    assert len(domains) == 20
    for domain in domains:
        groups[domain].sort(key=lambda c:digest([VERSION,c['scenario_id']]))
    @lru_cache(None)
    def solve(index, en, ur, mixed):
        if min(en,ur,mixed) < 0:
            return None
        if index == len(domains):
            return () if (en,ur,mixed)==(0,0,0) else None
        for rank, case in enumerate(groups[domains[index]]):
            lang=case['language']
            suffix=solve(index+1,en-(lang=='en'),ur-(lang=='ur'),mixed-(lang=='mixed'))
            if suffix is not None:
                return (rank,)+suffix
        return None
    ranks=solve(0,16,2,2)
    if ranks is None:
        raise ValueError('Requested language/domain balance infeasible; needs a documented design amendment')
    main=[groups[domain][rank] for domain,rank in zip(domains,ranks)]
    chosen={c['scenario_id'] for c in main}
    # Separate deterministic pilot, no prediction/old score access or optimization.
    pilots=sorted((c for c in cases if c['scenario_id'] not in chosen),key=lambda c:digest([VERSION,'pilot',c['scenario_id']]))[:2]
    return sorted(main,key=lambda c:c['scenario_id']),pilots


def audit(cases, schema, wire):
    from forecasting_assistant.application.state_reducer import apply_extraction
    issues=[]; oracle=[]; support=Counter()
    for case in cases:
        for turn in case['turns']:
            key=[case['scenario_id'],turn['turn_id']]
            support.update(u['slot_id'] for u in turn['gold_extraction']['updates'] if u['slot_id']!='intent')
            try:
                state=p.restore(turn,schema)
                raw=dumps(turn['gold_extraction'])
                gold=p.admit(raw,wire)
                for update in gold.updates:
                    if not update.evidence_text.strip() or update.evidence_text not in turn['message']:
                        raise ValueError('Gold evidence not an exact current span')
                after=apply_extraction(state,gold,schema,len(state.turns)+1,turn['message'])
                expected=turn['expected_state_after'];actual=p.snapshot(after)
                invalid=[k for k,s in after.slots.items() if s.status.value=='invalid']
                unexpected_invalid=[k for k in invalid if expected.get(k,{}).get('status')!='invalid']
                if actual!=expected or unexpected_invalid:
                    slots=sorted(k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k))
                    issues.append({'key':key,'kind':'current_reducer_gold_contract_conflict','message':turn['message'],'changed_slots':slots,'invalid_slots':invalid,'expected':{k:expected.get(k) for k in slots},'current_reducer':{k:actual.get(k) for k in slots},'validation_errors':{k:after.slots[k].validation_errors for k in invalid},'review_status':'pending_independent_adjudication'})
                pipeline_after,provenance=p.run_pipeline(state,turn['message'],schema,{'raw_output':raw},wire)
                if p.snapshot(pipeline_after)!=expected:
                    changed=sorted(k for k in set(p.snapshot(pipeline_after))|set(expected) if p.snapshot(pipeline_after).get(k)!=expected.get(k))
                    oracle.append({'key':key,'kind':'pipeline_diff_even_with_gold_extraction','changed_slots':changed,'recovered_slot_ids':provenance['recovered_slot_ids'],'expected':{k:expected.get(k) for k in changed},'current_pipeline':{k:p.snapshot(pipeline_after).get(k) for k in changed}})
            except Exception as exc:
                issues.append({'key':key,'kind':'gold_audit_exception','error_type':type(exc).__name__,'detail':str(exc)[:350]})
    return {'conversations':len(cases),'turns':sum(len(c['turns']) for c in cases),'blocking_gold_issues':issues,'pipeline_oracle_findings':oracle,'nonintent_slot_support':dict(support),'independent_human_review_complete':False}


def prepare(run,repo,adapter):
    run=Path(run).resolve();repo=Path(repo).resolve();fpy=run/'fpy';bundle=run/'bundle'
    p.configure(fpy,repo)
    assert subprocess.check_output(['git','-C',str(fpy),'rev-parse','HEAD'],text=True).strip()==FPY
    prep=json.loads((bundle/'PREPARATION.json').read_bytes())
    for name,wanted in prep['working_files_LF_sha256'].items():
        assert __import__('hashlib').sha256((fpy/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==wanted,name
    for name,wanted in WEIGHTS.items():
        assert sha(adapter/name)==wanted,name
    sys.path.insert(0,str(repo/'training/final_round'))
    import evaluate_development as old
    main_source=old.select_cases(list(old.rows(old.CORPUS/'splits/validation.jsonl.gz')))
    main,pilots=select(main_source)
    original_main=main
    prior_run=Path(json.loads((run/'adjudication-authorization.json').read_bytes())['prior_run'])
    prior_ids=json.loads((prior_run/'selection.json').read_bytes())['main_ids']
    assert [c['scenario_id'] for c in main]==prior_ids,'Selection changed after adjudication'
    main,changes=adjudicate(main)
    assert changes,'Approved state adjudication was not applied'
    for old_case,new_case in zip(original_main,main):
        assert [t['gold_extraction'] for t in old_case['turns']]==[t['gold_extraction'] for t in new_case['turns']]
    atomic(run/'adjudication-changes.json',{'cohort':'current-pipeline-dev20-20261023-r2','changes':changes,'raw_gold_unchanged':True,'selected_ids_unchanged':True,'source_main_sha256':digest(original_main),'adjudicated_main_sha256':digest(main)})
    schema=__import__('forecasting_assistant.domain.schema',fromlist=['load_schema']).load_schema()
    wire=p.wire_schema()
    from forecasting_assistant.prompts.extractor import build_extractor_instructions
    selection={'version':VERSION,'rule':'per-domain digest order with deterministic backtracking for exactly 16 en/2 ur/2 mixed; no predictions used','main_ids':[c['scenario_id'] for c in main],'pilot_ids':[c['scenario_id'] for c in pilots],'main_case_sha256':digest(main),'pilot_case_sha256':digest(pilots),'languages':dict(Counter(c['language'] for c in main)),'families':dict(Counter(c['domain_family'] for c in main)),'source_100_sha256':digest(main_source)}
    freeze(run/'selection.json',selection);freeze(run/'main-cases.json',main);freeze(run/'pilot-cases.json',pilots)
    main_audit=audit(main,schema,wire);pilot_audit=audit(pilots,schema,wire);source_audit=audit(main_source,schema,wire)
    atomic(run/'label-audit.json',{'main':main_audit,'pilot':pilot_audit,'source100':source_audit})
    atomic(run/'annotation-review-queue.json',main_audit['blocking_gold_issues']+pilot_audit['blocking_gold_issues'])
    sources={str(path.relative_to(fpy)).replace('\\','/'):sha(path) for path in (fpy/'src').rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    code={path.name:sha(path) for path in Path(__file__).parent.glob('*.py')}
    provider_sources={name:sha(repo/name) for name in ('local_slm_lab/peft_provider.py','local_slm_lab/v5_provider.py','local_slm_lab/v5_prompts.py','local_slm_lab/slm_prompts.py')}
    versions={}
    for name in ['torch','transformers','peft','openai','pydantic','jsonschema','pytest','pytest-asyncio','tiktoken']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='unavailable'
    gold_gate=not(main_audit['blocking_gold_issues'] or pilot_audit['blocking_gold_issues'])
    protocol={'version':'current-pipeline-comparison-4','cohort':'current-pipeline-dev20-20261023-r3','dispatch_namespace':run.name,'interface_correction_sha256':sha(run/'interface-correction.json'),'interface_version':'common-explicit-contract-1','local_technical_gate_sha256':sha(run/'local-technical-gate.json'),'budget_start_sha256':sha(run/'budget-at-start.json'),'provider_source_sha256':provider_sources,'adjudication_authorization_sha256':sha(run/'adjudication-authorization.json'),'adjudication_changes_sha256':sha(run/'adjudication-changes.json'),'fpy_commit':FPY,'bundle_patch_sha256':sha(bundle/'fpy-local-fixes.patch'),'fpy_source_sha256':sources,'runner_source_sha256':code,'adapter':str(adapter),'adapter_sha256':WEIGHTS,'base_model':'Qwen/Qwen3.5-2B','base_revision':BASE,'hosted_model':'gpt-5-mini-2025-08-07','instructions':p.build_instructions(repo),'response_schema':wire,'selection_sha256':digest(selection),'main_case_sha256':digest(main),'pilot_case_sha256':digest(pilots),'local_generation':{'max_new_tokens':1024,'thinking':False,'sampling':False,'seed':42,'dtype':'bfloat16'},'hosted_generation':{'max_output_tokens':2048,'reasoning_effort':'minimal','max_retries':0,'store':False},'fixed_script_assistant_context':None,'question_generation':'static local renderer only; not adaptive interview quality','state_definition':'all non-null slots: exact value, status and confirmed_by_user; includes intent; excludes timestamps, confidence, evidence and turn IDs','raw_score':'exact non-intent slot/value/status after exactly one wire decode; duplicate-preserving multiset; malformed/capped/refused contract yields zero TP, readable predictions FP, all gold FN; never credits recovery','application_score':'actual frozen ElicitationEngine, model records replayed once; recovery and normalizations recorded separately','gold_contract_gate_passed':gold_gate,'scope':'inspected public synthetic development, not independent final test','human_review_complete':False,'uncertainty_analysis':'descriptive only; 20 related synthetic conversations, no independence/significance/equivalence claims','shared_budget_ledger':str(SHARED_LEDGER),'rates':{'input_USD_per_million':0.25,'cached_input_USD_per_million':0.025,'output_USD_per_million':2.0,'discount_used_in_guard':False,'official_source':'https://developers.openai.com/api/docs/models/gpt-5-mini','verified_date':'2026-10-27'},'runtime':{'python':platform.python_version(),'packages':versions,'fpy_import_root':str(fpy/'src'),'evaluation_Python':sys.executable},'planned_main_calls_per_arm_track':sum(len(c['turns']) for c in main),'planned_pilot_calls_per_arm_track':sum(len(c['turns']) for c in pilots)}
    freeze(run/'protocol.json',protocol)
    Ledger(SHARED_LEDGER).db.close()
    atomic(run/'budget-status.json',Ledger(SHARED_LEDGER).summary())
    print(dumps({'gold_gate_passed':gold_gate,'main_blocking_issues':len(main_audit['blocking_gold_issues']),'pilot_blocking_issues':len(pilot_audit['blocking_gold_issues']),'main_pipeline_oracle_findings':len(main_audit['pipeline_oracle_findings']),'selection':selection,'planned_main_calls_per_arm_track':protocol['planned_main_calls_per_arm_track']}))
    return protocol


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--adapter',type=Path,required=True);args=parser.parse_args()
    prepare(args.run,args.repo,args.adapter)
