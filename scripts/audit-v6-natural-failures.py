"""Agent audit of existing development cases; strict scores remain immutable."""
from collections import Counter
from copy import deepcopy
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_slm_lab.v5_eval import digest, dumps, load_cases, load_schema, now, prf, sha256, tuples, build_call_plan, _snapshot_value
from forecasting_assistant.application.normalization import normalize_value
from local_slm_lab.v5_provider import strict_json_object, validate_raw_output
from forecasting_assistant.domain.models import ExtractorResult
from local_slm_lab.v5_prompts import build_v5_extractor_instructions
from local_slm_lab.v6_prompt_revision import build_revised_instructions

SPEC=importlib.util.spec_from_file_location('matched_analysis',ROOT/'scripts/analyze-v6-matched.py')
analysis=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(analysis)
SOURCE=ROOT/'research-checkpoints/v6-matched-evaluation-20261005/results'
TARGET=ROOT/'evaluation/v6-prompt-audit-20261006'

REVIEW_NOTES={
    'weather/01':'Unit, observation spacing and initial horizon are separate explicit facts. CSV is the canonical lowercase enum; uppercase emission is normalization-only.',
    'weather/02':'Use accumulated rainfall as target, millimetres as unit; hourly spacing and 36-hour horizon are explicit. Rain is introductory context, not a second target update.',
    'weather/03':'Upload and parquet are explicit. Every hour denotes observation spacing, 12 hours denotes horizon.',
    'weather/04':'humidity_pct and observation_date are explicitly distinct target and timestamp columns.',
    'weather/05':'Three-hour observation spacing must not be confused with the 18-hour horizon.',
    'weather/06':'Six-hour spacing and two-day horizon use different units. Hectopascals is the explicit unit.',
    'weather/07':'Two-day refresh belongs to prediction_frequency, independently of daily data and eight-day horizon.',
    'weather/08':'Uploaded file explicitly supports upload mode and xlsx; hours measures sunshine, not observation spacing.',
    'inflation/01':'Target is year-over-year inflation, explicitly excluding the CPI index. Percent is unit; five months is horizon.',
    'inflation/02':'Keep index base 2015=100 within the stated unit. It does not imply a target column, calendar or timezone.',
    'inflation/03':'Monthly spacing and three-month horizon are distinct; csv is explicitly stated.',
    'inflation/04':'food_inflation and month_end are distinct explicit columns. Percent is separate from the target description.',
    'inflation/05':'Quarterly spacing maps to one quarter; two quarters is horizon, not two months.',
    'inflation/06':'Monthly spacing and one-year horizon are independent. Upload and parquet are explicit.',
    'inflation/07':'Annual average inflation is the target. One-year spacing, three-year horizon and one-year refresh are three separate facts.',
    'inflation/08':'Uploaded xlsx is explicit. Monthly spacing and seven-month horizon are separate.',
    'stocks/01':'Keep NOVA and adjusted closing price in the verbatim target description; USD per share is the unit.',
    'stocks/02':'Without adjusting corporate actions qualifies raw closing price and should be preserved. Shorter descriptions may be semantically close but are not certified equivalent here.',
    'stocks/03':'Including dividends qualifies total share return. Upload mode and csv are explicit; percent is unit.',
    'stocks/04':'volume_shares is the explicit target column, trade_date is the timestamp column; shares is the unit.',
    'stocks/05':'Dividend yield is the target, percent the unit; monthly data and four-month horizon are explicit.',
    'stocks/06':'EUR per share must not be merged with frequency or horizon; parquet is explicitly stated.',
    'stocks/07':'Keep log return distinct from price. Three-day refresh is independent of daily data and twelve-day horizon.',
    'stocks/08':'ETF trading volume is the target, shares the unit. Upload mode and xlsx are explicitly requested.',
    'crypto/01':'Spot price, not futures price, is explicitly specified. USD per BTC is the unit; no timezone is given.',
    'crypto/02':'ETH trading volume is target; ETH is unit, two hours is observation spacing, fourteen hours is horizon.',
    'crypto/03':'Percent per 8 hours is the funding-rate unit. Eight-hour data spacing is a separate fact. Upload and csv are explicit.',
    'crypto/04':'open_contracts and event_time are explicitly distinct target and timestamp columns; contracts is unit.',
    'crypto/05':'Percent per year is the volatility unit, not a yearly observation frequency; observations are daily.',
    'crypto/06':'Futures-minus-spot basis and basis points are explicit target/unit. One-hour spacing and two-day horizon differ.',
    'crypto/07':'Six-hour refresh is distinct from hourly data and eighteen-hour horizon. Do not change log return to price.',
    'crypto/08':'Millions of USD is the full unit. Upload and JSON format are explicitly stated; do not invent columns.',
}

def normalized_tuple(update,schema):
    slot,value,status=update
    decoded=json.loads(value)
    try:
        normalized=normalize_value(schema.get(slot),decoded)
    except (KeyError,TypeError,ValueError):
        normalized=decoded
    # Keep normalized dates serializable using the existing state-snapshot policy.
    # This changes representation only; schema/status gates and FP counts remain.
    return slot,dumps(_snapshot_value(normalized)),status

def normalized_counts(records,schema):
    total=[0,0,0]
    for r in records:
        gold=Counter(normalized_tuple(u,schema) for u in tuples(r['expected']) if u[0]!='intent')
        observed=Counter(normalized_tuple(u,schema) for u in tuples(r.get('emitted')) if u[0]!='intent')
        tp=sum((gold&observed).values()) if r['prediction_valid'] else 0
        for i,n in enumerate((tp,sum(observed.values())-tp,sum(gold.values())-tp)):
            total[i]+=n
    return prf(*total)

def normalized_metrics(records,schema):
    corrections=[r for r in records if r.get('test_kind')=='correction']
    horizon_ok=local_ok=0
    for r in corrections:
        after=r.get('predicted_after')
        if not r['prediction_valid'] or after is None or not r['predicted']['correction_detected']:
            continue
        expected=r['gold_after']['slots']['forecast_horizon']
        if after['slots']['forecast_horizon']!=expected:
            continue
        horizon_ok+=1
        before=r['context']['state']
        others_unchanged=all(after['slots'][k]==v for k,v in before['slots'].items() if k not in ('forecast_horizon','intent'))
        local_ok+=others_unchanged and after['intent']==r['gold_after']['intent']
    return {'schema_gated_normalized_slot_value_status':normalized_counts(records,schema),
            'correction_horizon_after_reducer':{'numerator':horizon_ok,'denominator':len(corrections)},
            'correction_horizon_and_other_prior_slots_preserved':{'numerator':local_ok,'denominator':len(corrections)}}

def audit_records(records,schema):
    rows=[]
    for r in records:
        gold={u[0]:u for u in tuples(r['expected']) if u[0]!='intent'}
        observed={u[0]:u for u in tuples(r.get('emitted')) if u[0]!='intent'}
        for slot in sorted(gold.keys()|observed.keys()):
            g=gold.get(slot);p=observed.get(slot)
            if g is None:
                prior=r['context']['state']['slots'].get(slot)
                category='repeated_prior_fact' if prior and prior['value'] is not None and p[1]==dumps(prior['value']) else 'new_unmentioned_slot'
            elif p is None:category='omitted_slot'
            elif g==p:category='exact_content'
            elif normalized_tuple(g,schema)==normalized_tuple(p,schema):category='normalization_only'
            elif g[1]==p[1]:category='wrong_status'
            else:category='wrong_value_or_type'
            rows.append({'scenario_id':r['scenario_id'],'track':r['track'],'turn':r['key'][2],
                         'kind':r['test_kind'],'slot':slot,'category':category,
                         'prediction_valid':r['prediction_valid'],'message':r['context']['message'],
                         'expected_value':g[1] if g else None,'observed_value':p[1] if p else None,
                         'expected_status':g[2] if g else None,'observed_status':p[2] if p else None})
    return rows

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')

def prepare():
    TARGET.mkdir(parents=True,exist_ok=False)
    source=ROOT/'evaluation/v6-matched-20261005/natural.jsonl'
    cases,_=load_cases(source,split='validation')
    assert len(cases)==32
    schema=load_schema();plan=build_call_plan(cases,schema)
    (TARGET/'natural.jsonl').write_bytes(source.read_bytes())
    reviews=[]
    for case in cases:
        key='/'.join(case['scenario_id'].split('/')[-2:])
        first,correction,unknown=case['turns']
        facts={u['slot_id']:json.loads(u['candidate_value']) for u in first['gold_extraction']['updates']}
        assert facts['target_description'] in first['message'] and facts['target_unit'] in first['message']
        revised=json.loads(correction['gold_extraction']['updates'][0]['candidate_value'])
        assert revised['periods']==facts['forecast_horizon']['periods']+3 and revised['unit']==facts['forecast_horizon']['unit']
        assert unknown['gold_extraction']['updates']==[]
        reviews.append({'scenario_id':case['scenario_id'],'reviewer':'Codex agent; not an independent human reviewer',
                        'initial_facts':facts,'initial_label_review':REVIEW_NOTES[key],
                        'correction_review':'Explicit replacement horizon is annotated correctly; assess both local correction and full accumulated-state accuracy.',
                        'unknown_review':'Explicitly unknown/undecided information supplies no values. Empty updates are consistent with the extraction contract. This tests abstention, not generated clarification questions.',
                        'label_changes':[],'human_review_status':'pending'})
    write_json(TARGET/'label-review.json',reviews)
    summaries={};rows=[];wire=[]
    for track in ('natural_component','natural_rollout'):
        path=SOURCE/f'v6-{track}.json.gz'
        records=json.loads(gzip.decompress(path.read_bytes()))['records']
        assert len(records)==96
        audited=audit_records(records,schema);rows.extend(audited)
        summaries[track]={'strict':analysis.summarize(records),'supplementary_normalized':normalized_metrics(records,schema),
                          'content_categories_before_schema_gate':dict(Counter(row['category'] for row in audited))}
        for r in records:
            if r['prediction_valid']:continue
            issues=[]
            try:
                parsed=strict_json_object(r['raw_output'])
                ids=[u.get('slot_id') for u in parsed.get('updates',[])]
                if len(set(ids))!=len(ids):issues.append('duplicate_slot_updates')
                try:validate_raw_output(parsed,ExtractorResult)
                except Exception as error:issues.append('wire_contract_failure_'+type(error).__name__)
            except Exception as error:issues.append('strict_json_failure_'+type(error).__name__)
            wire.append({'track':track,'scenario_id':r['scenario_id'],'kind':r['test_kind'],
                         'issues':issues,'raw_output':r['raw_output']})
    with (TARGET/'baseline-slot-audit.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(TARGET/'baseline-audit.json',{'audited_utc':now(),'label_changes':0,'scenarios':32,
        'review_type':'Agent audit after seeing original outputs; independent human review pending',
        'normalization_policy':'Pinned application normalize_value; both gold and emitted values normalized, valid-schema gate retained. No free-text synonym matching, no output repair, no changed primary scores.',
        'limitations':['The 32 conversations are now development cases used to design the prompt, not an untouched test set.',
                       'Four domains use the same correction/unknown wording patterns; these are narrow diagnostics.',
                       'Unknown turns test abstention and state retention only, not the quality of generated questions.',
                       'Full-state correction accuracy includes initial-turn omissions. Local corrected horizon and preservation of other prior slots are separate.',
                       'Shortened free-text target descriptions may be semantically close; no independent semantic equivalence is certified.'],
        'tracks':summaries,'invalid_output_cases':wire})
    prompts={'original':build_v5_extractor_instructions(),'revised':build_revised_instructions()}
    for arm,text in prompts.items():(TARGET/f'{arm}-instructions.txt').write_text(text+'\n',encoding='utf-8',newline='\n')
    write_json(TARGET/'protocol.json',{'version':'v6-prompt-audit-20261006-1','frozen_before_new_inference_utc':now(),
        'natural_sha256':sha256(TARGET/'natural.jsonl'),'original_study_manifest_sha256':sha256(ROOT/'evaluation/v6-matched-20261005/study-manifest.json'),
        'base_model':'Qwen/Qwen3.5-2B','base_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc',
        'v6_adapter_sha256':{'adapter_model.safetensors':'e7add7b8b02d24615ed89067c7761c3985a4a6afb7c84ca2a72e3b06fd84e74d',
                             'adapter_config.json':'38f60de9678324e91fa882ed86c70d09e4a39b27ae28884374ddc3e6fece15ef'},
        'prompt_sha256':{arm:digest(text) for arm,text in prompts.items()},'scenarios':32,'calls_per_arm_track':96,'total_scored_calls':384,
        'tracks':['natural_component','natural_rollout'],'arms':['original','revised'],
        'primary':'Exact schema-gated slot/value/status F1 excluding intent; same unchanged gold labels',
        'supplementary':'Pinned application-normalized F1; local corrected-horizon/state-preservation metrics; raw content error categories',
        'contexts':'Gold prior state for component; own prior state for rollout, fixed user script. Both variants rerun in the same session.',
        'order':'Original/revised order alternates per scenario; serial batch 1 after equal unscored warmups',
        'decoding':{'seed':42,'do_sample':False,'max_new_tokens':1024,'enable_thinking':False,'dtype':'bfloat16','device':'cuda','torch_threads':4},
        'success_criteria':'Report paired F1 differences plus invented slot counts, JSON/schema validity, local corrections, unknown handling and latency. No winner from F1 alone; no further prompt tuning in this experiment.',
        'limitations':'Exploratory development comparison designed after original outputs; no independent held-out generalization claim.',
        'sealed_final_accessed':False,'training_started':False,'paid_apis_used':False,
        'ordered_keys_sha256':digest([c['key'] for c in plan]),
        'context_gold_sha256':digest([[c['key'],digest(c['context']),digest(c['expected'])] for c in plan])})
    (TARGET/'README.md').write_text('# V6 natural-label audit and frozen prompt comparison\n\nAgent review covers all 32 existing natural conversations and their 96 turns. No labels were changed. Independent human review remains pending. baseline-slot-audit.csv separates content omissions, wrong values, normalization-only differences and unexpected emissions; invalid outputs remain invalid. See baseline-audit.json for counts and limitations.\n\nprotocol.json freezes one original versus one revised prompt comparison: 384 new calls on the same v6 adapter, 32 conversations, gold-context component and own-state rollout tracks. This is a development experiment, not an independent test of generalization. The original study is preserved. No training, paid APIs or sealed labels.\n',encoding='utf-8',newline='\n')
    print(json.dumps({track:{'strict_f1':s['strict']['nonintent']['f1'],'normalized':s['supplementary_normalized'],
                            'categories':s['content_categories_before_schema_gate']} for track,s in summaries.items()},indent=2))

if __name__=='__main__':prepare()
