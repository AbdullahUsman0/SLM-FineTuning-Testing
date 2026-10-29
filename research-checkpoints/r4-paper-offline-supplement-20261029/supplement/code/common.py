"""Shared offline supplement helpers; published inputs are read-only."""
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'workspace.json').read_bytes())
E=Path(C['evidence']);OUT=ROOT/'deliverables';RUNTIME=ROOT/'runtime'
sys.path[:0]=[str(RUNTIME/'offline-deps'),str(RUNTIME/'evaluator')]
import pipeline as p
p.configure(RUNTIME/'fpy',Path(C['repo']))
from forecasting_assistant.domain.schema import load_schema


def load():
    cases=json.loads((E/'main-cases.json').read_bytes())
    rows=[json.loads(line) for line in gzip.decompress((E/'main-scored-records.jsonl.gz').read_bytes()).splitlines()]
    return cases,rows,load_schema()


def dump(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,default=str)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False,default=str)+'\n',encoding='utf-8')
def csv_write(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)


def counts_to_prf(counts):
    tp,fp,fn=map(int,counts)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def row_key(row):return tuple(row['key'][1:])
def slot_state(values):
    return {sid:{'value':deepcopy(v['value']),'status':v['status'],'confirmed_by_user':v.get('confirmed_by_user',False)} for sid,v in values.items() if v.get('value') is not None}


def prior_states(cases,rows):
    """Reconstruct recorded priors from gold component or preceding actual rollout.

    Only the same declared mutation operations as frozen p.intervention are used;
    no model output is replayed under a changed context.
    """
    index={row_key(r):r for r in rows};result={}
    for arm in ('v8','openai'):
        for track in ('component','rollout'):
            for case in cases:
                previous={}
                for turn in case['turns']:
                    key=(arm,track,case['scenario_id'],turn['turn_id']);row=index[key]
                    prior=slot_state(turn['context_state']) if track=='component' else deepcopy(previous)
                    if track=='rollout':
                        for m in (turn.get('state_intervention') or {}).get('mutations',[]):
                            if m['kind']=='omitted_prior_value':prior.pop(m['slot'],None)
                            elif m['kind']=='wrong_prior_value':
                                item=prior.setdefault(m['slot'],{'value':None,'status':'unmentioned','confirmed_by_user':False});item['value']=deepcopy(m['value'])
                                if item['value'] is None:prior.pop(m['slot'],None)
                            else:raise ValueError('Unknown declared intervention')
                    payload=json.loads(row['current_input'])
                    result[key]={'slots':prior,'intent':payload['intent_policy']['value'],'pending_slot':payload.get('pending_slot'),'assistant_question_context':payload.get('latest_assistant_question'),'scope':'supplied gold component prior' if track=='component' else 'actual accepted rollout prior after declared intervention'}
                    assert payload.get('latest_assistant_question') is None
                    previous=deepcopy(row['pipeline_state'])
    return result


def mismatches(actual,expected):return sorted(sid for sid in set(actual)|set(expected) if actual.get(sid)!=expected.get(sid))
