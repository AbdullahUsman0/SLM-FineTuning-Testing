"""Post-hoc numeric-only sensitivity. Frozen scores and responses stay unchanged."""
from collections import Counter
from decimal import Decimal
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from common import atomic,bootstrap,read,rows,sha

def canonical(v):
    if v is None:return ('null',)
    if isinstance(v,bool):return ('boolean',v)
    if isinstance(v,(int,float)):
        assert not isinstance(v,float) or math.isfinite(v)
        return ('number',Decimal(str(v)).normalize())
    if isinstance(v,str):return ('string',v)
    if isinstance(v,list):return ('list',tuple(canonical(x) for x in v))
    if isinstance(v,dict):return ('object',tuple((k,canonical(value)) for k,value in sorted(v.items())))
    raise TypeError(type(v).__name__)

def main():
    p=bootstrap();cfg=read(Path(r'D:\SLM\active-english-prospective-evaluation.json'));run=Path(cfg['run'])
    publication=Path(r'D:\SLM\FYP-publications')/run.name/'research-checkpoints'/('english-v8-openai-'+run.name.removeprefix('english-v8-openai-prospective-'))
    assert (publication/'POSTHOC_AUDIT_PLAN.md').exists()
    labels={(c['scenario_id'],t['turn_id']):t for c in rows(ROOT/'label-review-r2/CHECKED_CASES.jsonl') for t in c['turns']}
    def tuples(obj):
        result=[]
        for slot,value,status in p.tuples(obj):
            try:val=canonical(json.loads(value))
            except (ValueError,TypeError):val=('invalid-wire-value',value)
            result.append((slot,val,status))
        return Counter(result)
    grouped={};per_call=[]
    for path in (run/'scored-calls').glob('*.json'):
        r=read(path);turn=labels[(r['conversation_id'],r['key'][-1])]
        try:obj=json.loads(r['raw_output'])
        except ValueError:obj=None
        gold=tuples(turn['gold_extraction']);pred=tuples(obj)
        tp=sum((gold&pred).values()) if r['raw_contract_admitted'] else 0
        counts=[tp,sum(pred.values())-tp,sum(gold.values())-tp]
        item={'key':r['key'],'primary_counts':r['confusion'],'numeric_only_counts':counts,'additional_TP':tp-r['confusion'][0]}
        assert item['additional_TP']>=0
        grouped.setdefault((r['arm'],r['track']),[]).append(item);per_call.append(item)
    output={}
    for (arm,track),items in sorted(grouped.items()):
        strict=[sum(r['primary_counts'][j] for r in items) for j in range(3)]
        sensitivity=[sum(r['numeric_only_counts'][j] for r in items) for j in range(3)]
        def metric(c):
            tp,fp,fn=c;return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
        output.setdefault(arm,{})[track]={'calls':len(items),'frozen_primary':metric(strict),'posthoc_numeric_only_sensitivity':metric(sensitivity),'additional_TP':sum(r['additional_TP'] for r in items),'affected_calls':sum(r['additional_TP']>0 for r in items)}
    atomic(publication/'POSTHOC_NUMERIC_REPRESENTATION_AUDIT.json',{'status':'posthoc_sensitivity_not_primary','plan_sha256':sha(publication/'POSTHOC_AUDIT_PLAN.md'),'no_new_model_calls':True,'original_primary_and_labels_unchanged':True,'scope':'Numeric JSON equivalence only, booleans distinct. No string/status/slot semantic repair.','results':output})
    atomic(run/'POSTHOC_NUMERIC_REPRESENTATION_AUDIT.json',read(publication/'POSTHOC_NUMERIC_REPRESENTATION_AUDIT.json'))
    (run/'numeric-sensitivity-per-call.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in per_call),encoding='utf-8')
    print(json.dumps(output))

if __name__=='__main__':main()
