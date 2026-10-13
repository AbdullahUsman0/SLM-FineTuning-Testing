"""Prespecified primary/secondary metrics and grouped paired bootstrap."""
from collections import Counter,defaultdict
import statistics
import runtime as rt


def prf(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}


def rate(n,d):return {'numerator':n,'denominator':d,'rate':n/d if d else None}


def canonical(value):
    if type(value) is float and value.is_integer():return int(value)
    if isinstance(value,list):return [canonical(v) for v in value]
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    return value


def diagnostic(records,kind):
    totals=[0,0,0]
    for record in records:
        def items(value):
            result=[]
            for update in value.get('updates',[]):
                if update['slot_id']=='intent':continue
                key=(update['slot_id'],)
                if kind!='names':key+=(rt.dumps(canonical(update['candidate_value'])),)
                if kind=='numeric_status':key+=(update['status'],)
                result.append(key)
            return Counter(result)
        gold=items(record['expected']);emitted=items(record['emitted'])
        tp=sum((gold&emitted).values()) if record['prediction_valid'] else 0
        totals=[totals[0]+tp,totals[1]+sum(emitted.values())-tp,totals[2]+sum(gold.values())-tp]
    return prf(*totals)


def summary(records):
    counts=[0,0,0]
    for record in records:
        counts=[a+b for a,b in zip(counts,rt.confusion(record,nonintent=True))]
    primary=prf(*counts);corrections=[r for r in records if r['expected']['correction_detected']]
    unknown=[r for r in records if any(t in r['coverage'] for t in ['unknown_information','unknown_prior_ambiguous'])]
    unrelated=[];extra=[];extra_updates=[];convos=defaultdict(list)
    for r in records:
        changed={u['slot_id'] for u in r['expected']['updates']}
        emitted=r['emitted']['updates']
        raw=r.get('raw_value') if isinstance(r.get('raw_value'),dict) else {}
        raw_updates=raw.get('updates',[])
        raw_ids={u['slot_id'] for u in raw_updates if isinstance(u,dict) and isinstance(u.get('slot_id'),str)} if isinstance(raw_updates,list) else set()
        ids=raw_ids|{u['slot_id'] for u in emitted}
        extra.append(len(ids-changed-{'intent'}));extra_updates.append(sum(u['slot_id'] not in changed|{'intent'} for u in emitted))
        unrelated.append(all(r['predicted_after']['slots'][s]==v for s,v in r['prior_semantic_state']['slots'].items() if s not in changed|{'intent'}))
        convos[r['scenario_id']].append(r)
    latencies=sorted(r['latency_ms']/1000 for r in records)
    valid_latency=[r['latency_ms']/1000 for r in records if r['latency_ms'] is not None]
    result={'calls':len(records),'exact_nonintent':primary,'slot_name':diagnostic(records,'names'),
        'numeric_canonical_value_status':diagnostic(records,'numeric_status'),
        'numeric_canonical_value_without_status':diagnostic(records,'numeric_without_status'),
        'raw_JSON_valid':rate(sum(r['raw_json_valid'] for r in records),len(records)),
        'declared_schema_valid':rate(sum(r['raw_schema_valid'] for r in records),len(records)),
        'structural_parser_admission':rate(sum(r['structural_parser_admitted'] for r in records),len(records)),
        'primary_parser_admission':rate(sum(r['parser_admitted'] for r in records),len(records)),
        'application_admission':rate(sum(r['prediction_valid'] for r in records),len(records)),
        'exact_evidence_grounded':rate(sum(r['evidence_grounded'] for r in records),len(records)),
        'extra_current_turn_slot_flags':sum(extra),'extra_emitted_updates_including_repetitions':sum(extra_updates),
        'extra_flags_interpretation':'Includes rejected readable emissions and redundant/unsupported fields; not proof every flagged field is a fabricated fact.',
        'duplicate_slots':rate(sum(r['duplicate_slot_ids'] for r in records),len(records)),
        'correction_flag_correct':rate(sum(r['predicted'] is not None and r['predicted']['correction_detected']==r['expected']['correction_detected'] and r['prediction_valid'] for r in records),len(records)),
        'correction_flag_and_exact_replacement':rate(sum(r['prediction_valid'] and r['predicted']['correction_detected'] and rt.confusion(r,nonintent=True)[1:]==(0,0) for r in corrections),len(corrections)),
        'complete_state_after_correction':rate(sum(r['transition_correct'] for r in corrections),len(corrections)),
        'unrelated_nonintent_semantic_state_preserved':rate(sum(unrelated),len(records)),
        'unknown_admitted_no_updates':rate(sum(r['prediction_valid'] and not r['predicted']['updates'] for r in unknown),len(unknown)),
        'unknown_semantic_prior_preserved':rate(sum(r['prediction_valid'] and r['predicted_after']==r['prior_semantic_state'] for r in unknown),len(unknown)),
        'unknown_all_slot_metadata_preserved':rate(sum(r['raw_slot_metadata_preserved'] for r in unknown),len(unknown)),
        'unknown_top_level_intent_correct':rate(sum(r['prediction_valid'] and r['predicted']['intent']==r['expected']['intent'] for r in unknown),len(unknown)),
        'complete_state_every_turn':rate(sum(r['transition_correct'] for r in records),len(records)),
        'complete_conversation_final_state':rate(sum(max(rs,key=lambda r:r['key'][2])['transition_correct'] for rs in convos.values()),len(convos)),
        'latency_original_generation_seconds':{'mean':statistics.mean(valid_latency) if valid_latency else None,
            'median':statistics.median(valid_latency) if valid_latency else None,
            'p95':latencies[max(0,(95*len(latencies)+99)//100-1)] if latencies else None},
        'records_scored_from_durable_journal':sum(r['score_reconstructed_from_journal'] for r in records),
        'journal_replay_latency_included':False,'generation_sessions':len({r['generation_session'] for r in records}),
        'tokens':{key:sum(r.get(key) or 0 for r in records) for key in ['prompt_tokens','completion_tokens','reasoning_tokens']},
        'known_scored_hosted_cost_usd':sum(r.get('usage_cost_usd') or 0 for r in records)}
    return result


def paired(left,right):
    a={tuple(r['key']):r for r in left};b={tuple(r['key']):r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():raise ValueError('Matched inventory missing or duplicated')
    pairs=[]
    for key in sorted(a,key=str):
        x,y=a[key],b[key]
        if x['expected']!=y['expected'] or x['gold_sha256']!=y['gold_sha256'] or x['cluster_id']!=y['cluster_id']:
            raise ValueError('Paired gold/groups differ')
        if x['context']['message']!=y['context']['message']:raise ValueError('Paired user script differs')
        if x['track']=='component' and x['context']!=y['context']:raise ValueError('Component contexts differ')
        pairs.append((x,y))
    return pairs


def bootstrap(pairs,samples=10000,seed=42):
    import numpy as np
    groups=sorted({x['cluster_id'] for x,y in pairs});indexes={g:i for i,g in enumerate(groups)}
    counts=np.zeros((len(groups),2,3),dtype=np.int64)
    for x,y in pairs:
        for arm,r in enumerate((x,y)):counts[indexes[r['cluster_id']],arm]+=rt.confusion(r,nonintent=True)
    def f1(value):
        tp,fp,fn=value[...,0],value[...,1],value[...,2];denominator=2*tp+fp+fn
        return np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),where=denominator!=0)
    point=f1(counts.sum(axis=0));rng=np.random.default_rng(seed);deltas=[]
    for start in range(0,samples,256):
        selected=rng.integers(0,len(groups),size=(min(256,samples-start),len(groups)))
        scores=f1(counts[selected].sum(axis=1));deltas.extend(scores[:,0]-scores[:,1])
    return {'left_f1':float(point[0]),'right_f1':float(point[1]),'delta_left_minus_right':float(point[0]-point[1]),
        'ci95':[float(v) for v in np.quantile(deltas,[.025,.975])],'samples':samples,'seed':seed,
        'groups':len(groups),'interpretation':'Paired source/template-family percentile interval; zero-containing interval is not equivalence.'}


def failure_rows(records):
    rows=[]
    for r in records:
        gold={u['slot_id']:u for u in r['expected']['updates']};emitted={u['slot_id']:u for u in r['emitted']['updates']}
        missing=sorted(set(gold)-set(emitted)-{'intent'});extras=sorted(set(emitted)-set(gold)-{'intent'})
        wrong_values=[];wrong_status=[];role_candidates=[]
        for slot in set(gold)&set(emitted)-{'intent'}:
            if rt.dumps(gold[slot]['candidate_value'])!=rt.dumps(emitted[slot]['candidate_value']):wrong_values.append(slot)
            if gold[slot]['status']!=emitted[slot]['status']:wrong_status.append(slot)
        for extra in extras:
            for slot,u in gold.items():
                if rt.dumps(u['candidate_value'])==rt.dumps(emitted[extra]['candidate_value']):
                    role_candidates.append({'emitted_slot':extra,'gold_slot':slot})
        if missing or extras or wrong_values or wrong_status or not r['prediction_valid'] or not r['transition_correct']:
            rows.append({'call_id':r['call_id'],'domain':r['domain'],'language':r['language'],'track':r['track'],
                'missing_slots':missing,'wrong_values':sorted(wrong_values),'wrong_statuses':sorted(wrong_status),
                'extra_slots_for_human_review':extras,'possible_role_confusions_for_human_review':role_candidates,
                'structural_or_evidence_error':r['admission_error'],'application_error':r['application_error'],
                'state_correct':r['transition_correct'],'raw_output':r['raw_output'],'current_message':r['context']['message'],
                'classification':'automatic triage, not independent human adjudication'})
    return rows
