"""Reproduce a targeted agent pre-screen; never sign a human approval.

This is a rule-based audit of known review concerns, not an independent semantic
accuracy estimate. Pair and clause interpretation is documented separately.
"""
import argparse,csv,gzip,hashlib,json,re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

CHECKS={
 'unchanged_provided_update':('blocking','An unchanged prior value is labeled provided despite the no-repeat prompt.'),
 'scope_mismatch':('blocking','Success criteria refer to an inherited fictional scope different from the requested scope.'),
 'rejected_alternative_in_positive_target':('revise','The positive target value contains a comma-not rejection; the rubric excludes rejected alternatives.'),
 'weak_confirmation_wording':('revise','A Yes/Haan prefix is the sole confirmation cue; explicitly confirm the recorded value.'),
 'unclear_cadence_resolution':('revise','The earlier every-hour phrase is not explicitly resolved or discarded when another schedule is supplied.'),
 'roman_urdu_duration_inflection':('wording','A plural duration uses ghanta/mahina/hafta; improve ghantay/mahinay/haftay phrasing.'),
 'positive_target_also_rejected':('blocking','The same target is requested and then rejected in the initial message.'),
}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]
def write(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def audit(corpus,output,human_ledger=None):
    if output.exists():raise FileExistsError('Immutable agent review exists')
    manifest=json.loads((corpus/'manifest.json').read_bytes())
    lock=json.loads((corpus/'manifest.sha256.json').read_bytes())
    if sha(corpus/'manifest.json')!=lock['manifest.json']['sha256']:raise ValueError('Changed manifest')
    names=['splits/train.jsonl.gz','splits/validation.jsonl.gz','review-sample.jsonl.gz','overlap-review-queue.jsonl.gz']
    for name in names:
        if sha(corpus/name)!=manifest['files'][name]['sha256']:raise ValueError('Changed review input: '+name)
    ledger_before=sha(human_ledger) if human_ledger else None
    cases=[c for s in ('train','validation') for c in read(corpus/'splits'/f'{s}.jsonl.gz')]
    if any(c['split']=='final' for c in cases):raise ValueError('Final data in development inputs')
    indexed={c['scenario_id']:c for c in cases}; sample=read(corpus/'review-sample.jsonl.gz')
    findings=[]; by_case=defaultdict(set); horizons=defaultdict(set); horizon_periods=defaultdict(set); target_forms=defaultdict(set)
    booleans=defaultdict(Counter); methods=Counter()
    def flag(code,c,t,detail,slot=None):
        findings.append({'issue':code,'severity':CHECKS[code][0],'scenario_id':c['scenario_id'],
            'split':c['split'],'domain':c['domain'],'language':c['language'],
            'turn_id':t['turn_id'],'slot':slot,'message':t['message'],'detail':detail})
        by_case[c['scenario_id']].add(code)
    for c in cases:
        first={f['slot_id']:f['value'] for f in c['turns'][0]['facts']}
        target=first['target_description'];scope=target.rsplit(' for ',1)[1]
        key=(c['domain'],c['category']);horizons[key].add(json.dumps(first['forecast_horizon'],sort_keys=True))
        horizon_periods[key].add(first['forecast_horizon']['periods'])
        target_forms[key].add(target.rsplit(' for ',1)[0])
        if c['category']=='target_contrast':
            t=c['turns'][0]
            match=re.search(r'Do not substitute (.*?) as my requested target\.',t['message']) or re.search(r'\. ([^.]+) nahi chahiye\.',t['message'])
            if match:
                other=re.sub(r'^(daily|hourly|weekly|monthly|quarterly)\s+','',match.group(1))
                if target.rsplit(' for ',1)[0]==other:
                    flag('positive_target_also_rejected',c,t,{'positive':target,'rejected':match.group(1)},'target_description')
        for t in c['turns']:
            methods[t['behavior']]+=1
            for f in t['facts']:
                slot=f['slot_id']; old=t['context_state'].get(slot)
                if isinstance(f['value'],bool):booleans[slot][str(f['value']).lower()]+=1
                if old and old['value']==f['value'] and old['status']==f['status'] and f['status']=='provided':
                    flag('unchanged_provided_update',c,t,{'value':f['value']},slot)
                if slot=='success_criteria' and scope not in f['value']:
                    flag('scope_mismatch',c,t,{'requested_scope':scope,'criterion':f['value']},slot)
                if slot in ('problem_statement','target_description') and ', not ' in f['value']:
                    flag('rejected_alternative_in_positive_target',c,t,{'value':f['value']},slot)
            if 'confirmation' in t['behavior'] and t['message'].startswith(('Yes, ','Haan, ')):
                flag('weak_confirmation_wording',c,t,{'status':'confirmed','prior_value_matches':True},'forecast_horizon')
            if t['behavior']=='cadence_clarified' and not t['message'].startswith(('The earlier every-hour phrase','Pehli har-ghantay')):
                flag('unclear_cadence_resolution',c,t,{'earlier_unresolved_phrase':'every hour'})
            errors=re.findall(r'(?<!\d)(?:[2-9]|\d{2,}) (?:ghanta|mahina|hafta)\b',t['message'])
            if errors:flag('roman_urdu_duration_inflection',c,t,{'phrases':errors})
    pairs=[]
    for p in read(corpus/'overlap-review-queue.jsonl.gz'):
        a=indexed[p['nearest_train']];b=indexed[p['validation']]
        ta=a['turns'][0];tb=b['turns'][0]
        av={f['slot_id']:f['value'] for f in ta['facts']};bv={f['slot_id']:f['value'] for f in tb['facts']}
        pairs.append({**p,'review_status':'agent_shared_template_flag','reviewer_type':'agent',
            'exact_message_match':ta['message']==tb['message'],'same_slot_set':set(av)==set(bv),
            'same_target_except_scope':av['target_description'].rsplit(' for ',1)[0]==bv['target_description'].rsplit(' for ',1)[0],
            'same_horizon':av['forecast_horizon']==bv['forecast_horizon'],
            'same_category':a['category']==b['category'],
            'different_entity_groups':a['entity_group']!=b['entity_group'],
            'train_message':ta['message'],'validation_message':tb['message'],
            'decision':'Retain only as disclosed synthetic development data; not independent natural-language evidence.',
            'human_decision':'pending'})
    output.mkdir(parents=True)
    write(output/'findings.json',findings);write(output/'overlap-agent-decisions.json',pairs)
    sample_rows=[]
    for c in sample:
        issues=sorted(by_case.get(c['scenario_id'],set()))
        sample_rows.append({'scenario_id':c['scenario_id'],'split':c['split'],'domain':c['domain'],'language':c['language'],
            'category':c['category'],'reviewer_type':'agent','method':'systematic_checks_plus_template_review',
            'decision':'revise' if issues else 'no_issue_detected_by_checks',
            'issues':';'.join(issues),'human_decision':'pending',
            'rationale':'; '.join(CHECKS[i][1] for i in issues) if issues else
                         'Passed these targeted checks; this is not an individual human semantic approval.'})
    with (output/'sample-agent-decisions.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(sample_rows[0]));writer.writeheader();writer.writerows(sample_rows)
    constant=[{'domain':d,'category':c,'horizon':json.loads(next(iter(v)))} for (d,c),v in horizons.items() if len(v)==1]
    summary={'corpus_version':manifest['corpus_version'],'corpus_manifest_sha256':sha(corpus/'manifest.json'),
        'reviewer_type':'agent','audit_version':4,'reviewed_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Targeted programmatic pass over all development records; clause/template interpretation by Codex. Not 524 independently read human conversations.',
        'development_scenarios':len(cases),'development_turns':sum(len(c['turns']) for c in cases),
        'sample_scenarios':len(sample),'sample_decisions':dict(Counter(r['decision'] for r in sample_rows)),
        'issue_occurrences':dict(Counter(f['issue'] for f in findings)),
        'scenarios_with_issues':len({f['scenario_id'] for f in findings}),
        'affected_scenarios_by_issue':{i:len({f['scenario_id'] for f in findings if f['issue']==i}) for i in CHECKS},
        'boolean_supervision':{k:dict(v) for k,v in booleans.items()},
        'inferred_file_format_updates':sum(f['slot_id']=='file_format' and f['status']=='inferred' for c in cases for t in c['turns'] for f in t['facts']),
        'constant_horizon_domain_category_cells':len(constant),'constant_horizon_cells':constant,
        'constant_horizon_period_cells':sum(len(v)==1 for v in horizon_periods.values()),
        'target_form_counts_per_domain_category':[{'domain':d,'category':c,'target_forms':len(v)} for (d,c),v in sorted(target_forms.items())],
        'overlap_representatives':len(pairs),'overlap_shared_template_flags':len(pairs),
        'overlap_threshold_pairs':sum(v for k,v in manifest['statistics']['overlap']['Jaccard_bins'].items() if k!='<0.6'),
        'overlap_exact_message_matches':sum(p['exact_message_match'] for p in pairs),
        'overlap_same_horizon':sum(p['same_horizon'] for p in pairs),
        'training_recommendation':'hold_for_revision' if findings or constant else 'pending_human_review_and_policy_adjudication',
        'human_review':'pending','human_approvals_written':0,
        'final_labels_accessed':False,'final_labels_generated':False,'training_started':False,'new_API_calls':0,
        'human_ledger_sha256':ledger_before,
        'input_hashes':{n:sha(corpus/n) for n in names},'audit_script_sha256':sha(Path(__file__)),
        'limitations':['Flags are not an unbiased semantic error rate.',
            'Exact candidate labels and values follow the recorded rubric; no earlier gold/results changed.',
            'All flagged overlap representatives are a review queue, not every threshold pair or a full semantic audit.',
            'Unknown withdrawal, dont_care, conflicts, STT and independent user prose remain uncovered.',
            'Roman Urdu technical clauses retain English field/value vocabulary; fluent speaker review required.',
            'Annotation policy: '+manifest.get('annotation_policy_version','conventions pending human adjudication')+'. Human example review remains pending.']}
    write(output/'summary.json',summary)
    (output/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    if human_ledger and sha(human_ledger)!=ledger_before:raise ValueError('Human ledger changed during audit')
    return {k:summary[k] for k in ('corpus_version','development_scenarios','sample_decisions','issue_occurrences',
                                 'constant_horizon_domain_category_cells','training_recommendation')}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--human-ledger',type=Path)
    a=p.parse_args();print(json.dumps(audit(a.corpus,a.output,a.human_ledger),indent=2))
