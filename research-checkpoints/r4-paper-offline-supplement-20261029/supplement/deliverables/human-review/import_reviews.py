"""Validate human-filled forms and export proposals only; never edit evidence."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def required(row,fields):
    for field in fields:
        if not row.get(field,'').strip():raise ValueError('Completed row requires '+field)
def choice(value,allowed):
    if value not in allowed:raise ValueError('Invalid decision')

def strict_loads(raw):
    def pairs(items):
        obj={}
        for k,v in items:
            if k in obj:raise ValueError('Duplicate JSON field')
            obj[k]=v
        return obj
    def constant(value):raise ValueError('Nonfinite JSON number')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def validate_proposal(raw,schema,message):
    if not raw.strip():return None
    obj=strict_loads(raw)
    if not isinstance(obj,dict) or not obj or not set(obj)<={'gold_extraction','expected_state_after','context_state','notes'}:raise ValueError('Unknown proposal shape/field')
    if not set(obj)&{'gold_extraction','expected_state_after','context_state'}:raise ValueError('Proposal has no proposed label/state change')
    known=set(schema['slot_ids']);statuses=set(schema['statuses'])
    for name in ['expected_state_after','context_state']:
        if name in obj:
            if not isinstance(obj[name],dict):raise ValueError('Proposed state must be an object')
            for sid,item in obj[name].items():
                if sid not in known or not isinstance(item,dict) or set(item)!={'value','status','confirmed_by_user'}:raise ValueError('Invalid proposed state slot/fields')
                if item['status'] not in statuses or type(item['confirmed_by_user']) is not bool:raise ValueError('Invalid state status/confirmation')
    if 'gold_extraction' in obj:
        g=obj['gold_extraction']
        if not isinstance(g,dict) or set(g)!={'intent','intent_confidence','updates','correction_detected','unsupported_claims'}:raise ValueError('Invalid proposed extraction fields')
        if g['intent'] not in {'create_forecast','not_forecasting','ambiguous','unsupported'}:raise ValueError('Invalid intent')
        if type(g['intent_confidence']) not in (int,float) or not 0<=g['intent_confidence']<=1:raise ValueError('Invalid confidence')
        if type(g['correction_detected']) is not bool or not isinstance(g['updates'],list) or not isinstance(g['unsupported_claims'],list) or not all(isinstance(x,str) for x in g['unsupported_claims']):raise ValueError('Invalid extraction types')
        seen=set()
        for u in g['updates']:
            if not isinstance(u,dict) or set(u)!={'slot_id','candidate_value','status','confidence','evidence_text'}:raise ValueError('Invalid update fields')
            if u['slot_id'] not in known or u['slot_id'] in seen:raise ValueError('Unknown/duplicate proposed slot')
            seen.add(u['slot_id'])
            if u['status'] not in statuses or type(u['confidence']) not in (int,float) or not 0<=u['confidence']<=1:raise ValueError('Invalid update status/confidence')
            if not isinstance(u['candidate_value'],str):raise ValueError('Proposed gold candidate must be JSON text')
            strict_loads(u['candidate_value'])
            if not isinstance(u['evidence_text'],str) or not u['evidence_text'] or u['evidence_text'] not in message:raise ValueError('Proposed evidence is not verbatim current-message text')
    return obj


def validate_form(kit,template,form,allow_partial=False):
    manifest=json.loads((kit/'coordinator/FORM_REFERENCE_MANIFEST.json').read_bytes())
    if template not in manifest:raise ValueError('Unknown form template')
    spec=manifest[template];schema=json.loads((kit/'coordinator/SCHEMA_VALIDATION.json').read_bytes())
    with form.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    ids=[r.get('row_id') for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(spec['rows']):raise ValueError('Missing, extra or duplicate row IDs')
    expected_columns=set(spec['immutable_fields'])|set(spec['mutable_fields'])
    result=[];pending=0
    for row in rows:
        if set(row)!=expected_columns:raise ValueError('Form columns changed')
        base=spec['rows'][row['row_id']]
        if any(row[k]!=str(base[k]) for k in spec['immutable_fields']):raise ValueError('Immutable context/label/input changed')
        if all(not row[k].strip() for k in spec['mutable_fields']):pending+=1;continue
        proposals={}
        if spec['kind'] in {'gold','gold_second'}:
            required(row,['reviewer_identity','review_decision','rationale']);choice(row['review_decision'],{'approve','propose_change','unclear'})
            proposal=validate_proposal(row['proposed_correction_json'],schema,row['current_message'])
            if row['review_decision']=='propose_change' and proposal is None:raise ValueError('Change decision requires a proposal')
            if row['review_decision']=='approve' and proposal is not None:raise ValueError('Approve decision cannot change labels')
            if proposal:proposals['proposed_gold_or_state']=proposal
        elif spec['kind']=='paired':
            required(row,['reviewer_identity','A_decision','A_rationale','B_decision','B_rationale','pairwise_preference','pairwise_rationale'])
            for label in ['A','B']:
                choice(row[label+'_decision'],{'correct','partly_correct','incorrect','unclear'})
                proposal=validate_proposal(row[label+'_proposed_correction_json'],schema,row['current_message'])
                if proposal:proposals[label]=proposal
            choice(row['pairwise_preference'],{'A','B','tie','unclear'})
        else:
            required(row,['reviewer_1_identity','reviewer_2_identity','reviewer_1_decision','reviewer_2_decision','adjudicator_identity','adjudication_decision','rationale'])
            if row['reviewer_1_identity']==row['reviewer_2_identity']:raise ValueError('Two reviewers must have distinct identities')
            for f in ['reviewer_1_decision','reviewer_2_decision']:choice(row[f],{'approve','propose_change','unclear'})
            choice(row['adjudication_decision'],{'retain_original','propose_change','unresolved'})
            proposal=validate_proposal(row['proposed_correction_json'],schema,row['current_message'])
            if row['adjudication_decision']=='propose_change' and proposal is None:raise ValueError('Adjudicated change requires proposal')
            if row['adjudication_decision']=='retain_original' and proposal is not None:raise ValueError('Retain decision cannot change labels')
            if proposal:proposals['proposed_gold_or_state']=proposal
        result.append({'row_id':row['row_id'],'conversation_id':row['conversation_id'],'turn_id':row['turn_id'],'track':row.get('track'),'review':{k:row[k] for k in spec['mutable_fields']},'proposals':proposals})
    if pending and not allow_partial:raise ValueError('Pending blank rows remain; no complete human review imported')
    return {'form_template':template,'form_sha256':sha(form),'reference_manifest_sha256':sha(kit/'coordinator/FORM_REFERENCE_MANIFEST.json'),'completed_rows':len(result),'pending_rows':pending,'all_rows_completed':pending==0,'independent_human_review_complete':False,'human_identity_and_independence_verified':False,'original_labels_modified':False,'changes_applied':False,'reviews':result}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--kit',type=Path,required=True);parser.add_argument('--template',required=True);parser.add_argument('--form',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    result=validate_form(args.kit,args.template,args.form,args.allow_partial)
    args.output.mkdir(parents=True,exist_ok=False)
    result['imported_at_UTC']=datetime.now(timezone.utc).isoformat()
    (args.output/'PROPOSED_REVIEW_CHANGES.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='reviews'}))


if __name__=='__main__':main()
