"""Generate blank gold-first and conversation-stable blinded review forms."""
from copy import deepcopy
import csv
import json
import random
import re
import zipfile
from common import OUT,dump,write,csv_write,load,prior_states,sha


GOLD_MUTABLE=['reviewer_identity','review_decision','proposed_correction_json','rationale']
PAIR_MUTABLE=['reviewer_identity','A_decision','A_issue_categories','A_proposed_correction_json','A_rationale','B_decision','B_issue_categories','B_proposed_correction_json','B_rationale','pairwise_preference','pairwise_rationale']
ADJ_MUTABLE=['reviewer_1_identity','reviewer_1_decision','reviewer_2_identity','reviewer_2_decision','adjudicator_identity','adjudication_decision','proposed_correction_json','rationale']


def main():
    cases,rows,schema=load();priors=prior_states(cases,rows);index={tuple(r['key'][1:]):r for r in rows}
    root=OUT/'human-review';reviewers=root/'reviewers';coord=root/'coordinator';reviewers.mkdir(parents=True,exist_ok=True);coord.mkdir(exist_ok=True)
    assignment=['v8']*10+['openai']*10;random.Random(20261028).shuffle(assignment)
    mapping=[];gold=[];pairs=[];adjudication=[]
    for number,case in enumerate(cases,1):
        cid=f'C{number:02d}';a=assignment[number-1];b='openai' if a=='v8' else 'v8'
        mapping.append({'conversation_id':cid,'scenario_id':case['scenario_id'],'A_model':a,'B_model':b,'stable_across_tracks_and_turns':True})
        for turn in case['turns']:
            base={'conversation_id':cid,'turn_id':turn['turn_id'],'language':case['language'],'domain':case['domain'],'current_message':turn['message'],'prior_user_messages_json':dump(turn['prior_user_messages']),'declared_intervention_json':dump(turn.get('state_intervention'))}
            g={'row_id':f'G-{cid}-{turn["turn_id"]}',**base,'established_intent':turn['established_intent'],'supplied_gold_prior_state_json':dump(turn['context_state']),'expected_raw_extraction_json':dump(turn['gold_extraction']),'expected_state_after_json':dump(turn['expected_state_after']),**{f:'' for f in GOLD_MUTABLE}}
            gold.append(g)
            adjudication.append({'row_id':f'J-{cid}-{turn["turn_id"]}',**base,'original_expected_raw_extraction_json':dump(turn['gold_extraction']),'original_expected_state_json':dump(turn['expected_state_after']),**{f:'' for f in ADJ_MUTABLE}})
            for track in ('component','rollout'):
                pair={'row_id':f'P-{cid}-{turn["turn_id"]}-{track}',**base,'track':track}
                for label,arm in [('A',a),('B',b)]:
                    key=(arm,track,case['scenario_id'],turn['turn_id']);r=index[key]
                    pair[label+'_input_prior_json']=dump(priors[key]);pair[label+'_raw_response']=r['raw_output']
                pair.update({f:'' for f in PAIR_MUTABLE});pairs.append(pair)
    tables={'gold_first_review.csv':(gold,GOLD_MUTABLE,'gold'),'second_reviewer_gold.csv':(deepcopy(gold),GOLD_MUTABLE,'gold_second'),'blinded_paired_responses.csv':(pairs,PAIR_MUTABLE,'paired'),'adjudication.csv':(adjudication,ADJ_MUTABLE,'adjudication')}
    manifest={}
    for name,(items,mutable,kind) in tables.items():
        csv_write(reviewers/name,items)
        immutable=[k for k in items[0] if k not in mutable]
        manifest[name]={'kind':kind,'mutable_fields':mutable,'immutable_fields':immutable,'rows':{r['row_id']:{k:r[k] for k in immutable} for r in items},'blank_template_sha256':sha(reviewers/name)}
        assert all(not r[f] for r in items for f in mutable)
    write(coord/'BLINDING_KEY.json',{'seed':20261028,'mapping':mapping,'disclosure':'Coordinator only. Do not give this file or the full supplement/source ZIP to blinded reviewers. Allocation is balanced 10/10 and fixed within conversation across both tracks.'})
    write(coord/'FORM_REFERENCE_MANIFEST.json',manifest)
    write(coord/'SCHEMA_VALIDATION.json',{'slot_ids':[s.slot_id for s in schema.slots],'statuses':['unmentioned','provided','inferred','ambiguous','conflicting','invalid','dont_care','confirmed'],'independent_human_review_complete':False})
    csv_write(reviewers/'slot_reference.csv',[{'slot_id':s.slot_id,'description':s.description,'value_type':s.value_type,'allowed_values_json':dump(list(s.allowed_values) if s.allowed_values else [])} for s in schema.slots])
    # Predictions are kept separate so coordinators can release gold forms first.
    for name,selected in [('01-gold-first-review-kit.zip',['gold_first_review.csv','second_reviewer_gold.csv','adjudication.csv','slot_reference.csv']),('02-blinded-response-review-kit.zip',['blinded_paired_responses.csv','slot_reference.csv'])]:
        path=root/name
        if path.exists():raise FileExistsError('Preserve a previously distributed review kit')
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
            for f in selected:z.write(reviewers/f,f)
            guide=root/'ANNOTATION_GUIDE.md'
            if guide.exists():z.write(guide,'ANNOTATION_GUIDE.md')
    prohibited=re.compile(r'\b(?:openai|qwen|gpt-5|slm|v8)\b|latency_seconds|raw_exact_F1|confusion',re.I)
    hits=[]
    for path in reviewers.glob('*.csv'):
        if prohibited.search(path.read_text(encoding='utf-8-sig')):hits.append(path.name)
    assert not hits,('Reviewer-facing identity/metric leakage',hits)
    write(root/'REVIEW_KIT_STATUS.json',{'gold_rows':len(gold),'paired_rows':len(pairs),'second_reviewer_rows':len(gold),'adjudication_rows':len(adjudication),'conversations':20,'turns':101,'reviewer_decisions_filled':0,'independent_human_review_complete':False,'blinded_identity_score_latency_scan_passed':True,'mapping_is_coordinator_only':True})
    print(dump({'review_kit':str(root),'gold_rows':101,'paired_rows':202,'all_decision_fields_blank':True}))


if __name__=='__main__':main()
