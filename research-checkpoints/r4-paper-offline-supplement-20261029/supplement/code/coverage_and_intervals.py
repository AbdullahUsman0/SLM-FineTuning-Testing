"""Post-hoc coverage and paired exploratory bootstrap of saved R4 counts."""
from collections import Counter
import gzip
import hashlib
import json
import numpy as np
from common import OUT,E,ROOT,load,p,dump,write,csv_write,counts_to_prf,sha


def interval(values):
    lo,hi=np.percentile(values,[2.5,97.5],method='linear')
    return [float(lo),float(hi)]


def f1_array(counts):
    den=2*counts[:,0]+counts[:,1]+counts[:,2]
    return np.divide(2*counts[:,0],den,out=np.zeros(len(den)),where=den!=0)


def main():
    cases,rows,schema=load();out=OUT/'analysis';out.mkdir(exist_ok=True)
    case_index={c['scenario_id']:c for c in cases};turns={(c['scenario_id'],t['turn_id']):t for c in cases for t in c['turns']}
    slot_ids=[s.slot_id for s in schema.slots if s.slot_id!='intent'];assert len(slot_ids)==78
    support=Counter()
    for turn in turns.values():support.update(t[0] for t in p.tuples(turn['gold_extraction']))
    supported=sorted(s for s,n in support.items() if n);assert len(supported)==39 and sum(support.values())==257
    csv_write(out/'gold_slot_support.csv',[{'slot_id':sid,'gold_updates':support[sid],'positively_supported':sid in supported} for sid in slot_ids])
    summaries={};slot_rows=[];conversation_rows=[];subgroups=[];sensitivity=[];unknown_rows=[];macro=[]
    original=json.loads((E/'main-summary.json').read_bytes())
    for arm in ('v8','openai'):
        summaries[arm]={}
        for track in ('component','rollout'):
            selected=[r for r in rows if r['key'][1:3]==[arm,track]];per={sid:Counter(tp=0,fp=0,fn=0,predictions=0) for sid in slot_ids};outside=Counter()
            for row in selected:
                turn=turns[tuple(row['key'][3:])];gold=Counter(p.tuples(turn['gold_extraction']))
                try:pred=Counter(p.tuples(json.loads(row['raw_output'])))
                except Exception:pred=Counter()
                match=gold&pred if row['raw_contract_admitted'] else Counter()
                for sid in {t[0] for t in gold}|{t[0] for t in pred}:
                    tp=sum(n for t,n in match.items() if t[0]==sid);g=sum(n for t,n in gold.items() if t[0]==sid);n=sum(n for t,n in pred.items() if t[0]==sid)
                    if sid not in per:outside[sid]+=n;continue
                    per[sid].update(tp=tp,fp=n-tp,fn=g-tp,predictions=n)
            totals=[sum(v[k] for v in per.values()) for k in ['tp','fp','fn']];totals[1]+=sum(outside.values());prf=counts_to_prf(totals)
            assert prf==original[arm][track]['raw_exact']
            summaries[arm][track]=prf
            for sid in slot_ids:
                v=per[sid];slot_rows.append({'model':arm,'track':track,'slot_id':sid,'gold_support':support[sid],'predictions':v['predictions'],**counts_to_prf([v['tp'],v['fp'],v['fn']]),'supported_slot':sid in supported})
            for sid,n in outside.items():unknown_rows.append({'model':arm,'track':track,'unknown_slot_id':sid,'predictions':n,'fp':n})
            macro.append({'model':arm,'track':track,'supported_slot_universe':39,'supported_macro_F1':sum(counts_to_prf([per[s]['tp'],per[s]['fp'],per[s]['fn']])['f1'] for s in supported)/39,'unsupported_known_slots':39,'unsupported_slot_predictions':sum(per[s]['predictions'] for s in slot_ids if s not in supported),'unsupported_slot_FP':sum(per[s]['fp'] for s in slot_ids if s not in supported),'unknown_ID_FP':sum(outside.values())})
            for case in cases:
                group=[r for r in selected if r['key'][3]==case['scenario_id']];assert len(group)==len(case['turns'])
                conversation_rows.append({'model':arm,'track':track,'scenario_id':case['scenario_id'],'domain':case['domain'],'language':case['language'],'category':case['category'],'entity_group':case['entity_group'],'template_variant':case.get('template_variant'),'turns':len(group),'gold_updates':sum(r['confusion'][0]+r['confusion'][2] for r in group),'predictions':sum(r['confusion'][0]+r['confusion'][1] for r in group),**p.prf(group),'exact_pipeline_turns':sum(r['complete_pipeline_state'] for r in group),'final_pipeline_state_correct':group[-1]['complete_pipeline_state']})
            for field in ['domain','language','domain_family','category']:
                for value in sorted({c[field] for c in cases}):
                    group=[r for r in selected if case_index[r['key'][3]][field]==value]
                    subgroups.append({'model':arm,'track':track,'stratum':field,'value':value,'conversations':len({r['key'][3] for r in group}),'calls':len(group),'gold_updates':sum(r['confusion'][0]+r['confusion'][2] for r in group),**p.prf(group),'exact_pipeline_turns':sum(r['complete_pipeline_state'] for r in group)})
            declared={c['scenario_id'] for c in cases if any((t.get('state_intervention') or {}).get('mutations') for t in c['turns'])}
            variants={'all_recorded_turns':selected,'exclude_published_intervention_category':[r for r in selected if case_index[r['key'][3]]['category']!='explicit_state_repair'],'exclude_whole_declared_intervention_conversations':[r for r in selected if r['key'][3] not in declared],'declared_mutation_turns_only':[r for r in selected if (turns[tuple(r['key'][3:])].get('state_intervention') or {}).get('mutations')]}
            assert p.prf(variants['exclude_published_intervention_category'])==original[arm][track]['excluding_intervention_category']['raw_exact']
            for label,group in variants.items():sensitivity.append({'model':arm,'track':track,'subset':label,'calls':len(group),'conversations':len({r['key'][3] for r in group}),**p.prf(group),'interpretation':'subset of recorded trajectories; not a new no-intervention rollout'})
    csv_write(out/'per_slot_78.csv',slot_rows);csv_write(out/'per_conversation.csv',conversation_rows);csv_write(out/'domain_language_category.csv',subgroups);csv_write(out/'intervention_sensitivity.csv',sensitivity);csv_write(out/'supported_slot_macro.csv',macro)
    csv_write(out/'unknown_slot_ID_predictions.csv',unknown_rows,fields=['model','track','unknown_slot_id','predictions','fp'])
    write(out/'primary_reproduction.json',{'exactly_matches_publication':True,'metrics':summaries,'supported_nonintent_slots':supported,'supported_count':39,'unsupported_count':39,'slot_count':78,'gold_updates_per_track':257,'macro_definition':'Unweighted mean exact F1 over the fixed 39 gold-positive slots; unsupported-slot and unknown-ID FP retained separately; primary micro still includes them.'})
    # The plan is already saved and hashed before generating any new intervals.
    plan=OUT/'POSTHOC_ANALYSIS_PLAN.md';assert plan.exists()
    group_ids={c['scenario_id']:'|'.join([str(c.get('source_corpus_version') or c['corpus_version']),c['category'],str(c.get('template_variant','not_recorded'))]) for c in cases}
    csv_write(out/'dependence_groups.csv',[{'scenario_id':c['scenario_id'],'entity_group':c['entity_group'],'cluster_id':c['cluster_id'],'template_variant':c.get('template_variant'),'script_family':group_ids[c['scenario_id']]} for c in cases])
    rng=np.random.Generator(np.random.PCG64(20261028));B=10000;n=len(cases);draws=rng.integers(0,n,size=(B,n));families=sorted(set(group_ids.values()));family_draws=rng.integers(0,len(families),size=(B,len(families))) if len(families)>=4 else None
    output={'posthoc':True,'preregistered':False,'plan_sha256':sha(plan),'resamples':B,'seed':20261028,'rng':'NumPy PCG64','numpy_version':np.__version__,'percentile_method':'linear 2.5/97.5','conversation_units':20,'distinct_recorded_entity_groups':len({c['entity_group'] for c in cases}),'script_family_units':len(families),'template_variant_4_conversations':sum(c.get('template_variant')==4 for c in cases),'conversation_draws_sha256':hashlib.sha256(draws.astype('<i4').tobytes()).hexdigest(),'intervals':[],'limits':'Exploratory conditional intervals on inspected related synthetic development; paired units preserve outcomes, not population independence. Shared generation dependence may remain between script families; no p-values or universal-superiority conclusion.'}
    bootstrap_lines=[]
    for track in ('component','rollout'):
        arrays={arm:np.array([[sum(r['confusion'][i] for r in rows if r['key'][1:4]==[arm,track,c['scenario_id']]) for i in range(3)] for c in cases],dtype=np.int64) for arm in ('v8','openai')}
        for scheme in ['paired_conversation','paired_script_family']:
            if scheme=='paired_script_family' and family_draws is None:continue
            if scheme=='paired_conversation':sample={arm:f1_array(arrays[arm][draws].sum(axis=1)) for arm in arrays}
            else:
                grouped={arm:np.array([arrays[arm][[i for i,c in enumerate(cases) if group_ids[c['scenario_id']]==family]].sum(axis=0) for family in families]) for arm in arrays}
                sample={arm:f1_array(grouped[arm][family_draws].sum(axis=1)) for arm in arrays}
            difference=sample['v8']-sample['openai']
            output['intervals'].append({'track':track,'scheme':scheme,'units':20 if scheme=='paired_conversation' else len(families),'v8_point':summaries['v8'][track]['f1'],'openai_point':summaries['openai'][track]['f1'],'difference_point':summaries['v8'][track]['f1']-summaries['openai'][track]['f1'],'v8_interval':interval(sample['v8']),'openai_interval':interval(sample['openai']),'paired_difference_interval':interval(difference)})
            for i in range(B):bootstrap_lines.append(dump({'track':track,'scheme':scheme,'draw':i,'v8_F1':float(sample['v8'][i]),'openai_F1':float(sample['openai'][i]),'difference':float(difference[i])}))
    (out/'bootstrap_samples.jsonl.gz').write_bytes(gzip.compress(('\n'.join(bootstrap_lines)+'\n').encode(),mtime=0));write(out/'exploratory_paired_intervals.json',output)
    print(dump({'primary_scores_reproduced':True,'coverage_rows':len(slot_rows),'per_conversation_rows':len(conversation_rows),'script_families':len(families),'intervals':output['intervals']}))


if __name__=='__main__':main()
