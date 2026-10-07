"""Analyze a completed frozen two-decoder development comparison."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SPEC=importlib.util.spec_from_file_location('natural_audit',ROOT/'scripts/audit-v6-natural-failures.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)

def analyze(run,output):
    completion=json.loads((run/'completion.json').read_text());assert completion['status']=='complete' and completion['scored_calls']==384
    output.mkdir(parents=True,exist_ok=False);schema=audit.load_schema();records={};rows=[];slot_rows=[]
    arms=tuple(completion.get('arms',('original','revised')));assert len(arms)==2
    result={'status':'complete','completion':completion,'tracks':{},'paired_differences':{},
            'analysis_sha256':audit.sha256(Path(__file__)),
            'used_audit_sha256':audit.sha256(ROOT/'scripts/audit-v6-natural-failures.py'),
            'post_inference_analysis_note':'Supplementary date serialization uses the existing state-snapshot ISO policy. Inference inputs, raw outputs, gold labels and strict scoring unchanged; original failed analysis retained separately.',
            'interpretation':'Exploratory decoding comparison on the same 32 development conversations previously used in prompt studies. No independent held-out improvement claim. Original prompt, labels and primary scoring unchanged.'}
    for track in ('natural_component','natural_rollout'):
        result['tracks'][track]={}
        for arm in arms:
            path=run/f'{arm}-{track}.json';payload=path.read_bytes();report=json.loads(payload)
            assert report['status']=='complete' and report['fingerprint']==completion['fingerprint']
            group=report['records'];records[(track,arm)]=group
            metric_groups={'all':group,**{d:[r for r in group if r['evaluation_domain']==d] for d in ('weather','inflation','stocks','crypto')}}
            result['tracks'][track][arm]={}
            for domain,subset in metric_groups.items():
                m=audit.analysis.summarize(subset);n=audit.normalized_metrics(subset,schema)
                counts=dict(__import__('collections').Counter(r['category'] for r in audit.audit_records(subset,schema)))
                result['tracks'][track][arm][domain]={'strict':m,'supplementary_normalized':n,'content_categories_before_schema_gate':counts,
                    'generation_limit_calls_from_recorded_token_count':sum(r['hit_generation_limit'] for r in subset)}
                rows.append({'track':track,'arm':arm,'domain':domain,'calls':len(subset),
                    'strict_f1':m['nonintent']['f1'],'normalized_f1':n['schema_gated_normalized_slot_value_status']['f1'],
                    'slot_name_f1':m['slot_name_prf']['f1'],'json_valid':m['extract_json_valid']['rate'],
                    'schema_valid':m['extract_schema_valid']['rate'],'new_unmentioned_slots':m['new_unmentioned_slot_count'],
                    'wrong_stated_values':m['wrong_values_for_stated_slots'],
                    'correction_flag_values':m['correction_flag_and_exact_value_accuracy']['rate'],
                    'local_horizon_corrections':n['correction_horizon_after_reducer']['numerator'],
                    'local_corrections_preserving_other_prior_slots':n['correction_horizon_and_other_prior_slots_preserved']['numerator'],
                    'full_state_after_correction':m['correction_flag_and_reducer_transition_accuracy']['rate'],
                    'unknown_no_updates_valid':m['unknown_no_updates_and_valid']['rate'],
                    'unknown_prior_state_retained_valid':m['unknown_retains_prior_state_and_valid']['rate'],
                    'median_ms':m['extract_latency_ms']['p50'],'p95_ms':m['extract_latency_ms']['p95']})
            for r in audit.audit_records(group,schema):slot_rows.append({'arm':arm,**r})
            compressed=gzip.compress(payload,mtime=0);assert gzip.decompress(compressed)==payload
            (output/(path.name+'.gz')).write_bytes(compressed)
        left=records[(track,arms[0])];right=records[(track,arms[1])]
        lm={tuple(r['key']):r for r in left};rm={tuple(r['key']):r for r in right};assert lm.keys()==rm.keys()
        for key,l in lm.items():
            assert l['gold_sha256']==rm[key]['gold_sha256']
            if track!='natural_rollout':assert l['context_sha256']==rm[key]['context_sha256']
        l=result['tracks'][track][arms[0]]['all'];r=result['tracks'][track][arms[1]]['all']
        result['paired_differences'][track]={
            'left_arm':arms[0],'right_arm':arms[1],
            'strict_f1_delta':r['strict']['nonintent']['f1']-l['strict']['nonintent']['f1'],
            'normalized_f1_delta':r['supplementary_normalized']['schema_gated_normalized_slot_value_status']['f1']-l['supplementary_normalized']['schema_gated_normalized_slot_value_status']['f1'],
            'new_unmentioned_slot_delta':r['strict']['new_unmentioned_slot_count']-l['strict']['new_unmentioned_slot_count'],
            'mean_paired_latency_delta_ms':sum(rm[k]['latency_ms']-v['latency_ms'] for k,v in lm.items())/len(lm),
            'cluster_bootstrap_primary_f1_delta':audit.analysis.bootstrap(left,right)}
    for name,data in [('summary.csv',rows),('slot-error-audit.csv',slot_rows)]:
        with (output/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    audit.write_json(output/'analysis.json',result)
    for name in ('run-manifest.json','completion.json'):shutil.copyfile(run/name,output/name)
    audit.write_json(output/'execution-evidence.json',{'sessions':{p.name:json.loads(p.read_text()) for p in (run/'sessions').glob('*.json')}})
    text=[f'# V6 decoding comparison: {arms[0]} versus {arms[1]}','',
          'All 384 new scored calls are complete: two frozen decoding arms using the identical original prompt, two context tracks, the same 32 conversations and the same unchanged final-step v6 weights. No training, repairs, retries, paid APIs or sealed final labels. Decoder order alternated per scenario in one serial BF16 CUDA session. Complete raw predictions, contexts, timings and gold labels are preserved in four compressed reports.','',
          'Grammar uses the existing exported structural schema with fixed object-field order, one fresh XGrammar matcher per request, and torch_native masking on CUDA. Inner JSON strings, duplicate slot updates and semantic correctness still require the unchanged application validator. Compilation time is separate from warmed extraction timings.','',
          'All 32 conversation labels and 96 turns were reviewed by the Codex agent, with no clear factual label error or label changes. This is not independent human review. The original cohort was already inspected to design the earlier prompts, so this experiment measures development-set response to one fixed decoding change, not generalization. Input review notes and the frozen protocol are under evaluation/v6-constraints-20261007.','',
          'Strict F1 preserves the original schema-gated exact slot/value/status definition. Supplementary normalized F1 applies only the existing pinned application normalize_value to both gold and observed values, retains the schema-validity gate and status, and does not repair output. It distinguishes accepted enum capitalization and duration text from omissions and wrong values. Free-text synonyms and missing target qualifiers are not automatically forgiven.','',
          '| Track | Decoder | Strict F1 | Normalized F1 | JSON / schema valid | New unmentioned slots | Corrected horizon / 32 | Fully correct state after correction / 32 | Unknown valid/no updates / 32 | Median / p95 (s) |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for track in ('natural_component','natural_rollout'):
        for arm in arms:
            block=result['tracks'][track][arm]['all'];m=block['strict'];n=block['supplementary_normalized'];lat=m['extract_latency_ms']
            text.append(f'| {track} | {arm} | {m["nonintent"]["f1"]:.3f} | {n["schema_gated_normalized_slot_value_status"]["f1"]:.3f} | {m["extract_json_valid"]["rate"]:.3f} / {m["extract_schema_valid"]["rate"]:.3f} | {m["new_unmentioned_slot_count"]} | {n["correction_horizon_after_reducer"]["numerator"]} | {m["correction_flag_and_reducer_transition_accuracy"]["numerator"]} | {m["unknown_no_updates_and_valid"]["numerator"]} | {lat["p50"]/1000:.3f} / {lat["p95"]/1000:.3f} |')
    text+=['','Local corrected-horizon success and preservation of other prior slots are separate from matching the entire gold accumulated state. A correct correction can coexist with missing initial facts. Unknown tests measure abstention and prior-state retention, not generated clarification-question quality. Unmentioned slots are annotation-relative flags, not independently adjudicated semantic hallucinations.','',
           'See summary.csv for weather, inflation, stocks and crypto; analysis.json for full counts, normalized diagnostics, paired latency and 10,000-resample conversation bootstrap intervals; slot-error-audit.csv for every expected/emitted slot. Confidence intervals reflect resampling these development conversations, not unseen users or prompt-selection uncertainty. No fine-tuning decision is automated.','']
    (output/'README.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')
    hashes={p.name:{'sha256':audit.sha256(p),'bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()}
    audit.write_json(output/'artifact-hashes.json',hashes)
    print('\n'.join(text),flush=True)

if __name__=='__main__':analyze(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
