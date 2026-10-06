"""Publish interpretation separately from immutable evaluation artifacts."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path(r'D:\SLM\SLM-FineTuning-Testing')
RUN=Path(r'D:\SLM\FYP-model-runs\v6-schema-fix-20261006')
DEST=REPO/'research-checkpoints/v6-schema-fix-20261006'
current=json.loads((DEST/'results/analysis.json').read_text())
historical=json.loads((REPO/'research-checkpoints/v6-prompt-audit-20261006/results/analysis.json').read_text())
review=DEST/'review';review.mkdir(exist_ok=False)
for p in (RUN/'independent-review').iterdir():shutil.copyfile(p,review/p.name)
for p in (RUN/'comparison-refined-audit').iterdir():shutil.copyfile(p,review/p.name)
shutil.copyfile(Path(__file__).with_name('verify_v6_schema_fix.py'),review/'verify-v6-schema-fix.py')
shutil.copyfile(REPO/'scripts/refine-v6-content-audit.py',review/'refine-v6-content-audit.py')

def unique_pairs(items):
    obj={}
    for k,v in items:
        if k in obj:raise ValueError('duplicate_object_key:'+k)
        obj[k]=v
    return obj
duplicate_review={}
for p in sorted((DEST/'results').glob('*.json.gz')):
    records=json.loads(gzip.decompress(p.read_bytes()))['records'];rows=[]
    for r in records:
        if r['raw_json_valid']:continue
        try:json.loads(r['raw_output'],object_pairs_hook=unique_pairs);reason='independent_parser_accepts'
        except (ValueError,json.JSONDecodeError) as e:reason=str(e)
        rows.append({'key':r['key'],'reason':reason,'completion_tokens':r['completion_tokens']})
    duplicate_review[p.name]={'strict_json_failures':rows,'hit_generation_limit_calls':sum(r['hit_generation_limit'] for r in records)}
(review/'duplicate-key-review.json').write_text(json.dumps(duplicate_review,indent=2)+'\n',encoding='utf-8')

def block(track,arm):
    return (historical if arm=='original' else current)['tracks'][track][arm]['all']

text=['# Focused v6 schema reminder: reject the candidate','',
'The single frozen format-only prompt addition failed. Retain the original prompt as the baseline; adopt neither this schema reminder nor the earlier revised prompt as a proven replacement. No training or production default change was made. The result supports testing generation-time enforcement of the existing output schema next, with the same strict validator and a matched semantic evaluation. Such enforcement has not been installed or evaluated here and would not by itself prove correct slot values.','',
'## What was tested','',
'384 new scored extractions: revised control versus `schema_fixed`, each on 32 three-turn conversations in weather, inflation, stocks and crypto, under both gold-prior component contexts and each arm’s own accumulated rollout state. The 1,165-character addition specifies allowed keys, separate slot updates and no duplicates; its prefix is byte-identical to the prior revised prompt. This was one fixed candidate, frozen before inference. No examples with case answers were added. Labels, weights, validator and greedy decoding remained unchanged. Arms alternated order by conversation after equal warmups in one BF16 CUDA session.','',
'Base: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; final-step v6 adapter step 726. Seed 42, batch 1, 1,024 new-token limit, thinking disabled, RTX A4000. Run code pinned at `161f961`; raw reports published at `5b73d8d1a120b7a10b67011550eee33e3897e0f3`. The original prompt below is a historical reference from the prior study, not a third same-session arm.','',
'## Results','',
'Strict F1 measures exact slot name, value and status and retains the schema gate. Normalized F1 applies the pinned application normalizer to both sides, retains the same gate and status, and does not repair outputs or forgive arbitrary free-text differences. Each row has 96 calls; correction and unknown tests have 32 cases.','',
'| Track | Prompt | Strict F1 | Normalized F1 | Strict JSON / schema valid | Raw typed corrections | Local normalized horizon correct | Full state after correction | Unknown valid, no updates | New unmentioned slots |',
'| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for track in ('natural_component','natural_rollout'):
    for arm in ('original','revised','schema_fixed'):
        b=block(track,arm);m=b['strict'];n=b['supplementary_normalized']
        label={'original':'Original (historical)','revised':'Revised (current control)','schema_fixed':'Schema reminder (candidate)'}[arm]
        text.append(f'| {track.removeprefix("natural_")} | {label} | {m["nonintent"]["f1"]:.3f} | {n["schema_gated_normalized_slot_value_status"]["f1"]:.3f} | {m["extract_json_valid"]["numerator"]}/96 / {m["extract_schema_valid"]["numerator"]}/96 | {m["correction_flag_and_exact_value_accuracy"]["numerator"]}/32 | {n["correction_horizon_after_reducer"]["numerator"]}/32 | {m["correction_flag_and_reducer_transition_accuracy"]["numerator"]}/32 | {m["unknown_no_updates_and_valid"]["numerator"]}/32 | {m["new_unmentioned_slot_count"]} |')
text+=['','Component full-state correction starts with gold prior state, so its 32/32 score does not demonstrate reliable capture of initial facts. In rollout, the candidate correctly changed the local horizon in all cases and preserved all other prior slots, but **0/32** resulting states matched the complete gold requirements. Prior slots can be missing because the first response was rejected. Unknown abstention likewise does not establish a complete state or good generated clarification questions.','',
'The candidate missed the frozen success criteria: schema validity was 67/96 on each track, below both the original reference target of 93/96 and revised control’s 87/96. Normalized F1 fell from 0.670 to 0.241; rollout full-state accuracy fell from 8/32 to 0/32. It retained 32/32 valid unknown abstentions and zero annotation-relative new unmentioned slots, which is insufficient to offset these failures.','',
'| Rollout domain | Original (historical) F1 | Revised control F1 | Schema reminder F1 | Revised schema valid | Reminder schema valid |',
'| --- | ---: | ---: | ---: | ---: | ---: |']
for domain in ('weather','inflation','stocks','crypto'):
    old=historical['tracks']['natural_rollout']['original'][domain]['strict']
    left=current['tracks']['natural_rollout']['revised'][domain]['strict'];right=current['tracks']['natural_rollout']['schema_fixed'][domain]['strict']
    text.append(f'| {domain} | {old["nonintent"]["f1"]:.3f} | {left["nonintent"]["f1"]:.3f} | {right["nonintent"]["f1"]:.3f} | {left["extract_schema_valid"]["numerator"]}/24 | {right["extract_schema_valid"]["numerator"]}/24 |')
text+=['','The primary F1 difference, candidate minus revised control, is -0.424 in component (95% conversation-bootstrap interval [-0.554, -0.289]) and -0.373 in rollout ([-0.495, -0.245]). Both intervals exclude zero within this development cohort. These 10,000-resample intervals do not measure unseen-user generalization or prompt-selection uncertainty.','',
'## Failure mechanism','',
'Every invalid response occurred on the initial turn. On each track, the revised control had nine failures: two duplicated JSON object keys, six updates with an unsupported `type` field, and one merged update with extra `forecast_horizon`, `frequency` and `target_unit` keys. The reminder had 29 failures: sixteen responses added `type`, five added `update_type`, and eight duplicated object keys (seven `evidence_text`, one `candidate_value`). The format instructions did not prevent the behavior they described. None of the 384 calls hit the generation limit.','',
'“Strict JSON” includes rejecting duplicate object keys. These strings can be accepted by a permissive parser that discards an earlier key; they remain invalid under the frozen contract. The supplementary readable-content audit does not repair or apply them. It finds 24 readable slot omissions and 11 expected-slot entries unassessable due to duplicate-key JSON in the control, versus 15 readable omissions and 45 unassessable entries in the candidate. The smaller readable omission count is not evidence of improvement because substantially more candidate content is unassessable. Different free-text values remain unadjudicated semantic differences, rather than automatically counted as concept errors.','',
'## Timing and verification','',
'| Current-session arm | Component median / p95 seconds | Rollout median / p95 seconds |',
'| --- | ---: | ---: |']
for arm in ('revised','schema_fixed'):
    c=block('natural_component',arm)['strict']['extract_latency_ms'];r=block('natural_rollout',arm)['strict']['extract_latency_ms']
    text.append(f'| {arm} | {c["p50"]/1000:.3f} / {c["p95"]/1000:.3f} | {r["p50"]/1000:.3f} / {r["p95"]/1000:.3f} |')
text+=['','The candidate’s paired mean latency increased by 0.434 seconds in component and 0.508 seconds in rollout. Medians alone conceal this tail increase. No same-session speed claim is made against the historical original prompt. Timings measure local synchronized extraction, not a complete live conversation or external API calls.','',
'Independent verification checked all 11 hashed artifacts against disk, size and published Git blobs; all four gzip reports decode losslessly to the raw reports and contain 96 unique matched records. Gold labels match across arms and historical references; gold component contexts match across current arms. **All 192 revised-control raw outputs, contexts, labels and validity outcomes exactly repeated the prior study.** Adapter model/config SHA-256 values independently matched the frozen protocol after inference. Optional `model_calls` and `retry_count` record fields are null; the pinned runner has one extraction per turn and no repair/retry loop.','',
'The frozen protocol accidentally retained an old descriptive order string. Its arms field and executed alternating order were correct. The original bytes were retained, with the correction documented in [protocol-errata.json](../../evaluation/v6-schema-fix-20261006/protocol-errata.json).','',
'All cases were agent-authored and reviewed by the agent; independent human review is pending. These are the same development conversations already inspected to design the prompts. No unseen-case claim, sealed final-label access, paid API use or further training occurred. Keep these failed results as evidence and finish human case/error review before drawing broader model-quality or training conclusions.','',
'Raw outputs, metrics, timings and exact run code are in [results/README.md](results/README.md), [results/summary.csv](results/summary.csv) and [results/analysis.json](results/analysis.json). Independent integrity checks and every structurally invalid call are in [review/independent-verification.json](review/independent-verification.json), [review/failure-review.json](review/failure-review.json) and [review/duplicate-key-review.json](review/duplicate-key-review.json). The readable-content classification is in [review/content-assessment.csv](review/content-assessment.csv). Review scripts are preserved as provenance; they contain local run paths and are not standalone portable inference commands.','']
path=DEST/'findings.md';assert not path.exists();path.write_text('\n'.join(text),encoding='utf-8',newline='\n')
shutil.copyfile(Path(__file__),review/'write-v6-schema-fix-findings.py')
hashes={str(p.relative_to(DEST)).replace('\\','/'):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in [path,*review.iterdir()] if p.is_file()}
(DEST/'review-artifact-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
print('Created findings and review artifacts:',len(hashes))
