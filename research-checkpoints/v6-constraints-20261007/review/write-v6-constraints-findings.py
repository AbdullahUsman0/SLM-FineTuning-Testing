"""Summarize the completed frozen study without modifying its reports or labels."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path(r'D:\SLM\SLM-FineTuning-Testing')
RUN=Path(r'D:\SLM\FYP-model-runs\v6-constraints-20261007')
DEST=REPO/'research-checkpoints/v6-constraints-20261007'
analysis=json.loads((DEST/'results/analysis.json').read_text())
verification=json.loads((RUN/'independent-review/independent-verification.json').read_text())
failures=json.loads((RUN/'independent-review/failure-review.json').read_text())
review=DEST/'review';review.mkdir(exist_ok=False)
for folder in ('independent-review','comparison-refined-audit'):
    for p in (RUN/folder).iterdir():shutil.copyfile(p,review/p.name)
shutil.copyfile(RUN/'analysis-recovery.json',review/'analysis-recovery.json')
shutil.copyfile(Path(__file__).with_name('verify_v6_constraints.py'),review/'verify-v6-constraints.py')
shutil.copyfile(REPO/'scripts/refine-v6-content-audit.py',review/'refine-v6-content-audit.py')

def block(track,arm,domain='all'):return analysis['tracks']['natural_'+track][arm][domain]
criteria={}
for track in ('component','rollout'):
    left=block(track,'original');right=block(track,'constrained');l=left['strict'];r=right['strict']
    criteria[track]={
      'target_96_of_96_application_schema_valid':r['extract_schema_valid']['numerator']==96,
      'strict_f1_not_lower':r['nonintent']['f1']>=l['nonintent']['f1'],
      'normalized_f1_not_lower':right['supplementary_normalized']['schema_gated_normalized_slot_value_status']['f1']>=left['supplementary_normalized']['schema_gated_normalized_slot_value_status']['f1'],
      'full_state_correction_accuracy_not_lower':r['correction_flag_and_reducer_transition_accuracy']['numerator']>=l['correction_flag_and_reducer_transition_accuracy']['numerator'],
      'valid_unknown_abstention_not_lower':r['unknown_no_updates_and_valid']['numerator']>=l['unknown_no_updates_and_valid']['numerator'],
      'unmentioned_slot_count_not_higher':r['new_unmentioned_slot_count']<=l['new_unmentioned_slot_count']}
passed=all(all(v.values()) for v in criteria.values())
(review/'criterion-assessment.json').write_text(json.dumps({'frozen_criteria':criteria,'all_passed':passed,'deployment_adopted':False,'training_started':False,'case_human_review':'pending'},indent=2)+'\n',encoding='utf-8')

text=['# V6 generation-time schema enforcement: completed findings','',
('The constrained candidate met all frozen development-screen criteria. Independent human review and an unseen-case test remain necessary before adoption.' if passed else 'The constrained candidate failed one or more frozen development-screen criteria. Keep the original decoding baseline; do not adopt this candidate or infer a need for more training from structural validity alone.'),
'No production default or adapter weights were changed. This study tests one fixed decoding configuration, not a new fine-tuning round.','',
'## Matched design','',
'384 new extractions: original versus constrained decoding on 32 three-turn validation conversations (eight each weather, inflation, stocks and crypto), using gold prior states in the component track and each arm’s own accumulated state in rollout. Both arms used the byte-identical original prompt, same final-step v6 adapter, fixed script, unchanged labels, strict application validator and greedy generation settings. Order alternated by conversation in one serial BF16 CUDA session after equal unscored warmups.','',
'Run code was pinned at `2e25b15`; the base is `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; v6 final step 726. Seed 42, batch 1, 1,024 new-token limit, thinking disabled, four torch threads, RTX A4000. `xgrammar==0.2.7` and `apache-tvm-ffi==0.1.14.post1` were added; existing PyTorch, Transformers and other model libraries were unchanged. All scored inference used the local offline cache.','',
'The decoder compiles the existing exported structural JSON Schema, constructs one fresh matcher per generation and masks the full 248,320-token model vocabulary. It uses the actual model generation stop ID 248044. XGrammar’s CPU matcher creates the bitmask and its explicit `torch_native` backend masks CUDA logits; model weights remain on CUDA. Grammar setup/compilation time is separately recorded in execution evidence. There is no parsing repair, retry or per-case hint.','',
'Fixed field order is an explicit additional restriction: arbitrary-order mode accepted duplicate object keys in unscored contract tests. The fixed-order configuration passed those tests before protocol freeze. This restriction can affect generation, so the comparison isolates this complete decoder configuration rather than attributing every change to extra-key rejection alone. The grammar treats `candidate_value` as a JSON string; correctness of the JSON encoded inside that string, uniqueness of slot IDs and semantic grounding remain application checks.','',
'## Results','',
'Strict F1 retains schema gating and exact slot names, values and status. Supplementary normalized F1 applies only the existing pinned application normalizer to both sides, retains the same gate and status, and performs no output repair or arbitrary semantic paraphrase matching. “Schema valid” below means acceptance by the entire unchanged application wire contract, including inner JSON-text validation. It is stronger than outer JSON grammar compliance.','',
'| Track | Decoder | Strict F1 | Normalized F1 | Strict JSON / application schema valid | Raw typed corrections | Local normalized horizon correct | Full state after correction | Unknown valid, no updates | New unmentioned slots |',
'| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for track in ('component','rollout'):
    for arm in ('original','constrained'):
        b=block(track,arm);m=b['strict'];n=b['supplementary_normalized']
        text.append(f'| {track} | {arm} | {m["nonintent"]["f1"]:.3f} | {n["schema_gated_normalized_slot_value_status"]["f1"]:.3f} | {m["extract_json_valid"]["numerator"]}/96 / {m["extract_schema_valid"]["numerator"]}/96 | {m["correction_flag_and_exact_value_accuracy"]["numerator"]}/32 | {n["correction_horizon_after_reducer"]["numerator"]}/32 | {m["correction_flag_and_reducer_transition_accuracy"]["numerator"]}/32 | {m["unknown_no_updates_and_valid"]["numerator"]}/32 | {m["new_unmentioned_slot_count"]} |')
text+=['','A locally correct horizon correction can coexist with incomplete initial requirements. Component correction starts with gold prior state, so its full-state success cannot establish reliable initial extraction. Unknown tests measure valid abstention and prior-state retention, not generated clarification-question quality. New unmentioned slots are annotation-relative flags requiring semantic human review.','',
'| Rollout domain | Original strict F1 | Constrained strict F1 | Original application schema valid | Constrained application schema valid |',
'| --- | ---: | ---: | ---: | ---: |']
for domain in ('weather','inflation','stocks','crypto'):
    l=block('rollout','original',domain)['strict'];r=block('rollout','constrained',domain)['strict']
    text.append(f'| {domain} | {l["nonintent"]["f1"]:.3f} | {r["nonintent"]["f1"]:.3f} | {l["extract_schema_valid"]["numerator"]}/24 | {r["extract_schema_valid"]["numerator"]}/24 |')
text+=['','| Track | Candidate minus original F1 | 95% conversation-bootstrap interval | Paired mean latency difference, seconds |',
'| --- | ---: | --- | ---: |']
for track,d in analysis['paired_differences'].items():
    lo,hi=d['cluster_bootstrap_primary_f1_delta']['delta_f1_ci95']
    text.append(f'| {track.removeprefix("natural_")} | {d["strict_f1_delta"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {d["mean_paired_latency_delta_ms"]/1000:+.3f} |')
text+=['','Intervals use 10,000 resamples of the 32 authored conversation clusters. They describe uncertainty within this development cohort; they do not account for unseen users or configuration-selection uncertainty.','',
'## Failures and independent grammar checks','']
for name,result in failures['reports'].items():
    text.append(f'- `{name}`: {result["invalid_calls"]} application-invalid calls; observed categories: `{json.dumps(result["reason_counts"],sort_keys=True)}`. Categories may overlap.')
text+=['','Independent character-level checks against the same frozen grammar, using no model weights or repair:','']
for track,result in verification['independent_grammar_checks'].items():text.append(f'- `{track}`: `{json.dumps(result,sort_keys=True)}`.')
text+=['','Every failed call remains in the scored denominator. See failure-review.json for the complete raw trace of each invalid response. Content-assessment.csv separates readable slot omissions from strict JSON failures and normalization-only differences. A grammar-valid object can still omit stated facts, invent facts or contain invalid inner JSON; only the unchanged application gate determines whether it is applied.','',
'All 18 constrained component failures and 19 constrained rollout failures were invalid JSON text inside candidate_value, despite valid outer objects. Component failures comprised 14 initial turns and four corrections; rollout failures comprised 14 initial turns and five corrections. Strict JSON validity therefore reached 96/96 on each track, while application validity declined.','',
'The readable-content audit finds 55 initial slot omissions in each constrained track, including 24 missing target units, ten initial horizons, nine observation frequencies and four timestamp columns. These counts do not certify schema-invalid emissions as usable. Rollout also has 16 new-unmentioned-slot flags on correction turns and six on unknown turns; some corrections add target columns, units or dates rather than only changing the horizon. Local corrected-horizon success fell from 32/32 to 12/32, and horizon correction with all other prior slots preserved fell from 32/32 to 8/32. Semantic error flags remain subject to human adjudication.','',
'## Response time','',
'| Track | Decoder | Mean / median / p95 seconds | Generation-limit calls |',
'| --- | --- | ---: | ---: |']
for track in ('component','rollout'):
    for arm in ('original','constrained'):
        b=block(track,arm);lat=b['strict']['extract_latency_ms']
        text.append(f'| {track} | {arm} | {lat["mean"]/1000:.3f} / {lat["p50"]/1000:.3f} / {lat["p95"]/1000:.3f} | {b["generation_limit_calls_from_recorded_token_count"]} |')
text+=['','Timing includes warmed local extraction and synchronized CUDA execution, including matcher/mask overhead. It excludes one-time model loading and compilation and is not complete live-conversation or external-data API latency. There is no cross-session speed claim.','',
'## Verification and decision','',
f'All {len(verification["artifacts"])} hashed result artifacts independently matched disk SHA-256, size and published Git blobs at `{verification["published_commit"]}`. Four gzip files decode losslessly to the original raw reports, with 96 unique matched records each. Current-arm gold labels and component contexts match; historical labels also match. Original-control repeat checks against the previous study: `{json.dumps(verification["control_repeat"],sort_keys=True)}`. All calls record one generation and zero retries. Adapter SHA-256 values were independently verified unchanged after inference.','',
'The shared evaluator sanitizes away the optional per-token mask-step trace extension; the per-call backend and model-call counts are retained. The pinned decoder code and independent output-language checks are preserved, without claiming a token-by-token execution trace. No raw output, label or primary score was repaired or changed.','',
'The first supplementary analysis stopped because normalization of four invented forecast_start date emissions returned Python datetime values that the audit did not serialize. A separate analysis checkout at 2272807 uses the existing state-snapshot ISO serializer. All 24 relevant tests passed, and supplementary metrics on all 768 records from the two previous prompt studies remained unchanged. Original inference at 2e25b15 and the incomplete first analysis were retained; no generation was repeated. The raw reports and strict metric implementation are unchanged. See [review/analysis-recovery.json](review/analysis-recovery.json); reproduction preserves both frozen inference-era and executed analysis code.','',
'The frozen criterion outcomes are recorded in [review/criterion-assessment.json](review/criterion-assessment.json). Keep the original control as the reference. If further decoding work is pursued, distinguish enforcing the inner JSON-text contract from improving the extraction of facts, and evaluate both with the same application gate. This experiment does not establish that another fine-tuning round is necessary.','',
'These are agent-authored development cases already inspected in earlier studies. Independent human adjudication and unseen conversations remain pending. No training, paid APIs or sealed final-label access occurred. Default interactive inference is unchanged.','',
'See [results/README.md](results/README.md), [results/summary.csv](results/summary.csv) and [results/analysis.json](results/analysis.json) for all metrics, cases and raw outputs. Checks and errors are in [review/independent-verification.json](review/independent-verification.json), [review/failure-review.json](review/failure-review.json) and [review/content-assessment.csv](review/content-assessment.csv). The frozen [protocol](../../evaluation/v6-constraints-20261007/protocol.json) records exact settings and limitations. Scripts under review retain machine-specific local paths as provenance, not portable inference instructions.','',
'Primary references for the backend: [XGrammar installation](https://xgrammar.mlc.ai/docs/latest/start/installation.html), [engine integration](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html). Exact installed 0.2.7 package code and the locally passing tests govern this run.','']
path=DEST/'findings.md';assert not path.exists();path.write_text('\n'.join(text),encoding='utf-8',newline='\n')
shutil.copyfile(Path(__file__),review/'write-v6-constraints-findings.py')
hashes={str(p.relative_to(DEST)).replace('\\','/'):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in [path,*review.iterdir()] if p.is_file()}
(DEST/'review-artifact-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
print('Created findings. Frozen criteria all passed:',passed)
