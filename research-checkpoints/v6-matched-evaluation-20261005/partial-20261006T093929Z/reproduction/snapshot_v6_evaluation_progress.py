"""Publishable immutable snapshot; never modifies the running evaluation."""
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

REPO = Path(r'D:\SLM\SLM-FineTuning-Testing')
RUN = Path(r'D:\SLM\FYP-model-runs\matched-v6-evaluation-20261005-r2')
EXECUTED = Path(r'D:\SLM\SLM-v6-evaluation-20261005')
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location('matched_analysis', REPO/'scripts/analyze-v6-matched.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
destination = REPO/'research-checkpoints/v6-matched-evaluation-20261005'/('partial-'+stamp)
destination.mkdir(parents=True, exist_ok=False)

def digest(payload):
    return hashlib.sha256(payload).hexdigest()

def write_json(name, value):
    (destination/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')

status = json.loads((RUN/'status.json').read_text())
assert status['status'] == 'running'
fingerprint = status['fingerprint']
cohorts = {}
source_hashes = {}
coverage = {}
for track in analysis.TRACKS:
    cohorts[track] = {}
    for arm in analysis.ARMS:
        complete = {}
        for path in sorted((RUN/arm/track).glob('*.json')):
            if len(path.stem) != 4 or not path.stem.isdigit():
                continue
            payload = path.read_bytes()
            report = json.loads(payload)
            assert report['status'] == 'complete' and report['fingerprint'] == fingerprint
            assert report['arm'] == arm and report['track'] == track
            scenario = report['scenario_id']
            assert scenario not in complete
            complete[scenario] = report
            source_hashes[path.relative_to(RUN).as_posix()] = digest(payload)
        cohorts[track][arm] = complete
    shared = sorted(set.intersection(*(set(cohorts[track][arm]) for arm in analysis.ARMS)))
    coverage[track] = {
        'completed_scenarios_observed_per_arm': {arm:len(cohorts[track][arm]) for arm in analysis.ARMS},
        'matched_scenarios_in_snapshot': len(shared),
        'scenario_ids': shared,
        'full_track_scenario_count': 150 if track == 'frozen_component' else 32,
    }
    assert shared
    if track.startswith('natural'):
        assert len(shared) == 32
    for arm in analysis.ARMS:
        cohorts[track][arm] = [record for sid in shared for record in cohorts[track][arm][sid]['records']]

result = {'status':'partial', 'snapshot_utc':stamp, 'source_status_at_snapshot':status,
          'fingerprint':fingerprint, 'coverage':coverage, 'tracks':{},
          'training_started':False, 'sealed_final_accessed':False,
          'analysis_sha256':digest((REPO/'scripts/analyze-v6-matched.py').read_bytes()),
          'supplementary_readable_content_diagnostic':'Post hoc descriptive content scores do not repair outputs or change the primary operational score.',
          'sampling_warning':'Frozen coverage is an incomplete ordered subset. Do not infer full-study rankings or confidence intervals from it.'}
rows = []
errors = []
snapshot_calls = 0
for track in analysis.TRACKS:
    result['tracks'][track] = {}
    reference = cohorts[track]['base']
    for arm in analysis.ARMS:
        records = cohorts[track][arm]
        expected_keys = {tuple(r['key']):r for r in reference}
        observed_keys = {tuple(r['key']):r for r in records}
        assert expected_keys.keys() == observed_keys.keys()
        for key, record in observed_keys.items():
            assert record['gold_sha256'] == expected_keys[key]['gold_sha256']
            if track != 'natural_rollout':
                assert record['context_sha256'] == expected_keys[key]['context_sha256']
        snapshot_calls += len(records)
        report = {'status':'partial_study_snapshot', 'track_complete':track.startswith('natural'),
                  'fingerprint':fingerprint, 'arm':arm, 'track':track, 'records':records}
        payload = (json.dumps(report, ensure_ascii=False)+'\n').encode('utf-8')
        archive = gzip.compress(payload, mtime=0)
        assert gzip.decompress(archive) == payload
        (destination/f'{arm}-{track}.json.gz').write_bytes(archive)
        domains = {'all':records}
        domains.update({domain:[r for r in records if r['evaluation_domain'] == domain]
                        for domain in ('weather','inflation','stocks','crypto')})
        metrics = {domain:analysis.summarize(group) for domain,group in domains.items() if group}
        result['tracks'][track][arm] = metrics
        for domain,m in metrics.items():
            rows.append({'track':track,'arm':arm,'domain':domain,'calls':m['calls']['total'],
                         'slot_value_status_f1':m['nonintent']['f1'],
                         'slot_name_f1':m['slot_name_prf']['f1'],
                         'slot_value_f1':m['slot_value_prf_without_status']['f1'],
                         'readable_slot_value_f1':m['readable_slot_value_prf']['f1'],
                         'extract_json_valid':m['extract_json_valid']['rate'],
                         'extract_schema_valid':m['extract_schema_valid']['rate'],
                         'new_unmentioned_slots':m['new_unmentioned_slot_count'],
                         'wrong_stated_values':m['wrong_values_for_stated_slots'],
                         'correction_flag_and_values':m['correction_flag_and_exact_value_accuracy']['rate'],
                         'correction_flag_and_reducer_transition':m['correction_flag_and_reducer_transition_accuracy']['rate'],
                         'unknown_no_updates_valid':m['unknown_no_updates_and_valid']['rate'],
                         'unknown_state_retained_valid':m['unknown_retains_prior_state_and_valid']['rate'],
                         'extract_latency_mean_ms':m['extract_latency_ms']['mean'],
                         'extract_latency_p50_ms':m['extract_latency_ms']['p50'],
                         'extract_latency_p95_ms':m['extract_latency_ms']['p95']})
        for r in records:
            if r['task'] != 'extract':
                continue
            tp,fp,fn = analysis.confusion(r, nonintent=True)
            if not r['prediction_valid'] or fp or fn or r.get('test_kind') in ('correction','unknown'):
                errors.append({'arm':arm,'track':track,'domain':r['evaluation_domain'],
                               'scenario_id':r['scenario_id'],'turn':r['key'][2],
                               'test_kind':r.get('test_kind'),'valid':r['prediction_valid'],
                               'tp':tp,'fp':fp,'fn':fn,'latency_ms':r['latency_ms'],
                               'message':r['context']['message'],
                               'expected':json.dumps(r['expected'],ensure_ascii=False),
                               'raw_output':r.get('raw_output')})
result['matched_calls_in_snapshot'] = snapshot_calls
for name, data in [('summary.csv',rows),('errors-and-boundary-cases.csv',errors)]:
    with (destination/name).open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
write_json('analysis.json',result)
write_json('source-scenario-sha256.json',source_hashes)
shutil.copyfile(RUN/'run-manifest.json',destination/'run-manifest.json')
write_json('execution-evidence.json',{'sessions':{p.name:json.loads(p.read_text()) for p in (RUN/'sessions').glob('*.json')}})
reproduction = destination/'reproduction'
reproduction.mkdir()
for name in ('evaluate-v6-matched.py','prepare-v6-matched-evaluation.py'):
    shutil.copyfile(EXECUTED/'scripts'/name,reproduction/name)
shutil.copyfile(REPO/'scripts/analyze-v6-matched.py',reproduction/'analyze-v6-matched.py')
shutil.copyfile(Path(__file__),reproduction/Path(__file__).name)
readme = ['# Partial matched stock 2B / v5 / v6 evaluation','',
          f'Snapshot UTC: {stamp}. The full evaluation is still running. This is not the completed study or a decision to train again.',
          f'The worker had saved {status["completed_calls"]:,} of 2,469 calls at snapshot start; this archive contains {snapshot_calls:,} matched calls from complete scenarios only. In-flight calls and unmatched scenario fragments are excluded.', '',
          'Both natural tracks are complete: 32 conversations, eight each for weather, inflation, stocks and crypto, with an initial requirement, horizon correction and unknown-information turn. The frozen synthetic validation cohort is incomplete; only the intersection of completed scenario IDs across all three models is compared. Coverage and exact IDs are in analysis.json. Its ordered subset can be biased. Full-study confidence intervals and conclusions are deferred.', '',
          'Natural component calls use gold prior state. Natural rollout uses each model\'s own state and the same fixed user script. These are controlled diagnostics, pending independent human annotation review, rather than live-user trials. Source corpus human review and fuzzy-overlap review remain outstanding.', '',
          'Exact primary F1 requires valid wire schema plus exact slot name, JSON-decoded value and status. Slot-name and slot/value F1 are also reported. Readable-content F1 is a supplementary post hoc diagnostic for valid JSON with schema failures; it never repairs responses or enables state updates. Thus stock operational F1 can be zero despite partially readable content. JSON syntax and schema validity are separate.', '',
          'New unmentioned slots flag possible inventions against the annotations; repeated unchanged prior facts and wrong values for stated slots are counted separately. Correction metrics distinguish raw flag/value accuracy from common-reducer transition accuracy. Invalid outputs cannot certify safe unknown handling. Unknown-state retention includes intent and slots.', '',
          'All arms use the same pinned base revision, prompts, BF16 CUDA GPU, greedy generation, 1,024-token limit and serial execution. Native stock was checked against disabled-adapter stock before scoring; warmup calls are excluded. Response times exclude loading and adapter switching. Generation limits and full timing are in analysis.json. No new training, paid APIs or sealed final labels.', '',
          '| Track | Arm | Exact F1 | Readable value F1 | JSON valid | Schema valid | Extract median / p95 (s) |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for track in analysis.TRACKS:
    for arm in analysis.ARMS:
        m = result['tracks'][track][arm]['all']
        readme.append(f'| {track} | {arm} | {m["nonintent"]["f1"]:.3f} | {m["readable_slot_value_prf"]["f1"]:.3f} | {m["extract_json_valid"]["rate"]:.3f} | {m["extract_schema_valid"]["rate"]:.3f} | {m["extract_latency_ms"]["p50"]/1000:.3f} / {m["extract_latency_ms"]["p95"]/1000:.3f} |')
readme += ['', 'Study summary and per-domain comparisons: summary.csv. Raw expected labels beside responses: errors-and-boundary-cases.csv. Nine lossless compressed reports preserve contexts, labels, raw outputs, validation errors and latency. Source/code/model pins: run-manifest.json and reproduction/. Artifact integrity: artifact-hashes.json.', '',
           'The complete evaluation will be published separately under ../results after all 2,469 calls and integrity checks complete. Training loss is not evidence of conversational quality.','']
(destination/'README.md').write_text('\n'.join(readme),encoding='utf-8',newline='\n')
hashes = {p.relative_to(destination).as_posix():{'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size}
          for p in sorted(destination.rglob('*')) if p.is_file()}
write_json('artifact-hashes.json',hashes)
for name, expected in hashes.items():
    assert digest((destination/name).read_bytes()) == expected['sha256']
print(json.dumps({'destination':str(destination),'matched_calls':snapshot_calls,'coverage':coverage,'bytes':sum(x['bytes'] for x in hashes.values())}))
