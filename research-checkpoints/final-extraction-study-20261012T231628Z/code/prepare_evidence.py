"""Verify historical evidence and freeze model/runtime identities; no inference."""
from collections import Counter
import gzip
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile
import runtime as rt


def main():
    public = rt.CHECKPOINT
    intake = rt.read(public / 'workspace-intake.json')
    private = Path(intake['private'])
    reference = rt.ComparisonPolicy()
    old_lock = rt.read(reference.directory / 'freeze-manifest.json')
    assert old_lock['fpy_revision'] == rt.FPY_REVISION
    paths = list((rt.BUNDLE / 'local_slm_lab').glob('*.py')) + list(rt.FPY.rglob('*.py'))
    paths += [reference.directory / name for name in old_lock['files_sha256']]
    runtime_lock = {'fpy_revision': rt.FPY_REVISION, 'source_bundle_sha256':
        rt.sha(Path(intake['repo']) / 'research-checkpoints/v7-approved-policy-20261009/v7-approved-policy-development-20261009-r8.zip'),
        'bundle_files_sha256': {p.relative_to(rt.BUNDLE).as_posix(): rt.sha(p) for p in sorted(paths)},
        'parser_dependencies': old_lock['parser_dependencies'], 'historical_policy_fingerprint': reference.fingerprint,
        'sealed_final_accessed': False}
    assert runtime_lock['source_bundle_sha256'] == '9f8424895c4f7591b65a5f1aaa7e87104f8035671c272177cd8b6a5bb8bba22b'
    rt.atomic(public / 'runtime-lock.json', runtime_lock, exclusive=True)
    rt.verify_sources()
    import psutil
    import torch
    environment = {'recorded_at': rt.now(), 'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'RAM_bytes': psutil.virtual_memory().total,
        'RAM_available_bytes': psutil.virtual_memory().available,
        'packages': {name: md.version(name) for name in ('torch','transformers','peft','accelerate',
            'safetensors','openai','pydantic','jsonschema','numpy','psutil')},
        'cuda_available': torch.cuda.is_available(), 'torch_cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'VRAM_bytes': torch.cuda.get_device_properties(0).total_memory,
        'BF16_supported': torch.cuda.is_bf16_supported(),
        'nvidia_smi': subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
        'openai_credential_available_in_process': bool(os.environ.get('OPENAI_API_KEY')),
        'working_gpu_environment_upgraded': False, 'paid_API_calls': 0}
    assert environment['cuda_available'] and environment['BF16_supported']
    rt.atomic(public / 'environment.json', environment, exclusive=True)

    repo = Path(intake['repo'])
    v6_provenance = rt.read(repo / 'research-checkpoints/v6-training-20261003/results/completion.json')
    v6 = Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter')
    names = ('adapter_model.safetensors', 'adapter_config.json')
    wanted_v6 = {'adapter_model.safetensors': 'e7add7b8b02d24615ed89067c7761c3985a4a6afb7c84ca2a72e3b06fd84e74d',
                 'adapter_config.json': '38f60de9678324e91fa882ed86c70d09e4a39b27ae28884374ddc3e6fece15ef'}
    assert {name: rt.sha(v6 / name) for name in names} == wanted_v6
    # Require the provenance to contain each independently calculated digest.
    assert all(value in rt.dumps(v6_provenance) for value in wanted_v6.values())
    archive = Path.home() / 'Downloads/qwen35-2b-lora-v7-r8-pilot-20261010T002204Z-adapter.zip'
    wanted_zip = 'df34422e42fe7c03a4cac3160876061592b37e3e2ae974c5c97c099742a59da8'
    assert rt.sha(archive) == wanted_zip
    v7_container = private / 'models/v7-extracted'
    v7_container.mkdir(exist_ok=False)
    allowed = {'best-adapter/adapter_model.safetensors', 'best-adapter/adapter_config.json',
               'completion.json', 'adapter-files.json', 'README.md'}
    with zipfile.ZipFile(archive) as package:
        assert set(package.namelist()) == allowed and package.testzip() is None
        for name in sorted(allowed):
            path = v7_container / name
            assert path.resolve().is_relative_to(v7_container.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            with package.open(name) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target)
    v7 = v7_container / 'best-adapter'
    v7_provenance = rt.read(v7_container / 'completion.json')
    assert v7_provenance['status'] == 'complete' and v7_provenance['optimizer_steps'] == 65
    wanted_v7 = {'adapter_model.safetensors': 'a9ab464ae78f62e94b3dbb30b3d763563e5b7802dc437bb009acfae9b2a1a1db',
                 'adapter_config.json': 'fcbed9d64ab43e0a8fe5382bd7db03cf73cea6e6281d9d0950fe4537678c5f1e'}
    assert v7_provenance['adapter_sha256'] == wanted_v7 == {name: rt.sha(v7 / name) for name in names}
    base = Path(r'D:\SLM\hf-cache\hub\models--Qwen--Qwen3.5-2B\snapshots') / rt.BASE_REVISION / 'model.safetensors-00001-of-00001.safetensors'
    base_digest = rt.sha(base)
    assert base_digest == 'aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1'
    models = {'frozen_at': rt.now(), 'base_model': 'Qwen/Qwen3.5-2B', 'base_revision': rt.BASE_REVISION,
        'base_weights_sha256': base_digest, 'fpy_revision': rt.FPY_REVISION,
        'adapters': {'v6': {'sha256': wanted_v6, 'provenance_sha256': rt.digest(v6_provenance)},
                    'v7': {'sha256': wanted_v7, 'ZIP_sha256': wanted_zip, 'provenance_sha256': rt.digest(v7_provenance),
                           'selection': 'Frozen experimental final-step 65 pilot; no post-final checkpoint selection'}},
        'OpenAI_model': 'gpt-5-mini-2025-08-07', 'training_authorized': False}
    rt.atomic(public / 'models-freeze.json', models, exclusive=True)
    rt.atomic(private / 'runtime-config.json', {'checkpoint': str(public), 'private': str(private),
        'adapters': {'v6': str(v6), 'v7': str(v7)}, 'base_weights': str(base),
        'HF_HOME': r'D:\SLM\hf-cache', 'python': intake['python']}, exclusive=True)
    print('Pinned sources, parser versions, GPU inventory, v6 provenance, v7 ZIP/files and base weights verified.',flush=True)

    old = repo / 'research-checkpoints/policy-openai-slm-matched-20261008'
    manifest = rt.read(old / 'checkpoint-manifest.json')
    for name, identity in manifest['files'].items():
        assert rt.sha(old / name) == identity['sha256'], name
    hosted = {track: json.loads(gzip.decompress((old / f'hosted-source/{track}-report.json.gz').read_bytes()))
              for track in ('component','rollout')}
    prior_summary = rt.read(old / 'summary.json')
    gpu_archive = repo / 'research-checkpoints/policy-comparison-remote-20261007/reviewed-policy-gpu-results-20261007.zip'
    reports = {}
    with zipfile.ZipFile(gpu_archive) as package:
        assert package.testzip() is None
        for name, arm in (('stock','stock_2B'),('v5','v5'),('v6','v6')):
            for track in ('component','rollout'):
                reports[arm,track] = json.loads(package.read(f'training-runs/policy-{name}-bf16-20261007/{track}-report.json'))
    for track, report in hosted.items(): reports['openai',track] = report
    rows = []
    for (arm,track), report in reports.items():
        records = report['records']
        assert len(records) == 93 and len({tuple(r['key']) for r in records}) == 93
        metric = rt.score_records(records, [s.slot_id for s in rt.load_schema().slots])
        assert metric['nonintent'] == report['metrics']['nonintent']
        rows.append({'model': arm, 'track': track, **metric['nonintent']})
        for record in records:
            trace = record['policy_traces'][-1]
            trace = (trace.get('attempts') or [trace])[-1]
            admission = reference.parse(record['raw_output'], 'extract', response_status=trace.get('response_status','completed'),
                refused=trace.get('refused',False), hit_generation_limit=trace.get('hit_generation_limit',False))
            assert admission.predicted == record['predicted'] and admission.admitted == record['parser_admitted']
    for track in ('component','rollout'):
        left = {tuple(r['key']):r for r in hosted[track]['records']}
        for arm in ('stock_2B','v5','v6'):
            right = {tuple(r['key']):r for r in reports[arm,track]['records']}
            assert left.keys() == right.keys()
            for key in left:
                assert left[key]['expected'] == right[key]['expected']
                assert left[key]['context']['message'] == right[key]['context']['message']
                if track == 'component': assert left[key]['context'] == right[key]['context']
    rt.atomic(public / 'evidence/historical-matched-verification.json', {'status':'verified',
        'scored_calls':744, 'rows':rows,'human_review_completed':prior_summary['human_review_completed'],
        'review_interpretation':'31 historical development conversations were agent-reviewed; machine-readable independent-human review remains incomplete.',
        'source_checkpoint_files_verified':len(manifest['files']), 'old_policy_unchanged':True,
        'paid_API_calls':0,'sealed_final_accessed':False},exclusive=True)
    print('Historical 31-conversation metrics, admissions, component contexts and cohort matching reproduced.',flush=True)

    pilot = repo / 'research-checkpoints/v7-r8-pilot-20261009/results'
    pilot_zip = pilot / 'qwen35-2b-lora-v7-r8-pilot-20261010T002204Z-evaluation.zip'
    assert rt.sha(pilot_zip) == 'a253dc06fe165cbb2c516903e3cd928344ed010407747f51a73a9b494a74bb7a'
    findings = {'source_ZIP_sha256':rt.sha(pilot_zip), 'scope':'Historical synthetic development cohort; not the new study',
                'tracks':{}, 'paid_API_calls':0,'sealed_final_accessed':False}
    with zipfile.ZipFile(pilot_zip) as package:
        assert package.testzip() is None
        for track in ('component','rollout'):
            arms = {}
            for arm in ('v6','v7'):
                report = json.loads(package.read(f'evaluation/{arm}/{track}-report.json'))
                records = report['records']
                assert len(records) == 1040
                assert rt.score_records(records,[s.slot_id for s in rt.load_schema().slots])['nonintent'] == report['metrics']['nonintent']
                extra = Counter()
                wrong_evidence=[]; normalization_matches=0; offending_updates=0
                for record in records:
                    expected = {u['slot_id'] for u in record['expected']['updates']}
                    raw_ids = {u['slot_id'] for u in (record.get('raw_value') or {}).get('updates',[]) if isinstance(u,dict) and 'slot_id' in u}
                    raw_ids |= {u['slot_id'] for u in (record.get('emitted') or {}).get('updates',[])}
                    extra.update(raw_ids - expected)
                    if record['parser_admitted']:
                        for update in record['predicted']['updates']:
                            evidence=update['evidence_text']; message=record['context']['message']
                            if not evidence.strip() or evidence not in message:
                                offending_updates+=1
                                normalization_matches+=bool(evidence.strip() and ' '.join(evidence.split()) in ' '.join(message.split()))
                                wrong_evidence.append(record['key'])
                structural = [{'key':r['key'],'language':r['language'],'behavior':r['behavior'],
                    'hit_generation_limit':r['policy_traces'][-1].get('hit_generation_limit',False),
                    'raw_output':r['raw_output']} for r in records if not r['raw_json_valid']]
                arms[arm] = {'exact_F1':report['metrics']['nonintent']['f1'], 'extra_slot_flags_by_slot':dict(extra),
                    'extra_slot_flags_total':sum(extra.values()),'invalid_JSON_calls':structural,
                    'nonexact_evidence_calls':len({tuple(k) for k in wrong_evidence}),
                    'nonexact_evidence_updates':offending_updates,
                    'whitespace_collapsed_diagnostic_matches':normalization_matches}
            findings['tracks'][track]=arms
    rt.atomic(public / 'evidence/v7-failure-reproduction.json',findings,exclusive=True)
    print('V7 pilot raw failures and evidence/extra-slot counts reproduced without editing historical outputs.',flush=True)


if __name__ == '__main__':
    main()
