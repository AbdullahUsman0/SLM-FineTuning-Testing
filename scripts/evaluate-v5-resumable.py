"""Resumable raw base/LoRA evaluation on the frozen development validation cohort.

Uses the original v5 provider, prompts and scorer. Completed scenarios are
immutable; interrupted scenarios are preserved and replayed. Never reads final.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_slm_lab.v5_eval import (
    build_call_plan, digest, evaluate, load_cases, load_schema, now, provenance,
    runtime_metadata, safe_error, sha256, strict_json,
)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


RAW = module('raw_runner', 'evaluate-v5.py')
DURABLE = module('durable_runner', 'evaluate-structured-output-resumable.py')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--provider', required=True, choices=('base', 'lora'))
    result.add_argument('--model', required=True)
    result.add_argument('--revision', required=True)
    result.add_argument('--adapter', type=Path)
    result.add_argument('--run-dir', required=True, type=Path)
    result.add_argument('--cases', type=Path, default=ROOT / 'corpus-v5/v5-20260919-r1/splits/validation.jsonl')
    result.add_argument('--study-manifest', type=Path, default=ROOT / 'evaluation/v5-structured-output/prepared-validation-v2/study-manifest.json')
    result.add_argument('--warmup-cases', type=Path, default=ROOT / 'corpus-v5/v5-20260919-r1/splits/smoke.jsonl')
    result.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    result.add_argument('--device', default='cuda')
    result.add_argument('--max-new-tokens', type=int, default=1024)
    result.set_defaults(split='validation')
    return result


def run(args):
    run_dir = RAW.absolute(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir = run_dir / 'scenarios'
    scenario_dir.mkdir(exist_ok=True)
    cases, input_metadata = load_cases(RAW.absolute(args.cases), split='validation')
    schema = load_schema()
    plan = build_call_plan(cases, schema)
    study = DURABLE.BASE.verify_frozen_study(args.study_manifest, input_metadata, plan)
    audit = provenance(args.provider)
    settings = RAW.settings_for(args, audit)
    identity = {'input_sha256': input_metadata['sha256'], 'study': study,
                'settings': settings, 'schema_sha256': digest(schema.model_dump(mode='json')),
                'runner_sha256': sha256(Path(__file__)),
                'raw_runner_sha256': sha256(ROOT / 'scripts/evaluate-v5.py'),
                'durable_helpers_sha256': sha256(ROOT / 'scripts/evaluate-structured-output-resumable.py'),
                'study_helpers_sha256': sha256(ROOT / 'scripts/evaluate-structured-output.py'),
                'warmup_sha256': sha256(RAW.absolute(args.warmup_cases)),
                'versions': audit['versions']}
    fingerprint = digest(identity)
    manifest_path = run_dir / 'run-manifest.json'
    manifest = strict_json(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    if manifest and manifest['run_fingerprint'] != fingerprint:
        raise ValueError('run input, code, versions, adapter or settings changed')
    combined_path = run_dir / 'combined-report.json'
    if manifest and combined_path.exists():
        report = strict_json(combined_path.read_text(encoding='utf-8'))
        if report.get('status') != 'complete' or report.get('run_fingerprint') != fingerprint:
            raise ValueError('incompatible combined report')
        return report
    provider, instrumentation = RAW.create_provider(args, schema)
    runtime = runtime_metadata(provider)
    artifacts = RAW.model_artifacts(args, provider)
    if manifest:
        if manifest['runtime_signature'] != DURABLE.runtime_signature(runtime):
            raise ValueError('actual runtime changed')
        if manifest['metadata']['model_artifacts'] != artifacts:
            raise ValueError('base model artifact hashes changed')
    metadata = {'input': input_metadata, 'settings': settings, 'settings_sha256': digest(settings),
                'provenance': audit, 'runner_sha256': identity['runner_sha256'],
                'schema_sha256': identity['schema_sha256'], 'study': study,
                'runtime': runtime, 'instrumentation': instrumentation,
                'model_artifacts': artifacts, 'run_fingerprint': fingerprint,
                'warmup_policy': 'first smoke scenario; unscored; repeated each session'}
    if manifest is None:
        manifest = {'manifest_version': 'v5-raw-resumable-1', 'created_at_utc': now(),
                    'run_fingerprint': fingerprint, 'identity': identity, 'metadata': metadata,
                    'runtime_signature': DURABLE.runtime_signature(runtime), 'scenario_count': len(cases)}
        DURABLE.write_exclusive(manifest_path, manifest)
    # evaluate captures ordinary provider/parse failures, including for stock base.
    warmup_cases, _ = load_cases(RAW.absolute(args.warmup_cases), split='validation')
    warmup = asyncio.run(evaluate(provider, warmup_cases[:1], schema=schema,
                                metadata=metadata, plan=build_call_plan(warmup_cases[:1], schema)))
    sessions = run_dir / 'sessions'
    sessions.mkdir(exist_ok=True)
    DURABLE.write_exclusive(sessions / (now().replace(':', '-').replace('/', '-') + '.json'), warmup)
    grouped = defaultdict(list)
    for call in plan:
        grouped[call['scenario_id']].append(call)
    for index, scenario in enumerate(cases):
        sid = scenario.get('scenario_id', scenario.get('id'))
        completed = scenario_dir / DURABLE.safe_name(index, sid)
        if completed.exists():
            DURABLE.validate_completed(completed, sid, fingerprint)
            print(f'SKIP {index + 1}/{len(cases)} {sid}', flush=True)
            continue
        partial = scenario_dir / ('.' + completed.name + '.partial')
        DURABLE.quarantine_partial(partial, run_dir / 'interrupted')
        progress = {'status': 'partial', 'scenario_id': sid, 'run_fingerprint': fingerprint,
                    'started_at': now(), 'records': []}
        DURABLE.atomic_write(partial, progress)
        def recorded(record):
            progress['records'].append(record)
            DURABLE.atomic_write(partial, progress)
        report = asyncio.run(evaluate(provider, [scenario], schema=schema, metadata=metadata,
                                      on_record=recorded, plan=grouped[sid]))
        DURABLE.atomic_write(partial, report)
        os.replace(partial, completed)
        print(f'DONE {index + 1}/{len(cases)} {sid}', flush=True)
    combined = DURABLE.combine(run_dir, manifest, cases, schema)
    DURABLE.write_exclusive(combined_path, combined)
    return combined


if __name__ == '__main__':
    try:
        report = run(parser().parse_args())
        print(json.dumps({'status': report['status'], 'scenarios': report['scenario_count'],
                          'nonintent_f1': report['metrics']['nonintent']['f1']}), flush=True)
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': safe_error(exc)}), file=sys.stderr)
        raise SystemExit(1)
