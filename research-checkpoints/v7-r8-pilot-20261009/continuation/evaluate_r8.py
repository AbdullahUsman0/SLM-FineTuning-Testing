"""Fresh extraction-only r8 evaluation. No API client or training is invoked."""
import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

BUNDLE = Path(r'D:\SLM\v7-approved-policy-20261009-r8-review')
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(BUNDLE / 'runtime/fpy-v6-pinned/src'))
from forecasting_assistant.application.state_reducer import apply_extraction
from forecasting_assistant.domain.models import DialogueTurn, ExtractorResult
from forecasting_assistant.domain.schema import create_initial_state, load_schema
from local_slm_lab.comparison_policy import ComparisonPolicy
from local_slm_lab.corpus_v7_reviewed_r8 import state_for_turn, intervene
from local_slm_lab.peft_provider import PeftInferenceConfig
from local_slm_lab.policy_providers import PolicyTransformersProvider, evaluate_with_policy
from local_slm_lab.v5_eval import digest, score_records, state_snapshot
from local_slm_lab.v7_prompts import build_v7_extractor_instructions, build_v7_extractor_input

REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
V6 = Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z\full\best-adapter')
V6_HASHES = {'adapter_model.safetensors': 'e7add7b8b02d24615ed89067c7761c3985a4a6afb7c84ca2a72e3b06fd84e74d',
             'adapter_config.json': '38f60de9678324e91fa882ed86c70d09e4a39b27ae28884374ddc3e6fece15ef'}

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)

@contextmanager
def lock(path):
    import msvcrt
    with Path(path).open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

def build_plans(cases, schema, policy):
    plans = {}
    for case in cases:
        assert case['split'] == 'validation'
        calls = []
        for number, turn in enumerate(case['turns'], 1):
            state = state_for_turn(turn, schema)
            parsed = policy.parse(json.dumps(turn['gold_extraction']), 'extract')
            if not parsed.admitted:
                raise ValueError('r8 gold rejected by the frozen wire policy')
            after = apply_extraction(state, parsed.to_model('extract'), schema, number, turn['message'])
            calls.append({'key': [case['scenario_id'], 'extract', number], 'scenario_id': case['scenario_id'],
                          'cluster_id': case['entity_group'], 'category': case['category'], 'task': 'extract',
                          'expected': parsed.predicted, 'forbidden_slots': turn['forbidden_slots'],
                          'context': {'message': turn['message'], 'state': state_snapshot(state)},
                          'state': state, 'gold_after': state_snapshot(after),
                          'state_intervention': turn['state_intervention'], 'behavior': turn['behavior']})
        plans[case['scenario_id']] = calls
    return plans

def load_inputs(run):
    plan = json.loads((run / 'pilot-plan.json').read_bytes())
    inputs = json.loads((run / 'inputs-manifest.json').read_bytes())
    path = Path(inputs['inputs']['validation']['path'])
    if sha(path) != inputs['inputs']['validation']['sha256']:
        raise ValueError('Pilot validation SFT changed')
    ids = set(plan['splits']['validation']['scenario_ids'])
    corpus = BUNDLE / 'corpus-v7/v7-20261009-r8-dev'
    manifest = json.loads((corpus / 'manifest.json').read_bytes())
    if sha(corpus / 'manifest.json') != plan['corpus_manifest_sha256']:
        raise ValueError('r8 corpus manifest changed')
    source = corpus / 'splits/validation.jsonl.gz'
    if sha(source) != manifest['files']['splits/validation.jsonl.gz']['sha256']:
        raise ValueError('r8 validation trajectories changed')
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        cases = [c for c in map(json.loads, stream) if c['scenario_id'] in ids]
    cases.sort(key=lambda c: c['scenario_id'])
    if len(cases) != 200 or {c['scenario_id'] for c in cases} != ids:
        raise ValueError('Validation cohort inventory changed')
    schema, policy = load_schema(), ComparisonPolicy()
    plans = build_plans(cases, schema, policy)
    examples = { (e['metadata']['scenario_id'], e['metadata']['turn_id']): e
                 for e in map(json.loads, path.read_text(encoding='utf-8').splitlines()) }
    if len(examples) != 1040:
        raise ValueError('Pilot SFT keys differ')
    for c in cases:
        for turn, call in zip(c['turns'], plans[c['scenario_id']]):
            e = examples[(c['scenario_id'], turn['turn_id'])]
            if (e['messages'][0]['content'] != build_v7_extractor_instructions()
                or e['messages'][1]['content'] != build_v7_extractor_input(turn['message'], call['state'], schema)
                or json.loads(e['messages'][2]['content']) != turn['gold_extraction']):
                raise ValueError('Evaluation differs from the actual r8 SFT prompt, prior or gold')
    return schema, policy, cases, plans

def protocol(run):
    schema, policy, cases, plans = load_inputs(run)
    packed = [[call['key'], digest(call['context']), digest(call['expected']), digest(call['gold_after']),
               call['state_intervention']] for case in cases for call in plans[case['scenario_id']]]
    code = Path(__file__).resolve().parent
    value = {'runner_sha256': sha(__file__), 'support_code_sha256': {p.name: sha(p) for p in sorted(code.glob('*.py'))},
             'upstream_python_sources': {str(p.relative_to(BUNDLE)): sha(p) for p in sorted((BUNDLE / 'local_slm_lab').glob('*.py'))},
             'source_bundle_manifest_sha256': sha(BUNDLE / 'v7-approved-policy-bundle-manifest.json'),
             'corpus_manifest_sha256': sha(BUNDLE / 'corpus-v7/v7-20261009-r8-dev/manifest.json'),
             'v7_annotation_policy_sha256': sha(BUNDLE / 'corpus-v7/v7-20261009-r8-dev/annotation-policy.json'),
             'v7_prompt_sha256': digest(build_v7_extractor_instructions()), 'wire_policy_sha256': policy.fingerprint,
             'pretraining_behavioral_protocol_sha256': sha(run / 'behavioral-protocol.json'),
             'cases_sha256': digest(cases), 'context_gold_sha256': digest(packed),
             'base_model': 'Qwen/Qwen3.5-2B', 'base_revision': REVISION,
             'generation': {'dtype': 'bfloat16', 'device': 'cuda', 'max_new_tokens': 1024, 'greedy': True,
                            'enable_thinking': False, 'seed': 42, 'repetition_penalty': 1.0, 'batch_size': 1},
             'scenarios': len(cases), 'calls_per_track': len(packed), 'source_groups': len({c['entity_group'] for c in cases}),
             'component': 'Exact r8 gold prior state and user history, verified against actual validation SFT.',
             'rollout': 'Own accepted prior; original r8 synthetic context mutations applied at their declared turns. Report seeded-repair category separately and also report rollout excluding it.',
             'timing': 'One unscored warmup per arm; synchronized GPU extraction; serial arm runs.',
             'bootstrap': '10000 paired resamples of the 20 shared entity groups; seed42. Not 200 independent conversations.',
             'paid_API_calls': 0, 'sealed_final_accessed': False}
    return schema, policy, cases, plans, value

def verify_freeze(run):
    result = protocol(run)
    freeze = json.loads((run / 'evaluation-freeze.json').read_bytes())
    if freeze['protocol'] != result[-1] or freeze['fingerprint'] != digest(result[-1]):
        raise ValueError('Frozen evaluation code, sources, gold or context changed')
    return result

def advance(state, record, policy, schema):
    if record['prediction_valid']:
        parsed = policy.parse(record['raw_output'], 'extract')
        if not parsed.admitted:
            raise ValueError('Recorded admission cannot be reproduced')
        try:
            state = apply_extraction(state, parsed.to_model('extract'), schema, record['key'][2], record['context']['message'])
        except Exception:
            pass
    state.turns.append(DialogueTurn(turn_number=len(state.turns)+1, user_message=record['context']['message']))
    return state

class Provider(PolicyTransformersProvider):
    async def extract(self, message, state):
        self._torch.cuda.synchronize()
        try:
            return self._structured('extract', build_v7_extractor_instructions(),
                                    build_v7_extractor_input(message, state, self.schema), ExtractorResult)
        finally:
            self._torch.cuda.synchronize()

    def _generate_raw(self, *args, **kwargs):
        try:
            return super()._generate_raw(*args, **kwargs)
        except Exception as error:
            self.fatal_generation_error = type(error).__name__
            raise

async def one_call(provider, case, call, track, state, schema, policy):
    call = dict(call)
    if track == 'rollout':
        if call['state_intervention']:
            intervene(state, call['state_intervention'])
        call['state'] = state.model_copy(deep=True)
        call['context'] = {'message': call['context']['message'], 'state': state_snapshot(state)}
    report = await evaluate_with_policy(provider, [case], schema=schema, plan=[call])
    record = report['records'][0]
    record.update(track=track, domain=case['domain'], language=case['language'], behavior=call['behavior'],
                  state_intervention=call['state_intervention'], source_group=case['entity_group'])
    updates = (record.get('predicted') or {}).get('updates', [])
    record['evidence_grounded'] = record['prediction_valid'] and all(
        u['evidence_text'].strip() and u['evidence_text'] in call['context']['message'] for u in updates)
    if getattr(provider, 'fatal_generation_error', None):
        raise RuntimeError('Generation failed; stop rather than silently score an infrastructure failure')
    if track == 'rollout':
        state = advance(state, record, policy, schema)
    return record, state

async def run_arm(run, arm):
    schema, policy, cases, plans, frozen = verify_freeze(run)
    completion = json.loads((run / 'completion.json').read_bytes())
    if completion['status'] != 'complete' or completion['optimizer_steps'] != 65:
        raise ValueError('Pilot has not completed and verified')
    adapter = None if arm == 'stock' else V6 if arm == 'v6' else Path(completion['adapter'])
    hashes = None if adapter is None else {name: sha(adapter / name) for name in V6_HASHES}
    if hashes is not None and hashes != (V6_HASHES if arm == 'v6' else completion['adapter_sha256']):
        raise ValueError('Adapter checksum mismatch')
    identity = {'arm': arm, 'protocol_fingerprint': digest(frozen), 'adapter_sha256': hashes}
    fingerprint = digest(identity)
    out = run / 'evaluation' / arm
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / 'run.lock'):
        if (out / 'run-manifest.json').exists():
            if json.loads((out / 'run-manifest.json').read_bytes())['fingerprint'] != fingerprint:
                raise ValueError('Resume would change evaluation identity')
        else:
            atomic(out / 'run-manifest.json', {'identity': identity, 'fingerprint': fingerprint, 'started_at': now()})
        if (out / 'completion.json').exists():
            return
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Matched benchmark requires CUDA BF16')
        torch.set_num_threads(4)
        provider = Provider(PeftInferenceConfig(base_model='Qwen/Qwen3.5-2B', revision=REVISION,
                   variant='base' if adapter is None else 'custom', adapter_path=adapter,
                   device='cuda', dtype='bfloat16', max_new_tokens=1024), schema, policy=policy)
        await one_call(provider, cases[0], plans[cases[0]['scenario_id']][0], 'component', None, schema, policy)
        atomic(out / 'runtime.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(), 'warmup_calls': 1, 'chat_template_sha256': provider.chat_template_sha256})
        jobs = out / 'jobs'; jobs.mkdir(exist_ok=True)
        for track in ('component', 'rollout'):
            for index, case in enumerate(cases):
                path = jobs / f'{track}-{index:03d}.json'
                if path.exists():
                    job = json.loads(path.read_bytes())
                    if job['fingerprint'] != fingerprint or job['records_sha256'] != digest(job['records']) or job['scenario_id'] != case['scenario_id']:
                        raise ValueError('Completed evaluation job changed')
                    continue
                state, records = create_initial_state(schema), []
                for call in plans[case['scenario_id']]:
                    record, state = await one_call(provider, case, call, track, state, schema, policy)
                    record['arm'] = arm
                    records.append(record)
                    atomic(path.with_suffix('.partial.json'), {'fingerprint': fingerprint, 'records': records})
                atomic(path, {'status': 'complete', 'fingerprint': fingerprint, 'scenario_id': case['scenario_id'],
                             'track': track, 'arm': arm, 'records': records, 'records_sha256': digest(records)})
                path.with_suffix('.partial.json').unlink(missing_ok=True)
                atomic(out / 'status.json', {'status': 'running', 'arm': arm, 'track': track,
                       'completed_jobs': len(list(jobs.glob('[cr]*-[0-9][0-9][0-9].json'))), 'last_scenario': index+1, 'updated_at': now()})
                print(json.dumps({'arm': arm, 'track': track, 'scenario': index+1, 'of': 200}), flush=True)
            records = [r for p in sorted(jobs.glob(f'{track}-[0-9][0-9][0-9].json')) for r in json.loads(p.read_bytes())['records']]
            assert len(records) == 1040
            atomic(out / f'{track}-report.json', {'status': 'complete', 'fingerprint': fingerprint,
                   'protocol_fingerprint': digest(frozen), 'arm': arm, 'track': track, 'records': records,
                   'metrics': score_records(records, [s.slot_id for s in schema.slots]), 'api_cost_usd': 0})
        if adapter is not None and {name: sha(adapter / name) for name in hashes} != hashes:
            raise ValueError('Adapter changed during inference')
        atomic(out / 'completion.json', {'status': 'complete', 'fingerprint': fingerprint, 'arm': arm,
               'component_calls': 1040, 'rollout_calls': 1040, 'finished_at': now(), 'paid_API_calls': 0, 'sealed_final_accessed': False})

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--arm', choices=('stock', 'v6', 'v7'), required=True)
    args = p.parse_args()
    asyncio.run(run_arm(args.run.resolve(), args.arm))
