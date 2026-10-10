"""Inspect development records without approving human reviews or training."""
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CORPUS = ROOT / 'corpus-v7/v7-20261009-r8-dev'
REVIEW = ROOT / 'corpus-v7/reviews/v7-20261009-r8-dev'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runtime/fpy-v6-pinned/src'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    with gzip.open(CORPUS / name, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]

def dump(name, obj):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

ledger_before = sha(REVIEW / 'decisions.csv')
train = read('splits/train.jsonl.gz')
validation = read('splits/validation.jsonl.gz')
cases = train + validation
sample = read('review-sample.jsonl.gz')
indexed = {c['scenario_id']: c for c in cases}
counts = collections.Counter()
issues = []
filename_rows = []
for c in cases:
    assert c['split'] in ('train', 'validation')
    for t in c['turns']:
        counts['turns'] += 1
        facts = {f['slot_id']: f for f in t['facts']}
        updates = t['gold_extraction']['updates']
        assert len(facts) == len(t['facts']) == len(updates)
        assert {u['slot_id'] for u in updates} == set(facts)
        for u in updates:
            counts['updates'] += 1
            f = facts[u['slot_id']]
            assert isinstance(u['candidate_value'], str)
            assert json.loads(u['candidate_value']) == f['value']
            assert u['status'] == f['status']
            assert u['evidence_text'] and u['evidence_text'] in t['message']
            prior = t['context_state'].get(u['slot_id'])
            if u['status'] == 'confirmed':
                counts['confirmed_updates'] += 1
                assert prior and prior['value'] == f['value']
                assert not t['gold_extraction']['correction_detected']
            if prior and prior['value'] == f['value'] and prior['status'] == f['status'] == 'provided':
                issues.append({'code': 'unchanged_provided', 'scenario_id': c['scenario_id'], 'turn_id': t['turn_id'], 'slot_id': u['slot_id']})
            if u['slot_id'] == 'file_format' and u['status'] == 'inferred':
                counts['inferred_format_updates'] += 1
                source = facts['source_reference']['value']
                suffix = Path(unquote(urlsplit(source).path).replace('\\', '/')).suffix.lower()
                assert {'.csv': 'csv', '.json': 'json', '.parquet': 'parquet', '.xlsx': 'xlsx'}.get(suffix) == f['value']
                assert not {'source_mode', 'target_column', 'time_column', 'contains_sensitive_data', 'credential_ref'} & set(facts)
                filename_rows.append({'scenario_id': c['scenario_id'], 'split': c['split'], 'language': c['language'], 'format': f['value'], 'source': source})
        if t['behavior'] == 'unknown_omission':
            counts['unknown_turns'] += 1
            assert not updates and not t['gold_extraction']['correction_detected']
        if t['turn_id'] == 't1' and c['category'] == 'ambiguous_then_clarified':
            counts['ambiguous_cadence_turns'] += 1
            assert not {'frequency', 'prediction_frequency'} & set(facts)
        if 'target_description' in facts and 'problem_statement' in facts:
            assert facts['target_description']['value'] == facts['problem_statement']['value']

assert not issues, issues[:5]
fmt_table = []
for language in ('en', 'ur', 'mixed'):
    for fmt in ('csv', 'json', 'parquet', 'xlsx'):
        fmt_table.append({'language': language, 'format': fmt,
                          **{split: sum(r['language'] == language and r['format'] == fmt and r['split'] == split for r in filename_rows) for split in ('train', 'validation')}})
train_messages = {t['message'] for c in train for t in c['turns']}
matches = [{'scenario_id': c['scenario_id'], 'turn_id': t['turn_id'], 'behavior': t['behavior']} for c in validation for t in c['turns'] if t['message'] in train_messages]
train_sft = {hashlib.sha256(canonical(e['messages']).encode()).hexdigest() for e in read('sft/train.jsonl.gz')}
sft_collisions = sum(hashlib.sha256(canonical(e['messages']).encode()).hexdigest() in train_sft for e in read('sft/validation.jsonl.gz'))
entity_collision = {c['entity_group'] for c in train} & {c['entity_group'] for c in validation}
assert not sft_collisions and not entity_collision
overlap = []
for p in read('overlap-review-queue.jsonl.gz'):
    a, b = indexed[p['nearest_train']], indexed[p['validation']]
    overlap.append({**p, 'reviewer_type': 'agent', 'human_decision': 'pending',
                    'train_category': a['category'], 'validation_category': b['category'],
                    'different_entity_groups': a['entity_group'] != b['entity_group'],
                    'train_message': a['turns'][0]['message'], 'validation_message': b['turns'][0]['message'],
                    'assessment': 'Shared cadence clause and target form with changed scope/horizon; synthetic development overlap, not independent user prose.'})

spec = importlib.util.spec_from_file_location('r8_materializer', ROOT / 'scripts/materialize-v7-training.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
template = json.loads((REVIEW / 'approval-template.json').read_bytes())
gate_results = []
for case_name, expected in [('pending_review', 'reviews are still pending'), ('pending_ledger', 'Pending/rejected sample'), ('wrong_policy_hash', 'approved annotation policy')]:
    approval = dict(template, decision_ledger_sha256=ledger_before)
    if case_name == 'pending_ledger':
        approval.update(review_complete=True, overlap_review_complete=True,
                        reviewer='UNSIGNABLE TEST FIXTURE', review_scope='Not an actual human review', reviewed_at_utc='2026-10-09T00:00:00Z')
    if case_name == 'wrong_policy_hash':
        approval['annotation_policy_sha256'] = '0' * 64
    with tempfile.TemporaryDirectory(prefix='r8-negative-gate-') as tmp:
        tmp = Path(tmp)
        (tmp / 'fixture.json').write_text(json.dumps(approval), encoding='utf-8')
        (tmp / 'length.json').write_text('{}', encoding='utf-8')
        try:
            gate.materialize(CORPUS, REVIEW, tmp / 'fixture.json', tmp / 'length.json', tmp / 'must-not-exist')
        except ValueError as error:
            assert expected in str(error), str(error)
            assert not (tmp / 'must-not-exist').exists()
            gate_results.append({'check': case_name, 'passed': True, 'reason': str(error), 'inputs_created': False})
        else:
            raise AssertionError('Pending review unexpectedly admitted')

with (REVIEW / 'decisions.csv').open(encoding='utf-8', newline='') as stream:
    human = list(csv.DictReader(stream))
assert len(human) == 524 and all(r['decision'] == 'pending' for r in human)
assert sha(REVIEW / 'decisions.csv') == ledger_before
dump('independent-label-checks.json', {'corpus_manifest_sha256': sha(CORPUS / 'manifest.json'),
     'reviewer_type': 'agent', 'development_scenarios': len(cases), 'sample_scenarios': len(sample),
     'counts': dict(counts), 'targeted_issues': issues,
     'inference_by_language_and_format': fmt_table, 'human_pending_decisions': len(human),
     'human_ledger_sha256': ledger_before, 'human_ledger_unchanged': True,
     'scope': 'Independent structural/status checks over all development turns, supplemented by selected sentence review; not individual human approval.',
     'final_labels_accessed': False, 'training_started': False, 'paid_API_calls': 0})
dump('overlap-detail.json', {'pairs': overlap, 'validation_current_message_duplicates': len(matches),
     'validation_turns': sum(len(c['turns']) for c in validation),
     'duplicate_behaviors': dict(collections.Counter(r['behavior'] for r in matches)),
     'duplicate_current_message_records': matches, 'exact_full_sft_message_list_collisions': sft_collisions,
     'entity_group_collisions': len(entity_collision),
     'limitation': 'Repeated current messages are not identical complete training examples; their prior contexts differ. This is descriptive template overlap, not proof of an invalid split.'})
dump('r8-negative-gate-checks.json', gate_results)

# Freeze a small whole-group pilot plan, without materializing training inputs.
groups = collections.defaultdict(list)
for c in cases:
    groups[(c['split'], c['domain'], c['entity_group'])].append(c)
selected = {}
for split in ('train', 'validation'):
    chosen = []
    for domain in sorted({c['domain'] for c in cases}):
        candidates = [key for key in groups if key[:2] == (split, domain)]
        key = min(candidates, key=lambda k: hashlib.sha256(('v7-r8-pilot-seed42:' + k[2]).encode()).hexdigest())
        assert len(groups[key]) == 10
        chosen.extend(sorted(groups[key], key=lambda c: c['scenario_id']))
    selected[split] = chosen
    assert len(chosen) == 200
    assert collections.Counter(c['language'] for c in chosen) == {'en': 160, 'ur': 20, 'mixed': 20}
assert not {c['entity_group'] for c in selected['train']} & {c['entity_group'] for c in selected['validation']}
dump('pilot-plan.json', {'status': 'plan_only_pending_human_sample_and_overlap_review',
     'corpus_manifest_sha256': sha(CORPUS / 'manifest.json'), 'prompt_version': 'v7-user-approved-extraction-1',
     'selection': 'One complete ten-conversation entity group per domain per split; minimum SHA256(v7-r8-pilot-seed42:group_id).',
     'splits': {split: {'scenario_ids': [c['scenario_id'] for c in rows], 'conversations': len(rows),
                       'extractions': sum(len(c['turns']) for c in rows),
                       'languages': dict(collections.Counter(c['language'] for c in rows)),
                       'families': dict(collections.Counter(c['domain_family'] for c in rows))} for split, rows in selected.items()},
     'proposed_training': {'initialization': 'stock base, new LoRA adapter', 'model': 'Qwen/Qwen3.5-2B',
         'revision': '15852e8c16360a2fea060d615a32b45270f8a8fc', 'epochs': 1, 'learning_rate': 5e-5,
         'max_length': 3584, 'truncation': False, 'save_steps': 10, 'early_stopping_patience': 0,
         'seed': 42, 'batch_size': 1, 'gradient_accumulation_steps': 16,
         'expected_optimizer_steps': 65, 'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': .05},
     'before_execution': ['Complete genuine human sample and overlap decisions bound to this revision.',
         'Decide whether to retain r8 with disclosed filename-language gaps or build a new revision.',
         'Materialize through the unchanged gate, then subset only approved inputs using these IDs.',
         'Freeze a v7 evaluation runner and behavioral acceptance criteria before loading model weights.'],
     'behavioral_evaluation': 'Fresh stock/v6/pilot calls with identical v7 prompt, cases, parser and generation settings; component and rollout; retain failures and scenario-bootstrap intervals.',
     'training_inputs_created': False, 'training_started': False, 'final_labels_accessed': False})
print(json.dumps({'counts': dict(counts), 'gate_checks_passed': len(gate_results), 'duplicate_current_messages': len(matches),
                  'pilot_split_counts': {k: [len(v), sum(len(c['turns']) for c in v)] for k, v in selected.items()}}, indent=2))
