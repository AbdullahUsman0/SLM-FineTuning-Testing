"""Materialize the explicitly authorized pilot, retaining pending human labels."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

BUNDLE = Path(r'D:\SLM\v7-approved-policy-20261009-r8-review')
REPO = Path(r'D:\SLM\SLM-FineTuning-Testing')
CORPUS = BUNDLE / 'corpus-v7/v7-20261009-r8-dev'
PLAN = BUNDLE / 'remote-review-20261009/pilot-plan.json'
RECEIPT = BUNDLE / 'remote-review-20261009/local-tokenizer-preflight.json'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

payloads = json.loads((REPO / 'research-checkpoints/v7-approved-policy-20261009/bundle-manifest.json').read_bytes())
for name, rec in payloads['files'].items():
    path = BUNDLE / name
    assert path.stat().st_size == rec['bytes'] and sha(path) == rec['sha256'], name
plan = json.loads(PLAN.read_bytes())
receipt = json.loads(RECEIPT.read_bytes())
manifest_hash = sha(CORPUS / 'manifest.json')
assert manifest_hash == plan['corpus_manifest_sha256'] == receipt['corpus_manifest_sha256']
assert receipt['status'] == 'pass' and receipt['truncation'] is False
assert receipt['max_length'] == 3584
assert receipt['trainer_token_length_helper_sha256'] == sha(BUNDLE / 'training/train_lora.py')
base = Path(r'D:\SLM\hf-cache\hub\models--Qwen--Qwen3.5-2B\snapshots\15852e8c16360a2fea060d615a32b45270f8a8fc\model.safetensors-00001-of-00001.safetensors')
base_hash = sha(base)
assert base_hash == 'aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1'
run = Path(r'D:\SLM\FYP-model-runs') / ('qwen35-2b-lora-v7-r8-pilot-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
run.mkdir()
(run / 'inputs').mkdir()
authorization = {
    'record_type': 'explicit_user_training_authorization',
    'user_message': 'approved, start training',
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'The proposed one-epoch, 200-conversation v7 r8 pilot; retain r8 with its disclosed development limitations.',
    'corpus_manifest_sha256': manifest_hash, 'pilot_plan_sha256': sha(PLAN),
    'policy_sha256': sha(CORPUS / 'annotation-policy.json'),
    'rubric_sha256': sha(CORPUS / 'annotation-rubric.md'),
    'preflight_sha256': sha(RECEIPT),
    'human_review_ledger_sha256': sha(BUNDLE / 'corpus-v7/reviews/v7-20261009-r8-dev/decisions.csv'),
    'human_sample_review_complete': False, 'human_overlap_review_complete': False,
    'authorization_precedence': 'The user explicitly authorized starting after being told the original materializer requires pending human review. This pilot uses a separate documented authorization route; the original gate and human ledger remain unchanged.',
    'human_approvals_fabricated': False, 'full_training_authorized_by_this_record': False,
    'final_labels_accessed': False, 'paid_APIs_authorized': False,
}
write(run / 'user-training-authorization.json', authorization)
write(run / 'pilot-plan.json', plan)
write(run / 'tokenizer-preflight.json', receipt)
inputs = {}
for split in ('train', 'validation'):
    ids = set(plan['splits'][split]['scenario_ids'])
    seen = Counter()
    dest = run / 'inputs' / (split + '.jsonl')
    with gzip.open(CORPUS / 'sft' / (split + '.jsonl.gz'), 'rb') as source, dest.open('xb') as target:
        for line in source:
            row = json.loads(line)
            sid = row['metadata']['scenario_id']
            if sid in ids:
                seen[sid] += 1
                target.write(line)
    assert set(seen) == ids and sum(seen.values()) == plan['splits'][split]['extractions'] == 1040
    inputs[split] = {'path': str(dest), 'sha256': sha(dest), 'records': sum(seen.values()),
                     'conversations': len(seen), 'source_gzip_sha256': sha(CORPUS / 'sft' / (split + '.jsonl.gz')),
                     'rows_are_byte_identical_source_lines': True}
write(run / 'inputs-manifest.json', {'authorization_sha256': sha(run / 'user-training-authorization.json'),
      'corpus_manifest_sha256': manifest_hash, 'pilot_plan_sha256': sha(PLAN), 'inputs': inputs,
      'selection': plan['selection'], 'human_review_complete': False, 'final_labels_accessed': False})
protocol = {
    'record_type': 'pretraining_behavioral_evaluation_plan',
    'corpus_manifest_sha256': manifest_hash, 'prompt_version': plan['prompt_version'],
    'cohort': plan['splits']['validation']['scenario_ids'],
    'scope': 'Development pilot validation; synthetic templates, not an unseen-user final test.',
    'arms': ['stock_2B', 'v6', 'v7_r8_pilot_final_step'],
    'tracks': ['component_gold_prior', 'rollout_own_accepted_prior'],
    'decoding': {'seed': 42, 'do_sample': False, 'enable_thinking': False, 'max_new_tokens': 1024, 'batch_size': 1},
    'wire_policy': 'closed-schema-pydantic-once-20261006-1',
    'metrics': ['exact_slot_value_status_F1', 'JSON_validity', 'parser_admission', 'unmentioned_updates',
                'correction_values_and_full_state', 'unknown_no_updates_and_retained_state', 'latency',
                'per_domain_language_and_slot_diagnostics'],
    'failed_calls': 'Retain in denominators.', 'intervals': 'Paired conversation-cluster bootstrap, 10000 samples, seed 42.',
    'technical_completion': '65 optimizer steps, finite losses, verified resumable checkpoints and final-step adapter hashes.',
    'advance_criteria': 'No automatic full training. Require matched fresh behavior reports, human failure review, and explicit next-round decision. Loss is never sufficient.',
    'model_winner_claim': 'Only if the predeclared primary rollout F1 improvement over v6 has a 95% interval above zero, without worse parser admission or more unmentioned updates; otherwise exploratory/inconclusive.',
    'evaluator_implementation': 'Freeze and hash an r8-specific extractor evaluator against this protocol before the first scored calls; do not repoint the historical 31-case evaluator.',
    'training_inputs_created': True, 'final_labels_accessed': False, 'paid_API_calls': 0,
}
write(run / 'behavioral-protocol.json', protocol)
config = {'python': str(REPO / '.venv/Scripts/python.exe'), 'bundle': str(BUNDLE),
          'trainer': str(BUNDLE / 'training/train_lora.py'), 'trainer_sha256': sha(BUNDLE / 'training/train_lora.py'),
          'run': str(run), 'output': str(run / 'pilot'), 'inputs': inputs,
          'settings': plan['proposed_training'], 'base_weights_sha256': base_hash,
          'env': {'HF_HOME': r'D:\SLM\hf-cache', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                  'PYTHONPATH': str(BUNDLE / '.review-deps'), 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                  'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}}
write(run / 'launch-config.json', config)
(Path(r'D:\SLM') / 'active-v7-pilot-run.txt').write_text(str(run) + '\n', encoding='utf-8')
print(json.dumps({'run': str(run), 'inputs': inputs, 'optimizer_steps': 65, 'human_ledger_unchanged': True}, indent=2))
