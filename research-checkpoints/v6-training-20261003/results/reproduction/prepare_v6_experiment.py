"""Materialize user-authorized experimental v6 inputs without claiming human review."""
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(r'D:\SLM\SLM-FineTuning-Testing')
RUN = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO))
from local_slm_lab.corpus_v6 import verify_artifacts

corpus = REPO / 'corpus-v6/v6-20261002-r4'
verification = verify_artifacts(corpus)
manifest = json.loads((corpus / 'manifest.json').read_bytes())
manifest_hash = hashlib.sha256((corpus / 'manifest.json').read_bytes()).hexdigest()
assert manifest_hash == '0782956c587b5996a2df3f7b799fb0b9e572942b063c8fc1b623ea691fc5aeb9'
RUN.mkdir(parents=True, exist_ok=False)
inputs = RUN / 'inputs'
inputs.mkdir()
records = {}
for source, name in (('sft/train.jsonl.gz', 'train.jsonl'),
                     ('sft/validation.jsonl.gz', 'validation.jsonl'),
                     ('splits/validation.jsonl.gz', 'validation-cases.jsonl'),
                     ('splits/smoke.jsonl.gz', 'smoke-cases.jsonl')):
    destination = inputs / name
    h = hashlib.sha256()
    with gzip.open(corpus / source, 'rb') as handle, destination.open('xb') as out:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
            out.write(block)
        out.flush()
        os.fsync(out.fileno())
    assert h.hexdigest() == manifest['files'][source]['uncompressed_sha256']
    records[name] = {'sha256': h.hexdigest(), 'bytes': destination.stat().st_size}
authorization = {
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'source': 'Explicit user request in this Codex conversation on 2026-10-03 (America/Los_Angeles)',
    'request': 'analyze the results, then pull the latest and fine tune the model on the newly ggenrated dataset v6 that is domain specific in the background',
    'scope': 'Experimental fine-tuning on the frozen v6 development data, in the background',
    'corpus_manifest_sha256': manifest_hash,
    'independent_human_review_complete': False, 'overlap_review_complete': False,
    'unresolved_overlap_pairs': 10219,
    'final_labels_accessed': False,
    'note': 'Direct user training authorization takes precedence over the repository review-first workflow. This is not a review approval or evidence of resolved leakage.'}
(RUN / 'authorization.json').write_text(json.dumps(authorization, indent=2) + '\n', encoding='utf-8')
lock = {'corpus_version': manifest['corpus_version'], 'corpus_manifest_sha256': manifest_hash,
        'fpy_commit': manifest['fpy_commit'], 'files': records, 'verification': verification,
        'authorization': authorization, 'status': 'prepared_experimental_human_review_pending',
        'final_labels_accessed': False}
(inputs / 'input-lock.json').write_text(json.dumps(lock, indent=2) + '\n', encoding='utf-8')
print('Verified and materialized development inputs', flush=True)
os.environ.update(HF_HOME=r'D:\SLM\hf-cache', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B', revision='15852e8c16360a2fea060d615a32b45270f8a8fc', local_files_only=True)
spec = importlib.util.spec_from_file_location('trainer_helpers', REPO / 'training/train_lora.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
report = {}
for split in ('train', 'validation'):
    lengths = []
    with (inputs / (split + '.jsonl')).open(encoding='utf-8') as handle:
        for index, line in enumerate(handle, 1):
            row = trainer.to_prompt_completion(json.loads(line))
            lengths.append(trainer.token_length(tokenizer, row['prompt'] + row['completion'], row['chat_template_kwargs']))
            if index % 500 == 0:
                print(f'{split} token lengths {index}', flush=True)
    report[split] = {'examples': len(lengths), 'minimum': min(lengths), 'maximum': max(lengths),
                     'p95': sorted(lengths)[int(.95 * (len(lengths) - 1))]}
assert report['train']['examples'] == 5800 and report['validation']['examples'] == 1253
maximum = max(x['maximum'] for x in report.values())
limit = max(3584, ((maximum + 255) // 256) * 256)
report.update(max_length=limit, truncation_allowed=False, enable_thinking=False,
              model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc')
(RUN / 'token-lengths.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
