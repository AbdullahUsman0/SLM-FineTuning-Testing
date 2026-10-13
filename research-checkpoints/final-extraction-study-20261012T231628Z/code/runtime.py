"""Pinned dependencies and durable serialization for the fresh extraction study."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

CHECKPOINT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get('STUDY_R8_BUNDLE', r'D:\SLM\v7-approved-policy-20261009-r8-review')).resolve()
FPY = BUNDLE / 'runtime/fpy-v6-pinned/src'
for path in (BUNDLE, BUNDLE / '.review-deps', FPY):
    sys.path.insert(0, str(path))
from forecasting_assistant.domain.models import DialogueTurn
from forecasting_assistant.domain.schema import create_initial_state, load_schema
from forecasting_assistant.application.state_reducer import apply_extraction
from local_slm_lab.comparison_policy import ComparisonPolicy, strict_json
from local_slm_lab.v5_eval import state_snapshot, confusion, score_records
from local_slm_lab.v7_prompts import build_v7_extractor_instructions, build_v7_extractor_input

BASE_REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
FPY_REVISION = '04d52c015d1e3ecdefe92b87116f209361509b4b'


def now():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode('utf-8')).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return strict_json(Path(path).read_text(encoding='utf-8-sig'))


def require_study_execution():
    decision = CHECKPOINT / 'EXECUTION_DECISION.json'
    if not decision.exists() or read(decision).get('status') != 'review_and_budget_gates_enabled':
        raise ValueError('User deferred the new matched study; use existing results until explicit scope and budget authorization')


def atomic(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path.name)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if exclusive and path.exists():
        temporary.unlink()
        raise FileExistsError(path.name)
    os.replace(temporary, path)


@contextmanager
def lock(path):
    import msvcrt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush(); os.fsync(stream.fileno())
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def semantic_state(state):
    # Exclude confidence/evidence/timestamps from semantic-state accuracy; preserve them in raw snapshots.
    snapshot = state_snapshot(state)
    return {'intent': snapshot['intent'], 'slots': snapshot['slots']}


def append_user(state, message):
    state.turns.append(DialogueTurn(turn_number=len(state.turns) + 1, user_message=message))
    return state


def instructions():
    return build_v7_extractor_instructions() + (
        '\nFresh-study contract: evidence must match current_message exactly, including case and whitespace. '
        'For an unknown-information follow-up retain established intent; if prior intent is ambiguous '
        'and no forecasting goal is stated, keep intent ambiguous. No default values or new non-intent '
        'updates follow from unknown information. An explicitly JSON-encoded string stays a string '
        'even when its contents look like a number, boolean, null, list or object. '
        'The target phrase includes stated metric modifiers; excludes request verbs, source instructions '
        'and forecast-horizon clauses. Preserve its original text.'
    )


def context(message, state, schema):
    return build_v7_extractor_input(message, state, schema)


def verify_sources(provider=None):
    manifest = read(CHECKPOINT / 'runtime-lock.json')
    for name, expected in manifest['bundle_files_sha256'].items():
        if sha(BUNDLE / name) != expected:
            raise ValueError('Pinned research dependency changed: ' + name)
    for name, expected in manifest['parser_dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Parser dependency changed: ' + name)
    if (CHECKPOINT / 'environment.json').exists():
        required={'local':{'torch','transformers','peft','accelerate','safetensors'},
                  'openai':{'openai'},'verify':{'numpy'}}.get(provider,set())
        for name, expected in read(CHECKPOINT / 'environment.json')['packages'].items():
            if name not in required:
                continue
            if importlib.metadata.version(name) != expected:
                raise ValueError('Recorded inference dependency changed: ' + name)
    ComparisonPolicy()
    if not Path(sys.modules['forecasting_assistant.domain.models'].__file__).resolve().is_relative_to(FPY):
        raise ValueError('Wrong fpy package imported')
    if len(load_schema().slots) != 79:
        raise ValueError('Full 79-slot schema required')
    return manifest


def verify_models(config, *, check_base=False):
    manifest = read(CHECKPOINT / 'models-freeze.json')
    for arm in ('v6', 'v7'):
        path = Path(config['adapters'][arm])
        for name, expected in manifest['adapters'][arm]['sha256'].items():
            if sha(path / name) != expected:
                raise ValueError('Frozen adapter changed: ' + arm + '/' + name)
    if check_base:
        cached = Path(config['HF_HOME']) / 'hub/models--Qwen--Qwen3.5-2B/snapshots' / BASE_REVISION / 'model.safetensors-00001-of-00001.safetensors'
        if cached.resolve() != Path(config['base_weights']).resolve() or sha(cached) != manifest['base_weights_sha256']:
            raise ValueError('Cached base weights do not match the frozen identity')
    return manifest
