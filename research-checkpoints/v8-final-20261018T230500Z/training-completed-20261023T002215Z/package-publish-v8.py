"""Prepare and publish the verified experimental adapter; never put weights in Git."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
REPO = Path('D:/SLM/SLM-FineTuning-Testing')
RUN = Path('D:/SLM/FYP-model-runs/qwen35-2b-lora-v8-final-20261018T230500Z')
RELATIVE = 'research-checkpoints/v8-final-20261018T230500Z/training-completed-20261023T002215Z'
RESULT = REPO / RELATIVE
TAG = 'v8-adapter-20261022'
BASE = 'https://github.com/AbdullahUsman0/SLM-FineTuning-Testing'
API = 'https://api.github.com/repos/AbdullahUsman0/SLM-FineTuning-Testing'
POINTER = CONTROL / 'v8-release-preparation.json'
EXPECTED = {
    'adapter_model.safetensors': 'c83e0f1a458125970eb70ff360ebead02618f2f48b5bbb59e2ca91a5b7c4c0df',
    'adapter_config.json': 'a1ce14418c5c22375ccfd773338ef1b09117ec9b9e72db034c71ade886b4664c',
}

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')

def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()

def manifest():
    files = {p.relative_to(RESULT).as_posix(): sha(p) for p in sorted(RESULT.rglob('*'))
             if p.is_file() and p.name != 'artifact-manifest.json'}
    token = re.compile(rb'gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}')
    for name, expected in files.items():
        data = (RESULT / name).read_bytes()
        if name.endswith('.gz'):
            data = gzip.decompress(data)
        if token.search(data):
            raise RuntimeError('Credential-shaped literal in publication: ' + name)
    write(RESULT / 'artifact-manifest.json', {'at_UTC': now(), 'files_sha256': files,
          'weights_optimizer_states_datasets_caches_in_normal_Git': False})

def prepare():
    if POINTER.exists() or RESULT.exists():
        raise RuntimeError('Existing preparation preserved; inspect it instead of overwriting')
    sys.path.insert(0, str(REPO / 'training/final_round'))
    import run as worker
    worker.check_config(RUN)
    trainer = worker.helper()
    training = read(RUN / 'training-completion.json')
    run_manifest = trainer.load_manifest(RUN / 'full')
    final = trainer.verify_checkpoint(RUN / 'full/checkpoint-2234', run_manifest['run_fingerprint'])
    expected_steps = list(range(10, 2231, 10)) + [2234]
    if training['status'] != 'complete' or training['optimizer_steps'] != 2234 or training['checkpoint_steps'] != expected_steps:
        raise RuntimeError('Training completion/schedule mismatch')
    if training['run_fingerprint'] != run_manifest['run_fingerprint']:
        raise RuntimeError('Run fingerprint mismatch')
    adapter = RUN / 'full/best-adapter'
    for name, expected in EXPECTED.items():
        actual = sha(adapter / name)
        if actual != expected or actual != training['adapter_files_sha256'][name] or actual != sha(RUN / 'full/checkpoint-2234' / name):
            raise RuntimeError('STOP: adapter/receipt/final-checkpoint hash mismatch: ' + name)
    for name, expected in training['adapter_files_sha256'].items():
        if sha(adapter / name) != expected:
            raise RuntimeError('Completion member mismatch: ' + name)
    selection = read(adapter / 'selection.json')
    if selection['task_winner_selected'] is not False:
        raise RuntimeError('Experimental final-step selection expected')
    RESULT.mkdir(parents=True, exist_ok=False)
    sources = {
        'training-completion.json': RUN / 'training-completion.json',
        'training-evaluation.json': RUN / 'full/evaluation.json',
        'training-run-manifest.json': RUN / 'full/run-manifest.json',
        'adapter-selection.json': adapter / 'selection.json',
        'launch-config.json': RUN / 'launch-config.json',
        'launch-config.sha256.json': RUN / 'launch-config.sha256.json',
        'preflight.json': RUN / 'preflight.json',
        'inventory.json': RUN / 'inventory.json',
    }
    for name, path in sources.items():
        shutil.copyfile(path, RESULT / name)
    shutil.copyfile(__file__, RESULT / Path(__file__).name)
    (RESULT / 'ENVIRONMENT_FREEZE.txt').write_text(run_manifest['runtime']['pip_freeze'].rstrip() + '\n', encoding='utf-8')
    inference = {'python': run_manifest['runtime']['python'], 'CUDA_runtime': run_manifest['runtime']['cuda'],
        'GPU': run_manifest['runtime']['devices'][0]['name'],
        'packages': run_manifest['immutable']['packages'] | {'tokenizers': '0.22.2', 'pydantic': '2.13.5', 'jsonschema': '4.26.0'},
        'base_model': 'Qwen/Qwen3.5-2B', 'base_revision': '15852e8c16360a2fea060d615a32b45270f8a8fc',
        'architecture': 'Qwen3_5ForConditionalGeneration',
        'fpy_revision': '04d52c015d1e3ecdefe92b87116f209361509b4b'}
    write(RESULT / 'inference-dependencies.json', inference)
    readme = '''# Experimental final-step v8 adapter

This is the fresh stock-base LoRA trained on corpus-v8/v8-final-20261018-r1 for two epochs, seed 42, 2,234 optimizer steps. Base: Qwen/Qwen3.5-2B, revision 15852e8c16360a2fea060d615a32b45270f8a8fc. Load this adapter onto that stock base; never stack it onto another adapter. The architecture is Qwen3_5ForConditionalGeneration. adapter_config.json has no pinned revision value, so the base revision must be supplied explicitly by the loader.

The folder name best-adapter refers to the fixed FINAL STEP. Behavioral selection, development comparison and independent human review are pending at this packaging snapshot. This is not a final-test or OpenAI comparison, and no model winner has been established. The final teacher-forced development-validation loss is 0.0033881000708788633; it is not extraction F1.

See inference-dependencies.json and training-run-manifest.json for exact versions. The validated stack used Python 3.12.10, CUDA torch 2.11.0+cu128, transformers 5.15.0, peft 0.20.0, accelerate 1.14.0, safetensors 0.8.0 and tokenizers 0.22.2. The extraction pipeline also uses pydantic 2.13.5 and jsonschema 4.26.0. Preserve a working CUDA installation; do not replace it with a CPU wheel. The base weights are downloaded/cached separately and are excluded here.

## Reproduce the frozen local development diagnostic

From the repository branch experiment/v5-grounded-20260919, use the dedicated clean fpy dependency at commit 04d52c015d1e3ecdefe92b87116f209361509b4b. Substitute local paths and an existing compatible CUDA Python. Use a NEW output directory; do not touch the running remote evaluation:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
& <existing-CUDA-python> -B training/final_round/evaluate_development.py `
  --fpy runtime/fpy-final-pinned `
  --adapter <extracted-bundle>/best-adapter `
  --output <new-local-development-output>
```

The script pins the base revision, uses the v8 corpus instructions/response schema, disables thinking and sampling, and requires BF16 CUDA. It selects the same 100 development conversations for component and rollout tracks. It does not call a paid API or read final-test labels. A CPU/quantized manual run is a separate testing configuration and cannot replace these CUDA results. The older chat-peft.py v5/v6 prompts are not automatically the frozen v8 evaluation prompt.

The training receipt records all 224 complete resumable checkpoints. This transfer archive contains adapter files and metadata only, not optimizer/RNG checkpoints. It excludes base weights, datasets, caches and credentials. Keep the ZIP checksum and ARCHIVE_MANIFEST.json; verify member hashes before loading.
'''
    (RESULT / 'ADAPTER_README.md').write_text(readme, encoding='utf-8')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    downloads = Path.home() / 'Downloads'
    downloads.mkdir(exist_ok=True)
    archive = downloads / (RUN.name + '-experimental-final-step-' + stamp + '.zip')
    members = {
        'best-adapter/adapter_model.safetensors': adapter / 'adapter_model.safetensors',
        'best-adapter/adapter_config.json': adapter / 'adapter_config.json',
        'best-adapter/selection.json': adapter / 'selection.json',
        'training-completion.json': RESULT / 'training-completion.json',
        'training-evaluation.json': RESULT / 'training-evaluation.json',
        'training-run-manifest.json': RESULT / 'training-run-manifest.json',
        'inference-dependencies.json': RESULT / 'inference-dependencies.json',
        'README.md': RESULT / 'ADAPTER_README.md',
    }
    hashes = {name: sha(path) for name, path in members.items()}
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for name, path in members.items():
            package.write(path, name)
        package.writestr('ARCHIVE_MANIFEST.json', json.dumps({'files_sha256': hashes}, indent=2) + '\n')
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or set(package.namelist()) != set(hashes) | {'ARCHIVE_MANIFEST.json'}:
            raise RuntimeError('ZIP inventory or CRC mismatch')
        for name, expected in hashes.items():
            if hashlib.sha256(package.read(name)).hexdigest() != expected:
                raise RuntimeError('Archived member mismatch: ' + name)
    checksum = Path(str(archive) + '.sha256')
    archive_sha = sha(archive)
    checksum.write_text(archive_sha + '  ' + archive.name + '\n', encoding='ascii')
    receipt = {'scope': 'Verified experimental final-step adapter; behavioral comparison pending',
        'prepared_at_UTC': now(), 'tag': TAG, 'release_url': BASE + '/releases/tag/' + TAG,
        'zip_name': archive.name, 'zip_sha256': archive_sha, 'zip_bytes': archive.stat().st_size,
        'zip_download_url': BASE + '/releases/download/' + TAG + '/' + archive.name,
        'checksum_name': checksum.name, 'checksum_sha256': sha(checksum),
        'checksum_download_url': BASE + '/releases/download/' + TAG + '/' + checksum.name,
        'members_sha256': hashes, 'adapter_files_sha256': training['adapter_files_sha256'],
        'publication_status': 'prepared_not_yet_uploaded', 'weights_in_normal_Git': False,
        'development_comparison_status_at_preparation': 'running; not a complete behavioral result',
        'model_winner_established': False, 'independent_human_review_complete': False,
        'paid_API_calls': 0, 'final_test_labels_accessed': False}
    write(RESULT / 'ADAPTER_RELEASE.json', receipt)
    write(RESULT / 'checkpoint-2234-independent-verification.json', final)
    evaluation = read(RUN / 'full/evaluation.json')
    write(RESULT / 'RESEARCH_STATUS.json', {
        'at_UTC': now(), 'training_status': 'complete_and_all_224_checkpoints_verified_by_supplied_worker',
        'optimizer_steps': 2234, 'epochs': 2, 'seed': 42,
        'dataset': 'corpus-v8/v8-final-20261018-r1',
        'corpus_manifest_sha256': training['corpus_manifest_sha256'],
        'training_source_commit': 'bd36ff605d1d7e6c5f79de6e65d60785e0afe49d',
        'fpy_revision': inference['fpy_revision'], 'final_evaluation': evaluation,
        'development_evaluation': {'status': 'running', 'version': 'v8-development-diagnostic-1',
            'freeze_file_in_running_evaluation': str(RUN / 'development-evaluation/freeze.json'),
            'completed_results_expected_under': 'research-checkpoints/v8-final-20261018T230500Z',
            'v8_summary': 'v8/summary.json', 'v6_summary': 'v6/summary.json',
            'comparison': 'comparison.json', 'summary_table': 'summary.csv'},
        'final_test_accessed': False, 'independent_human_review_complete': False,
        'new_paid_API_calls': 0, 'model_winner_established': False})
    write(POINTER, {'zip': str(archive), 'checksum': str(checksum), 'result': str(RESULT),
        'relative': RELATIVE, 'tag': TAG, 'prepared_at_UTC': now()})
    manifest()
    print(json.dumps({'result': str(RESULT), 'zip': str(archive), 'sha256': archive_sha,
                      'verified_archive_members': len(hashes) + 1}))

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        result = super().redirect_request(request, fp, code, msg, headers, newurl)
        if result and urllib.parse.urlparse(request.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            result.remove_header('Authorization')
        return result

def publish():
    prepared = read(POINTER)
    receipt = read(RESULT / 'ADAPTER_RELEASE.json')
    archive = Path(prepared['zip'])
    checksum = Path(prepared['checksum'])
    if sha(archive) != receipt['zip_sha256'] or sha(checksum) != receipt['checksum_sha256']:
        raise RuntimeError('STOP: package/checksum changed')
    if git('status', '--porcelain', '--', RELATIVE):
        raise RuntimeError('Commit and push the reviewable metadata before publishing')
    commit = git('rev-parse', 'HEAD')
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'Never'}
    credential = subprocess.run(['git', '-c', 'credential.interactive=never', 'credential', 'fill'],
        input='protocol=https\nhost=github.com\n\n', capture_output=True, text=True, env=env, cwd=REPO)
    if credential.returncode:
        raise RuntimeError('Existing authenticated GitHub credentials unavailable; ZIP preserved in Downloads')
    values = dict(line.split('=', 1) for line in credential.stdout.splitlines() if '=' in line)
    token = values.get('password')
    if not token:
        raise RuntimeError('No existing GitHub credential; ZIP preserved in Downloads')
    opener = urllib.request.build_opener(SafeRedirect())

    def request(url, method='GET', data=None, content_type=None, binary=False):
        headers = {'Authorization': 'Bearer ' + token,
            'Accept': 'application/octet-stream' if binary else 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'FYP-verified-adapter-publication'}
        if data is not None:
            headers['Content-Type'] = content_type or 'application/json'
        with opener.open(urllib.request.Request(url, data=data, method=method, headers=headers), timeout=120) as response:
            return response.read() if binary else json.load(response)

    try:
        release = request(API + '/releases/tags/' + TAG)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise RuntimeError('GitHub release lookup failed: HTTP ' + str(error.code)) from None
        body = {'tag_name': TAG, 'target_commitish': commit,
            'name': 'Experimental final-step v8 adapter (training complete; behavioral comparison pending)',
            'body': 'Fresh Qwen/Qwen3.5-2B LoRA on v8-final-20261018-r1; 2 epochs, seed 42, 2234 steps. '
                'All 224 checkpoints verified by the supplied worker. Final teacher-forced validation loss: 0.0033881000708788633. '
                'This is an experimental final-step candidate; behavioral quality and the v6 comparison are pending, '
                'not a final-test/OpenAI winner. Independent human review remains incomplete. '
                'ZIP contains adapter/config/selection and provenance only: no base weights, optimizer states, caches or credentials. '
                'Download the checksum alongside the ZIP and verify before loading. '
                'Training evidence and local Codex paper handoff: ' + BASE + '/tree/' + commit + '/' + RELATIVE,
            'draft': False, 'prerelease': True}
        release = request(API + '/releases', 'POST', json.dumps(body).encode())
    if not release['prerelease'] or release['draft']:
        raise RuntimeError('Existing release has different visibility/type; no overwrite')
    assets = {item['name']: item for item in release.get('assets', [])}
    uploaded = []
    for path, content_type in [(archive, 'application/zip'), (checksum, 'text/plain')]:
        if path.name in assets:
            asset = assets[path.name]
            if asset['size'] != path.stat().st_size:
                raise RuntimeError('Existing release asset differs; preserved without overwrite')
        else:
            url = release['upload_url'].split('{', 1)[0] + '?' + urllib.parse.urlencode({'name': path.name})
            asset = request(url, 'POST', path.read_bytes(), content_type)
        if asset['state'] != 'uploaded' or asset['size'] != path.stat().st_size:
            raise RuntimeError('Incomplete release asset; preserved without overwriting')
        # Download the actual published bytes through the API, removing auth on CDN redirects.
        downloaded = request(asset['url'], binary=True)
        if hashlib.sha256(downloaded).hexdigest() != sha(path):
            raise RuntimeError('Published asset failed independent download SHA verification')
        uploaded.append({'name': asset['name'], 'id': asset['id'], 'url': asset['browser_download_url'],
            'bytes': asset['size'], 'sha256': sha(path), 'github_digest': asset.get('digest'),
            'download_sha256_verified': True})
    receipt.update(publication_status='published_prerelease_download_verified',
        published_at_UTC=now(), release_id=release['id'], release_url=release['html_url'],
        publication_metadata_commit=commit, assets=uploaded)
    write(RESULT / 'ADAPTER_RELEASE.json', receipt)
    manifest()
    print(json.dumps({'published': release['html_url'], 'zip_sha256': receipt['zip_sha256'],
                      'download_verified_assets': len(uploaded)}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'publish', 'manifest'])
    args = parser.parse_args()
    {'prepare': prepare, 'publish': publish, 'manifest': manifest}[args.action]()
