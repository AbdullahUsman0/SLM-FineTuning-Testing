"""Run one authorized offline pilot and record verified completion/failure."""
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

RUN = Path(sys.argv[1]).resolve()
CONFIG = json.loads((RUN / 'launch-config.json').read_bytes())

def write_status(value):
    tmp = RUN / 'pilot-status.json.tmp'
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump({**value, 'updated_at_utc': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write('\n')
    tmp.replace(RUN / 'pilot-status.json')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

os.environ.update(CONFIG['env'])
os.chdir(CONFIG['bundle'])
assert sha(Path(CONFIG['trainer'])) == CONFIG['trainer_sha256']
for data in CONFIG['inputs'].values():
    assert sha(Path(data['path'])) == data['sha256']
awake = ctypes.windll.kernel32.SetThreadExecutionState
assert awake(0x80000001), 'Cannot maintain awake state'
try:
    write_status({'status': 'running', 'phase': 'loading_and_training', 'pid': os.getpid(),
                  'expected_steps': 65, 'training_output': CONFIG['output'], 'automatic_restarts': False})
    import torch
    torch.set_num_threads(4)
    sys.argv = [CONFIG['trainer'], '--model', 'Qwen/Qwen3.5-2B', '--revision', CONFIG['settings']['revision'],
                '--train', CONFIG['inputs']['train']['path'], '--validation', CONFIG['inputs']['validation']['path'],
                '--output', CONFIG['output'], '--epochs', '1', '--learning-rate', '5e-5', '--max-length', '3584',
                '--save-steps', '10', '--early-stopping-patience', '0', '--seed', '42']
    print('Starting explicitly authorized v7 r8 pilot; 65 optimizer steps, offline stock initialization.', flush=True)
    runpy.run_path(CONFIG['trainer'], run_name='__main__')
    spec = importlib.util.spec_from_file_location('v7_pilot_verifier', CONFIG['trainer'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = Path(CONFIG['output'])
    manifest = module.load_manifest(output)
    checkpoints = sorted(output.glob('checkpoint-*'), key=module.checkpoint_step)
    assert checkpoints and module.checkpoint_step(checkpoints[-1]) == 65
    verified = [module.verify_checkpoint(p, manifest['run_fingerprint']) for p in checkpoints]
    adapter = output / 'best-adapter'
    hashes = {p.name: sha(p) for p in (adapter / 'adapter_model.safetensors', adapter / 'adapter_config.json')}
    receipt = {'status': 'complete', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
               'run_fingerprint': manifest['run_fingerprint'], 'optimizer_steps': 65,
               'checkpoint_steps': [module.checkpoint_step(p) for p in checkpoints],
               'checkpoint_hash_verification': 'passed', 'adapter': str(adapter), 'adapter_sha256': hashes,
               'evaluation': json.loads((output / 'evaluation.json').read_bytes()),
               'selection': 'experimental final-step pilot adapter; behavioral evaluation pending',
               'human_review_complete': False, 'final_labels_accessed': False, 'paid_API_calls': 0}
    with (RUN / 'completion.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    write_status({'status': 'complete', 'optimizer_steps': 65, 'adapter': str(adapter), 'behavioral_evaluation': 'pending'})
    print('Pilot complete; checkpoints and adapter hashes verified. No behavioral winner selected.', flush=True)
except BaseException as error:
    write_status({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                  'automatic_restarts': False, 'training_output': CONFIG['output']})
    traceback.print_exc()
    raise
finally:
    awake(0x80000000)
