"""Prepare a public, clearly incomplete launch snapshot after checkpoint verification."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,subprocess,sys,psutil,re

active=json.loads(Path(r'D:\SLM\active-v8-final.json').read_text(encoding='utf-8-sig'))
control=Path(active['control']);run=Path(active['run']);repo=Path(r'D:\SLM\SLM-FineTuning-Testing')
proof=json.loads((control/'first-checkpoint-verification.json').read_bytes())
sys.path.insert(0,str(repo/'training/final_round'))
import run as worker
trainer=worker.helper();manifest=trainer.load_manifest(run/'full')
receipt=trainer.verify_checkpoint(Path(proof['checkpoint']),manifest['run_fingerprint'])
state=json.loads((run/'status.json').read_bytes());process=psutil.Process(state['pid'])
if not process.is_running() or state['status']!='running':raise ValueError('Worker no longer verified live')
spec=importlib.util.spec_from_file_location('launch_followup_helpers',control/'followup-r4.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
helper.check_sources(json.loads((control/'workflow-config-r4.json').read_bytes()),repo)
result=repo/active['results_relative']
result.mkdir(parents=True,exist_ok=False)
(result/'.gitattributes').write_text('* -text whitespace=trailing-space,space-before-tab,cr-at-eol\n',encoding='ascii')
files={'inventory.json':run/'inventory.json','preflight.json':run/'preflight.json',
    'launch-config.json':run/'launch-config.json','launch-config.sha256.json':run/'launch-config.sha256.json',
    'training-run-manifest.json':run/'full/run-manifest.json',
    'followup.py':control/'followup-r4.py','workflow-config.json':control/'workflow-config-r4.json',
    'first-checkpoint-verification.json':control/'first-checkpoint-verification.json',
    'LAUNCH_HANDOFF.md':control/'HANDOFF.md','launch-notes.json':control/'launch-notes.json',
    'base-weights-verification.json':control/'base-weights-verification.json',
    'followup-publication-test.json':control/'followup-publication-test.json',
    'publish_launch_snapshot.py':Path(__file__)}
for name,source in files.items():helper.stable_copy(source,result/name)
memory=psutil.virtual_memory()
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
helper.write(result/'LAUNCH_STATUS.json',{'record_type':'historical_launch_snapshot_not_current_status',
    'observed_at_UTC':datetime.now(timezone.utc).isoformat(),'status':'training_active_complete_checkpoint_verified',
    'source_commit':active['source_commit'],'run':str(run),'worker_PID':state['pid'],
    'checkpoint_step':receipt['global_step'],'expected_optimizer_steps':2234,'checkpoint':proof['checkpoint'],
    'all_checkpoint_artifact_hashes_independently_verified':True,'GPU_driver_memory_utilization_csv':hardware,
    'RAM_total_bytes':memory.total,'RAM_available_bytes':memory.available,
    'final_adapter_available':False,'development_measurements_available':False,'training_complete':False,
    'followup':'Supplied v8 diagnostic; identical verified-v6 diagnostic; verified local adapter ZIP; safe result-record commit/push.',
    'paid_API_calls':0,'independent_human_review_complete':False,'final_test_accessed':False,'model_winner_established':False})
(result/'LAUNCH_README.md').write_text(
    '# Historical launch snapshot: v8 training is still in progress\n\n'
    'The fixed two-epoch experimental run started from source commit bd36ff605d1d7e6c5f79de6e65d60785e0afe49d. '
    'LAUNCH_STATUS.json records an observed live CUDA worker and an independently SHA-verified complete checkpoint. '
    'This snapshot is not a completed training result or development evaluation. No final adapter or measured development scores exist at this launch snapshot.\n\n'
    'The exact run and environment paths refer to the GPU training PC. Read LAUNCH_HANDOFF.md for live status and recovery commands. '
    'The supplied worker owns training and the v8 diagnostic; the detached companion owns the same-cohort v6 diagnostic, local adapter packaging and final scientific-record publication. '
    'The final-step best-adapter folder will not be behaviorally selected. No paid API calls or final-test labels are authorized. '
    'Human annotation remains pending; training loss/token accuracy is not extraction F1.\n\n'
    'When completion.json, comparison.json and README.md arrive later, use those for final development measurements. '
    'Keep this dated launch record as provenance. Do not start a duplicate trainer because the public record is a snapshot. '
    'The frozen training/diagnostic code and settings were not changed while creating this metadata. No weights or optimizer states are published here.\n',encoding='utf-8')
credential=re.compile(r'(?<![A-Za-z0-9])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,})')
for path in result.rglob('*'):
    if path.is_file() and credential.search(path.read_text(encoding='utf-8-sig')):raise ValueError('Credential-shaped value in launch snapshot')
helper.write(result/'LAUNCH_ARTIFACT_MANIFEST.json',{'scope':'Incomplete historical launch snapshot, no final results',
    'files_sha256':{p.relative_to(result).as_posix():helper.sha(p) for p in result.rglob('*') if p.is_file()}})
print(json.dumps({'launch_records_prepared':str(result),'checkpoint_step':receipt['global_step'],'training_complete':False}))
