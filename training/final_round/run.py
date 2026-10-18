"""Prepare, detach, monitor and resume one fixed final experimental GPU run."""
from __future__ import annotations
import argparse,ctypes,gzip,hashlib,importlib.util,json,math,os,platform,shutil,subprocess,sys,time,traceback
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from data_tools import ROOT,CORPUS,BASE_REVISION,FPY_REVISION,sha,atomic,dumps,use_fpy,verify

HERE=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def helper():
    spec=importlib.util.spec_from_file_location('final_trainer_helpers',ROOT/'training/train_lora.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@contextmanager
def run_lock(path):
    """OS-owned lock releases on a crash, unlike a stale PID-only lockfile."""
    with Path(path).open('a+b') as f:
        f.seek(0)
        if not f.read(1):f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def background(command,log):
    flags={}
    if os.name=='nt':flags['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
    else:flags['start_new_session']=True
    with Path(log).open('ab',buffering=0) as stream:
        return subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            cwd=ROOT,close_fds=True,**flags)

def code_lock():
    return {str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'training/train_lora.py',*sorted(HERE.glob('*.py'))]}

def frozen_environment():
    result={'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','0')}
    if os.environ.get('HF_HOME'):result['HF_HOME']=str(Path(os.environ['HF_HOME']).expanduser().resolve())
    for name in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_XET']:
        if name in os.environ:result[name]=os.environ[name]
    return result

def execution_environment(config):
    env=dict(os.environ);env.pop('OPENAI_API_KEY',None)
    env.update(config['environment'])
    env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    return env

def check_config(run):
    c=json.loads((run/'launch-config.json').read_bytes())
    lock=json.loads((run/'launch-config.sha256.json').read_bytes())
    if sha(run/'launch-config.json')!=lock['sha256']:raise ValueError('Launch config changed')
    if c['code_hashes']!=code_lock():raise ValueError('Frozen training/runner source changed')
    if sha(CORPUS/'manifest.json')!=c['corpus_manifest_sha256']:raise ValueError('Corpus identity changed')
    for item in c['inputs'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Materialized inputs changed')
    if sha(run/'preflight.json')!=c['preflight_sha256']:raise ValueError('Preflight changed')
    if sha(HERE/'TRAINING_AUTHORIZATION.json')!=c['training_authorization_sha256']:raise ValueError('Training scope receipt changed')
    return c

def prepare(run,fpy,seed):
    use_fpy(fpy);verified=verify(CORPUS,semantic=True)
    authorization=json.loads((HERE/'TRAINING_AUTHORIZATION.json').read_bytes())
    if authorization.get('corpus_manifest_sha256')!=verified['manifest_sha256'] or authorization.get('full_training_authorized') is not True:
        raise ValueError('Missing corpus-bound experimental training authorization')
    if run.exists():raise FileExistsError('Use a NEW unique run directory; use start/status for an existing run')
    run.parent.mkdir(parents=True,exist_ok=True)
    # 2220 steps / ten-step saves can retain about 45+ GB of adapter and optimizer state.
    if shutil.disk_usage(run.parent).free<80*1024**3:raise RuntimeError('Choose an output disk with at least 80 GiB free for all ten-step checkpoints')
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required; no CPU/NPU fallback')
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES before prepare/start')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('This frozen run requires BF16 CUDA; do not silently substitute precision')
    run.mkdir();(run/'inputs').mkdir()
    atomic(run/'status.json',{'status':'preparing','at':now(),'training_started':False})
    inventory={'python':sys.version,'platform':platform.platform(),'gpu':torch.cuda.get_device_name(0),
        'gpu_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda,
        'torch':torch.__version__,'disk_free_bytes':shutil.disk_usage(run).free,'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    atomic(run/'inventory.json',inventory)
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B',revision=BASE_REVISION)
    trainer=helper();inputs={};lengths={}
    manifest=json.loads((CORPUS/'manifest.json').read_bytes())
    for split in ('train','validation'):
        dest=run/'inputs'/f'{split}.jsonl'
        with gzip.open(CORPUS/'sft'/f'{split}.jsonl.gz','rb') as source,dest.open('xb') as target:shutil.copyfileobj(source,target)
        if sha(dest)!=manifest['files'][f'sft/{split}.jsonl.gz']['raw_sha256']:raise ValueError('Decompression changed input bytes')
        values=[]
        with dest.open(encoding='utf-8') as source:
            for line in source:values.append(trainer.token_length(tokenizer,json.loads(line)['messages']))
        lengths[split]={'examples':len(values),'max_tokens':max(values),'over_limit':sum(v>4096 for v in values)}
        if lengths[split]['over_limit']:raise ValueError('Overlength examples: revise under a NEW corpus version; no truncation')
        inputs[split]={'path':str(dest),'sha256':sha(dest),'examples':len(values)}
    atomic(run/'preflight.json',{'status':'pass','revision':BASE_REVISION,'max_length':4096,'truncation':False,'lengths':lengths,
        'chat_template_kwargs':{'enable_thinking':False},'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'corpus_verification':verified,'human_review_complete':False,'final_test_accessed':False})
    c={'schema_version':1,'created_at':now(),'python':sys.executable,'fpy':str(Path(fpy).resolve()),'seed':seed,
       'corpus_manifest_sha256':sha(CORPUS/'manifest.json'),'inputs':inputs,'epochs':2,'learning_rate':5e-5,
       'max_length':4096,'save_steps':10,'expected_optimizer_steps':math.ceil(inputs['train']['examples']/16)*2,
       'checkpoint_retention':'all','initialization':'fresh pinned stock base plus new LoRA; not continued from v6/v7',
       'base_revision':BASE_REVISION,'preflight_sha256':sha(run/'preflight.json'),'code_hashes':code_lock(),
       'development_diagnostic':'100 stratified validation conversations, component and rollout; not final test',
       'human_review_complete':False,'training_authorization_sha256':sha(HERE/'TRAINING_AUTHORIZATION.json'),'environment':frozen_environment()}
    atomic(run/'launch-config.json',c);atomic(run/'launch-config.sha256.json',{'sha256':sha(run/'launch-config.json')})
    atomic(run/'status.json',{'status':'ready','at':now(),'training_started':False,'expected_optimizer_steps':c['expected_optimizer_steps']})
    print(dumps({'status':'ready','run':str(run),'preflight':lengths,'expected_optimizer_steps':c['expected_optimizer_steps']}))

def train_command(run,c):
    command=[c['python'],str(ROOT/'training/train_lora.py'),'--model','Qwen/Qwen3.5-2B','--revision',BASE_REVISION,
        '--train',c['inputs']['train']['path'],'--validation',c['inputs']['validation']['path'],'--output',str(run/'full'),
        '--epochs','2','--learning-rate','5e-5','--max-length','4096','--save-steps','10','--early-stopping-patience','0','--seed',str(c['seed'])]
    if (run/'full/run-manifest.json').exists():command+=['--resume-from-checkpoint','auto']
    return command

def worker(run):
    # The lock is held by the worker for training AND the development diagnostic.
    with run_lock(run/'worker.lock'):
        c=check_config(run);awake=None;phase='training'
        if os.name=='nt':
            awake=ctypes.windll.kernel32.SetThreadExecutionState
            if not awake(0x80000001):raise RuntimeError('Windows sleep inhibition failed')
        try:
            env=execution_environment(c)
            def status(state,**items):atomic(run/'status.json',{'status':state,'phase':phase,'at':now(),'pid':os.getpid(),**items})
            completion=run/'training-completion.json'
            if not completion.exists():
                status('running',expected_optimizer_steps=c['expected_optimizer_steps'])
                command=train_command(run,c)
                with (run/'training-console.log').open('ab',buffering=0) as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                trainer=helper();manifest=trainer.load_manifest(run/'full')
                paths=sorted((run/'full').glob('checkpoint-*'),key=trainer.checkpoint_step)
                verified=[p for p in paths if trainer.verify_checkpoint(p,manifest['run_fingerprint']) is not None]
                if not verified or max(map(trainer.checkpoint_step,verified))!=c['expected_optimizer_steps']:raise ValueError('Final trainer progress not verified')
                adapter=run/'full/best-adapter'
                files={name:sha(adapter/name) for name in ['adapter_model.safetensors','adapter_config.json','selection.json']}
                atomic(completion,{'status':'complete','at':now(),'optimizer_steps':c['expected_optimizer_steps'],'checkpoint_steps':list(map(trainer.checkpoint_step,verified)),
                    'adapter':str(adapter),'adapter_files_sha256':files,'run_fingerprint':manifest['run_fingerprint'],
                    'corpus_manifest_sha256':c['corpus_manifest_sha256'],'selection':'fixed two-epoch final step; behavior not selected',
                    'human_review_complete':False,'paid_API_calls':0,'final_test_accessed':False})
            else:
                receipt=json.loads(completion.read_bytes())
                if receipt['corpus_manifest_sha256']!=c['corpus_manifest_sha256']:raise ValueError('Completion mismatch')
                for name,value in receipt['adapter_files_sha256'].items():
                    if sha(run/'full/best-adapter'/name)!=value:raise ValueError('Completed adapter changed')
            phase='development_diagnostic';status('running')
            command=[c['python'],str(HERE/'evaluate_development.py'),'--fpy',c['fpy'],'--adapter',str(run/'full/best-adapter'),'--output',str(run/'development-evaluation')]
            with (run/'development-console.log').open('ab',buffering=0) as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            status('complete',training_completion=str(completion),development_report=str(run/'development-evaluation/summary.json'),model_winner_established=False)
        except BaseException as exc:
            atomic(run/'status.json',{'status':'failed','phase':phase,'at':now(),'pid':os.getpid(),'error_type':type(exc).__name__,
                'resume':'Inspect logs and rerun start on THIS run; verified checkpoints only. No automatic restart from zero.'})
            traceback.print_exc();raise
        finally:
            if awake:awake(0x80000000)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','start','worker','status']);p.add_argument('--run',type=Path,required=True);p.add_argument('--fpy',type=Path);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();run=a.run.resolve()
    if a.action=='prepare':
        if not a.fpy:p.error('--fpy is required for prepare')
        prepare(run,a.fpy,a.seed)
    elif a.action=='worker':worker(run)
    elif a.action=='start':
        c=check_config(run)
        # Starting twice is harmless: only one worker can acquire the OS lock.
        process=background([c['python'],'-u',str(Path(__file__).resolve()),'worker','--run',str(run)],run/'worker-console.log')
        atomic(run/'last-launch.json',{'pid':process.pid,'at':now(),'launch_is_not_completion':True})
        time.sleep(2)
        if process.poll() is not None and process.returncode:raise RuntimeError('Worker exited early; inspect worker-console.log and status.json')
        print(dumps({'launched_pid':process.pid,'run':str(run),'status_file':str(run/'status.json'),'lock_screen_supported':True,'sleep_shutdown_logout_not_supported':True}))
    else:
        result={'run':str(run),'status':json.loads((run/'status.json').read_bytes())}
        markers=list((run/'full').glob('checkpoint-*/checkpoint-complete.json')) if (run/'full').exists() else []
        completed=[]
        for marker in markers:
            try:
                record=json.loads(marker.read_bytes())
                if record.get('complete') is True:completed.append(int(record['global_step']))
            except (ValueError,KeyError,TypeError):continue
        if completed:result['latest_reported_complete_checkpoint_step']=max(completed)
        result['status_note']='A stale running status after power loss is not proof of a live process. Resume still verifies checkpoints.'
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
