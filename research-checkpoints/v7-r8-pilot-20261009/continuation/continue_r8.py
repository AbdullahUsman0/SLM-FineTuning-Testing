"""Wait for verified pilot completion, run only local comparisons, then publish."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from evaluate_r8 import atomic, lock, now, sha, verify_freeze
from finish_r8 import finish

def run_pipeline(run):
    code=Path(__file__).resolve().parent
    config=json.loads((run/'launch-config.json').read_bytes())
    env=dict(os.environ,**config['env'])
    env.pop('OPENAI_API_KEY',None)
    def status(phase,**fields):
        atomic(run/'continuation-status.json',dict(status='running',phase=phase,pid=os.getpid(),updated_at=now(),**fields))
    with lock(run/'continuation.lock'):
        verify_freeze(run)
        if (run/'continuation-completion.json').exists():
            print('Continuation already complete; no inference repeated.',flush=True)
            return
        awake=ctypes.windll.kernel32.SetThreadExecutionState
        if not awake(0x80000001):raise RuntimeError('Cannot request awake state')
        training_process=None
        training_started=None
        first=json.loads((run/'pilot-status.json').read_bytes())
        if first.get('status')=='running':
            try:
                training_process=psutil.Process(first['pid'])
                training_started=training_process.create_time()
            except psutil.NoSuchProcess:pass
        try:
            print('Continuation armed: waiting for verified training, then stock/v6/v7 fresh r8 evaluation.',flush=True)
            while not (run/'completion.json').exists():
                pilot=json.loads((run/'pilot-status.json').read_bytes())
                if pilot['status']=='failed':raise RuntimeError('Pilot failed; no evaluation or automatic retraining started')
                if (run/'continuation-stop.request').exists():raise InterruptedError('Stop requested')
                if training_process is not None and (not training_process.is_running() or training_process.create_time()!=training_started):
                    raise RuntimeError('Training process exited without verified completion')
                status('waiting_for_verified_pilot_completion',pilot_status=pilot['status'])
                time.sleep(15)
            # Do not overlap model loading with the trainer's final cleanup.
            while training_process is not None and training_process.is_running() and training_process.create_time()==training_started:
                status('waiting_for_training_process_exit')
                time.sleep(2)
            completed=json.loads((run/'completion.json').read_bytes())
            if completed['status']!='complete' or completed['optimizer_steps']!=65:raise RuntimeError('Unexpected pilot completion')
            status('verifying_cached_base_weights')
            base=Path(config['env']['HF_HOME'])/'hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/model.safetensors-00001-of-00001.safetensors'
            if sha(base)!=config['base_weights_sha256']:raise ValueError('Cached base weights changed')
            base_stat=base.stat()
            for arm in ('stock','v6','v7'):
                if (run/'continuation-stop.request').exists():raise InterruptedError('Stop requested')
                if (base.stat().st_size,base.stat().st_mtime_ns)!=(base_stat.st_size,base_stat.st_mtime_ns):raise ValueError('Cached base changed between arms')
                verify_freeze(run)
                status('evaluating',arm=arm,planned_scored_calls=6240)
                with (run/f'evaluation-{arm}.console.log').open('ab') as stdout,(run/f'evaluation-{arm}.stderr.log').open('ab') as stderr:
                    process=subprocess.Popen([config['python'],'-B','-u',str(code/'evaluate_r8.py'),'--run',str(run),'--arm',arm],
                                             cwd=config['bundle'],env=env,stdout=stdout,stderr=stderr)
                    while process.poll() is None:
                        if (run/'continuation-stop.request').exists():
                            process.terminate();process.wait()
                            raise InterruptedError('Stop requested; completed jobs retained')
                        progress_path=run/'evaluation'/arm/'status.json'
                        progress=json.loads(progress_path.read_bytes()) if progress_path.exists() else {'status':'loading'}
                        status('evaluating',arm=arm,arm_pid=process.pid,progress=progress,planned_scored_calls=6240)
                        time.sleep(15)
                if process.returncode!=0:raise RuntimeError('Local evaluation failed for '+arm+'; inspect its stderr log')
                print('Completed arm '+arm,flush=True)
            status('verifying_comparing_packaging_and_publishing')
            result=finish(run)
            atomic(run/'continuation-completion.json',result)
            atomic(run/'continuation-status.json',dict(result,phase='complete'))
            print(json.dumps(result),flush=True)
        except BaseException as error:
            atomic(run/'continuation-status.json',{'status':'stopped' if isinstance(error,InterruptedError) else 'failed',
                   'phase':'stopped','error_type':type(error).__name__,'error':str(error),'updated_at':now(),
                   'automatic_retraining':False,'paid_API_calls':0})
            traceback.print_exc()
            raise
        finally:
            awake(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run_pipeline(args.run.resolve())
