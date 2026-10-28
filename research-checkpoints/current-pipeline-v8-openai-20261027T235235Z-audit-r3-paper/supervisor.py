"""Single bounded evaluation supervisor; reads fresh user credential, never logs it."""
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import winreg

RUN=Path(__file__).resolve().parent
CONFIG=json.loads((RUN/'supervisor-config.json').read_bytes())
CODE=Path(CONFIG['code']);REPO=Path(CONFIG['repo'])
sys.path.insert(0,str(CODE))
from controls import Ledger,atomic,lock,sha


def status(value):
    atomic(RUN/'supervisor-status.json',{'at_UTC':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),**value})


def main():
    with lock(RUN/'supervisor.lock'):
        for name,wanted in CONFIG['supervisor_sources_sha256'].items():
            if sha(RUN/name)!=wanted:raise RuntimeError('Frozen supervisor/delivery source changed')
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as envkey:
            key=winreg.QueryValueEx(envkey,'OPENAI_API_KEY')[0]
        authentication=json.loads((RUN/'server-auth-latest.json').read_bytes())
        if not authentication['authentication_passed']:raise RuntimeError('Server metadata authentication has not passed')
        env=dict(os.environ);env['OPENAI_API_KEY']=key;env['OPENAI_BASE_URL']='https://api.openai.com/v1'
        for name in ('OPENAI_ORG_ID','OPENAI_PROJECT_ID','OPENAI_ORGANIZATION','OPENAI_PROJECT'):env.pop(name,None)
        hook=CONFIG['console_hook_directory']
        env['PYTHONPATH']=os.pathsep.join([hook,str(RUN/'offline-deps'),str(CODE)])
        awake=ctypes.windll.kernel32.SetThreadExecutionState
        if not awake(0x80000001):raise RuntimeError('Cannot inhibit remote sleep')
        try:
            status({'status':'starting','credential_source':'fresh Windows user environment; never inherited key','provider':'https://api.openai.com/v1','no_automatic_paid_retries':True})
            for stage in ('pilot','main'):
                receipt=RUN/(stage+'-completion.json')
                if receipt.exists():
                    existing=json.loads(receipt.read_bytes())
                    if existing['status']=='complete':
                        if existing['protocol_sha256']!=sha(RUN/'protocol.json'):raise RuntimeError('Completed stage protocol mismatch')
                        continue
                for name,wanted in CONFIG['supervisor_sources_sha256'].items():
                    if sha(RUN/name)!=wanted:raise RuntimeError('Supervisor source drift')
                cmd=[CONFIG['python'],'-B','-u',str(CODE/'runner.py'),'--run',str(RUN),'--repo',str(REPO),'--stage',stage]
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                with (RUN/(stage+'-worker-'+stamp+'.log')).open('ab',buffering=0) as log:
                    child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
                    while child.poll() is None:
                        status({'status':'running','stage':stage,'worker_pid':child.pid,'log':str(RUN/log.name) if not Path(log.name).is_absolute() else str(log.name)})
                        time.sleep(10)
                    if child.returncode:
                        status({'status':'stopped_preserved','stage':stage,'worker_returncode':child.returncode,'reason':'Inspect sanitized worker status and shared ledger; no automatic retry'})
                        return 2
                complete=json.loads(receipt.read_bytes())
                if complete['status']!='complete':raise RuntimeError('Planned paired stage incomplete')
                if stage=='pilot' and not complete['main_affordability_passed']:
                    status({'status':'budget_preflight_stopped','reason':'Pilot projection cannot defend the original main design within remaining budget'})
                    from delivery import deliver
                    deliver(RUN,REPO,CODE,key,'budget_preflight_stopped')
                    return 2
            status({'status':'packaging','scope':'matched development complete; no universal winner established'})
            from delivery import deliver
            delivered=deliver(RUN,REPO,CODE,key)
            status({'status':'complete','delivery':delivered})
            return 0
        except BaseException as error:
            status({'status':'stopped_preserved','error_type':type(error).__name__,'reason':'Scientific, infrastructure or delivery failure; preserve all raw calls and shared ledger. No automatic inference retry.'})
            return 2
        finally:
            awake(0x80000000)


if __name__=='__main__':raise SystemExit(main())
