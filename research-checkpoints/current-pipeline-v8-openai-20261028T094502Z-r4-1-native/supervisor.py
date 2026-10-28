"""Detached bounded OpenAI rerun; no automatic inference retry."""
import ctypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import winreg

RUN=Path(__file__).resolve().parent
CONFIG=json.loads((RUN/'config.json').read_bytes())
CODE=Path(CONFIG['code']);REPO=Path(CONFIG['repo'])
sys.path.insert(0,str(CODE))
from controls import atomic,lock,sha


def status(value):
    atomic(RUN/'supervisor-status.json',{'at_UTC':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),**value})


def main():
    with lock(RUN/'supervisor.lock'):
        freeze=json.loads((RUN/'supervisor-source-freeze.json').read_bytes())
        if sha(__file__)!=freeze['supervisor_sha256']:raise RuntimeError('Supervisor source drift')
        if not json.loads((RUN/'server-auth-latest.json').read_bytes())['authentication_passed']:raise RuntimeError('Server authentication gate failed')
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:
            credential=winreg.QueryValueEx(key,'OPENAI_API_KEY')[0]
        env=dict(os.environ);env['OPENAI_API_KEY']=credential;env['OPENAI_BASE_URL']='https://api.openai.com/v1'
        for name in ('OPENAI_ORG_ID','OPENAI_PROJECT_ID','OPENAI_ORGANIZATION','OPENAI_PROJECT'):env.pop(name,None)
        env['PYTHONPATH']=os.pathsep.join([str(RUN/'offline-deps'),str(CODE)])
        awake=ctypes.windll.kernel32.SetThreadExecutionState
        if not awake(0x80000001):raise RuntimeError('Cannot inhibit sleep')
        try:
            for stage in ('pilot','main'):
                completion=RUN/(stage+'-completion.json')
                if completion.exists():
                    prior=json.loads(completion.read_bytes())
                    if prior['status']=='complete':
                        if prior['protocol_sha256']!=sha(RUN/'protocol.json'):raise RuntimeError('Completed protocol drift')
                        continue
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                logpath=RUN/(stage+'-worker-'+stamp+'.log')
                with logpath.open('ab',buffering=0) as log:
                    command=[CONFIG['python'],'-B','-u',str(CODE/'runner.py'),'--run',str(RUN),'--repo',str(REPO),'--stage',stage]
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
                    while child.poll() is None:
                        status({'status':'running','stage':stage,'worker_pid':child.pid,'log':str(logpath),'SLM':'verified reused responses; no new GPU inference'})
                        time.sleep(3)
                    if child.returncode:
                        status({'status':'stopped_preserved','stage':stage,'worker_exit':child.returncode,'reason':'No automatic paid retry; inspect sanitized worker status and ledger'})
                        return 2
                receipt=json.loads(completion.read_bytes())
                if receipt['status']!='complete':raise RuntimeError('Incomplete matched stage')
                if stage=='pilot' and not receipt['main_affordability_passed']:
                    status({'status':'stopped_budget_preflight','reason':'Original main plan not affordable within amended limits'});return 2
            status({'status':'evaluation_complete','main_calls':404,'fresh_OpenAI_main_calls':202,'reused_SLM_main_calls':202,'publication_pending':True})
            return 0
        except BaseException as error:
            status({'status':'stopped_preserved','error_type':type(error).__name__,'reason':'Preserve raw responses and ledger; no automatic retry'});return 2
        finally:awake(0x80000000)


if __name__=='__main__':raise SystemExit(main())
