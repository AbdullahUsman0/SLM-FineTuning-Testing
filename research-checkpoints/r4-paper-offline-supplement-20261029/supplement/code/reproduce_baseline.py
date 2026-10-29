"""Reproduce original tests/replay in the prepared workspace, without inference."""
import json
import os
from pathlib import Path
import subprocess
from common import C,ROOT,RUNTIME,write

env=dict(os.environ)
for name in list(env):
    if any(s in name.upper() for s in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL','OPENAI_ORG','OPENAI_PROJECT','OPENAI_BASE_URL']):env.pop(name,None)
env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',CURRENT_PIPELINE_RUN=str(RUNTIME),TEMP=str(RUNTIME/'tmp'),TMP=str(RUNTIME/'tmp'),PYTHONPATH=os.pathsep.join([C['offline_deps'],str(RUNTIME/'fpy/src'),str(RUNTIME/'fpy'),str(RUNTIME/'evaluator')]))
assert not (ROOT/'t').exists(), 'Use a NEW workspace; never delete an existing temporary tree'
commands=[('baseline-tests',[C['python'],'-B',str(ROOT/'scripts/offline_guard.py'),str(RUNTIME/'run-offline-tests.py'),'-p','pytest_asyncio.plugin','-q',str(RUNTIME/'evaluator/test_r4.py'),str(RUNTIME/'fpy/tests'),'--basetemp',str(ROOT/'t'),'--override-ini','asyncio_mode=auto','--override-ini','cache_dir='+str(RUNTIME/'pytest-cache')],RUNTIME/'fpy'),('baseline-replay',[C['python'],'-B',str(ROOT/'scripts/offline_guard.py'),str(RUNTIME/'verify_recorded_main.py')],RUNTIME)]
receipts=[]
for label,command,cwd in commands:
    log=ROOT/'logs'/(label+'-reproduction.log')
    if log.exists():raise FileExistsError('Preserve earlier reproduction logs')
    with log.open('wb') as output:result=subprocess.run(command,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT)
    receipts.append({'stage':label,'command':command,'returncode':result.returncode,'log':str(log)})
    if result.returncode:break
write(ROOT/'deliverables/REPRODUCED_BASELINE.json',{'passed':len(receipts)==2 and all(x['returncode']==0 for x in receipts),'stages':receipts,'new_model_calls':0,'new_paid_calls':0,'shared_ledger_accessed':False})
print(json.dumps(receipts))
