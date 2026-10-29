"""Credential-free offline stage launcher with preserved logs and receipts."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1];c=json.loads((root/'workspace.json').read_bytes())
for name in list(os.environ):
    if any(p in name.upper() for p in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL','OPENAI_ORG','OPENAI_PROJECT','OPENAI_BASE_URL']):os.environ.pop(name,None)
env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=c['offline_deps'],TEMP=str(root/'runtime/tmp'),TMP=str(root/'runtime/tmp'))
script=root/'scripts'/sys.argv[1];stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');log=root/'logs'/(script.stem+'-'+stamp+'.log')
command=[c['python'],'-B',str(root/'scripts/offline_guard.py'),str(script),*sys.argv[2:]]
with log.open('wb') as output:result=subprocess.run(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
receipt={'stage':script.name,'command':command,'log':str(log),'returncode':result.returncode,'credentials_removed':True,'outbound_network_blocked':True,'shared_ledger_access_blocked':True}
log.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(log.read_text(encoding='utf-8',errors='replace')[-3500:]);print(json.dumps(receipt));raise SystemExit(result.returncode)
