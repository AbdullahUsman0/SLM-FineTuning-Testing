"""Run a script with credentials removed, outbound networking and ledger access blocked."""
import os
from pathlib import Path
import runpy
import socket
import sqlite3
import sys

for name in list(os.environ):
    upper=name.upper()
    if any(part in upper for part in ('API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL','OPENAI_ORG','OPENAI_PROJECT','OPENAI_BASE_URL')):
        os.environ.pop(name,None)
connect=socket.socket.connect
connect_ex=socket.socket.connect_ex
def guarded(sock,address):
    if not isinstance(address,tuple) or address[0] not in {'localhost','127.0.0.1','::1'}:raise RuntimeError('Offline analysis: outbound network prohibited')
    return connect(sock,address)
def guarded_ex(sock,address):
    if not isinstance(address,tuple) or address[0] not in {'localhost','127.0.0.1','::1'}:raise RuntimeError('Offline analysis: outbound network prohibited')
    return connect_ex(sock,address)
socket.socket.connect=guarded;socket.socket.connect_ex=guarded_ex
original_sqlite=sqlite3.connect
def sqlite_guard(database,*args,**kwargs):
    if 'FYP-experiment-control' in str(database) and 'budget.sqlite3' in str(database):raise RuntimeError('Offline supplement: shared spending ledger access prohibited')
    return original_sqlite(database,*args,**kwargs)
sqlite3.connect=sqlite_guard
script=Path(sys.argv[1]).resolve();sys.argv=sys.argv[1:]
sys.path.insert(0,str(script.parent))
runpy.run_path(str(script),run_name='__main__')
