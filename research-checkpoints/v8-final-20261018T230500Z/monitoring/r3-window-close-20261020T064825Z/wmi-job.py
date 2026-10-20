"""Run a fixed command outside the caller's terminal/job; logs stay on disk."""
import argparse,ctypes,hashlib,json,os,subprocess,sys,psutil
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--capsule',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
body=a.capsule.read_bytes()
if hashlib.sha256(body).hexdigest()!=a.sha256:raise ValueError('Launch capsule changed')
c=json.loads(body);kernel=ctypes.windll.kernel32;inside=ctypes.c_int()
kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetConsoleWindow.restype=ctypes.c_void_p
kernel.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
kernel.OpenProcess.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.c_uint];kernel.OpenProcess.restype=ctypes.c_void_p
kernel.CloseHandle.argtypes=[ctypes.c_void_p]
# Windows venv redirectors can allocate a hidden console for the actual Python.
# Detach this bootstrap before it starts the supplied worker command.
if kernel.GetConsoleWindow():kernel.FreeConsole()
if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(inside)):raise RuntimeError('Cannot verify job membership')
root_inside=ctypes.c_int();parent=kernel.OpenProcess(0x1000,False,os.getppid())
if not parent:raise RuntimeError('Cannot inspect WMI-created venv launcher')
try:
    if not kernel.IsProcessInJob(parent,None,ctypes.byref(root_inside)):raise RuntimeError('Cannot verify WMI root job membership')
finally:kernel.CloseHandle(parent)
receipt={'at_UTC':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'parent_pid':os.getppid(),
    'console_window':kernel.GetConsoleWindow() or 0,'in_job':bool(inside.value),
    'wmi_root_in_job':bool(root_inside.value),'command':c['command'],
    'capsule_sha256':a.sha256}
provider=psutil.Process(psutil.Process(os.getppid()).ppid())
receipt.update(provider_pid=provider.pid,provider_name=provider.name())
a.capsule.with_suffix('.started.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
if receipt['console_window'] or receipt['provider_name'].lower()!='wmiprvse.exe':
    raise RuntimeError('Isolation failed: detached process is not owned by the local WMI provider')
env=dict(os.environ);env.pop('OPENAI_API_KEY',None)
for path,expected in c.get('files_sha256',{}).items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Isolated launch helper changed')
env.update(c.get('environment',{}))
with Path(c['log']).open('ab',buffering=0) as log:
    try:
        result=subprocess.run(c['command'],cwd=c['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        receipt.update(finished_UTC=datetime.now(timezone.utc).isoformat(),returncode=result.returncode)
        a.capsule.with_suffix('.finished.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        sys.exit(result.returncode)
    except Exception as error:
        import traceback
        log.write(traceback.format_exc().encode());raise
