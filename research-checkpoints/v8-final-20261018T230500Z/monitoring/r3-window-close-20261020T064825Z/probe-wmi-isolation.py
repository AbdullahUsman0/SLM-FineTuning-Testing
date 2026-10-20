"""Harmless nested-process test matching supplied starter/worker/trainer flags."""
import ctypes,json,os,subprocess,sys,time
from pathlib import Path
mode=sys.argv[1];folder=Path(sys.argv[2]);kernel=ctypes.windll.kernel32;inside=ctypes.c_int()
kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetConsoleWindow.restype=ctypes.c_void_p
kernel.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(inside)):raise RuntimeError('Cannot verify job membership')
record={'pid':os.getpid(),'parent_pid':os.getppid(),'console_window':kernel.GetConsoleWindow() or 0,'in_job':bool(inside.value)}
(folder/(mode+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
if mode=='starter':
    subprocess.Popen([sys.executable,'-B',__file__,'worker',str(folder)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
elif mode=='worker':
    subprocess.run([sys.executable,'-B',__file__,'trainer',str(folder)],check=True)
else:
    time.sleep(3);(folder/'completed.txt').write_text('Nested trainer survived starter exit, no console or job required.',encoding='utf-8')
