"""Windows process-lifetime repair only; no model/data/trainer modifications."""
import os
if os.name=='nt' and os.environ.get('SLM_DETACH_WINDOWS_CONSOLE')=='1':
    try:
        import ctypes,hashlib,json
        from pathlib import Path
        from datetime import datetime,timezone
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.GetConsoleWindow.restype=ctypes.c_void_p
        kernel.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
        before=kernel.GetConsoleWindow() or 0
        if before and not kernel.FreeConsole():raise RuntimeError('FreeConsole failed')
        after=kernel.GetConsoleWindow() or 0
        if after:raise RuntimeError('Console detachment failed')
        inside=ctypes.c_int()
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(inside)):
            raise RuntimeError('Cannot verify job membership')
        proof={'at_UTC':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'parent_pid':os.getppid(),
            'console_before':before,'console_after':after,'in_job':bool(inside.value),
            'hook_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        folder=Path(os.environ['SLM_CONSOLE_PROOF_DIR']);folder.mkdir(parents=True,exist_ok=True)
        (folder/(str(os.getpid())+'.json')).write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    except BaseException as error:
        # sitecustomize import errors normally allow startup to continue. Fail
        # closed for this process instead of silently losing the isolation gate.
        try:
            from pathlib import Path
            folder=Path(os.environ['SLM_CONSOLE_PROOF_DIR']);folder.mkdir(parents=True,exist_ok=True)
            (folder/(str(os.getpid())+'.error.txt')).write_text(type(error).__name__+': '+str(error),encoding='utf-8')
        except BaseException:pass
        os._exit(86)
