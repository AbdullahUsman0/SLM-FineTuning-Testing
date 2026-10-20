"""Local WMI launch with job breakaway; no credentials or interactive windows."""
import hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
def launch(config,command,label):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');control=Path(config['control'])
    bootstrap=control/'wmi-job.py'
    if hashlib.sha256(bootstrap.read_bytes()).hexdigest()!=config['wmi_job_sha256']:raise ValueError('Isolation bootstrap changed')
    capsule=control/(label+'-'+stamp+'.json');log=control/(label+'-'+stamp+'.log')
    hook=control/'windows-console-isolation/sitecustomize.py'
    if hashlib.sha256(hook.read_bytes()).hexdigest()!=config['console_hook_sha256']:raise ValueError('Console hook changed')
    body=json.dumps({'command':command,'cwd':config['repo'],'log':str(log),
        'environment':{'PYTHONPATH':str(hook.parent),'SLM_DETACH_WINDOWS_CONSOLE':'1',
                       'SLM_CONSOLE_PROOF_DIR':str(control/'console-isolation-proofs')},
        'files_sha256':{str(hook):config['console_hook_sha256']}},indent=2).encode()
    capsule.write_bytes(body);digest=hashlib.sha256(body).hexdigest()
    cmdline=subprocess.list2cmdline([config['python'],'-B',str(bootstrap),'--capsule',str(capsule),'--sha256',digest])
    request=control/(label+'-'+stamp+'-wmi-request.json')
    request.write_text(json.dumps({'commandline':cmdline,'cwd':config['repo']}),encoding='utf-8')
    script=control/(label+'-'+stamp+'-wmi.ps1')
    quoted="'"+str(request).replace("'","''")+"'"
    script.write_text("$ErrorActionPreference='Stop'\n$c=Get-Content -LiteralPath "+quoted+" -Raw | ConvertFrom-Json\n"+
        "$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0;CreateFlags=[uint32]16777736}\n"+
        "$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$c.commandline;CurrentDirectory=$c.cwd;ProcessStartupInformation=$startup}\n"+
        "[ordered]@{return_value=$r.ReturnValue;pid=$r.ProcessId}|ConvertTo-Json -Compress\n"+
        "if($r.ReturnValue -ne 0){exit 1}\n",encoding='utf-8')
    # This shell starts only a local WMI request. The child is created by the
    # provider, with BREAKAWAY_FROM_JOB | DETACHED_PROCESS | NEW_PROCESS_GROUP.
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-WindowStyle','Hidden','-ExecutionPolicy','Bypass','-File',str(script)],
        stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:raise RuntimeError('WMI launch failed: '+result.stderr.strip()+' '+result.stdout.strip())
    receipt=json.loads(result.stdout)
    for attempt in range(100):
        if capsule.with_suffix('.started.json').exists():
            proof=json.loads(capsule.with_suffix('.started.json').read_bytes())
            if proof['parent_pid']!=receipt['pid'] or proof['provider_name'].lower()!='wmiprvse.exe' or proof['console_window']:
                raise RuntimeError('WMI process did not pass isolation gate')
            receipt.update(at=datetime.now(timezone.utc).isoformat(),capsule=str(capsule),log=str(log),
                command=command,isolation=proof)
            return receipt
        time.sleep(.1)
    raise RuntimeError('WMI returned a PID but bootstrap isolation was not confirmed')
