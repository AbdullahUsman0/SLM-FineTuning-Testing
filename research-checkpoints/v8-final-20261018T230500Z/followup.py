"""Local v8 completion supervisor: v6 development reference, verified packaging and safe publication.

No training settings, labels or final-test inputs are changed here. Training belongs
to the supplied detached worker. A training failure is retained for inspection;
this companion never restarts from zero or launches a second trainer.
"""
from pathlib import Path
import argparse,ctypes,csv,gzip,hashlib,json,os,re,shutil,subprocess,sys,time,zipfile
from datetime import datetime,timezone


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def stable_copy(source,target):
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target):raise ValueError('Refusing to replace different artifact: '+str(target))
    else:shutil.copyfile(source,target)


def checked_zip(path,members):
    path=Path(path);hashes={name:sha(source) for name,source in members.items()}
    if not path.exists():
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as package:
            for name,source in sorted(members.items()):package.write(source,name)
            package.writestr('ARCHIVE_MANIFEST.json',json.dumps({'files_sha256':hashes},sort_keys=True,indent=2))
    with zipfile.ZipFile(path) as package:
        if package.testzip() is not None or set(package.namelist())!=set(members)|{'ARCHIVE_MANIFEST.json'}:raise ValueError('Archive CRC/inventory mismatch')
        if json.loads(package.read('ARCHIVE_MANIFEST.json'))['files_sha256']!=hashes:raise ValueError('Archive belongs to different inputs')
        for name,wanted in hashes.items():
            if hashlib.sha256(package.read(name)).hexdigest()!=wanted:raise ValueError('Archived member SHA mismatch')
    checksum=sha(path);sidecar=path.with_suffix(path.suffix+'.sha256');body=checksum+'  '+path.name+'\n'
    if sidecar.exists() and sidecar.read_text(encoding='ascii')!=body:raise ValueError('Existing archive checksum differs')
    if not sidecar.exists():sidecar.write_text(body,encoding='ascii')
    return {'filename':path.name,'sha256':checksum,'bytes':path.stat().st_size,'members_sha256':hashes,'CRC_and_extracted_SHA256_verified':True}


def validate_runtime(folder):
    runtime=read(folder/'runtime.json')
    if not runtime['active_adapters'] or any(not device.startswith('cuda:') for device in runtime['parameter_devices']):
        raise ValueError('Development runtime did not use active LoRA entirely on CUDA')
    summary=read(folder/'summary.json')
    if summary['status']!='complete' or summary['conversations']!=100 or summary['paid_API_calls']!=0:raise ValueError('Incomplete/wrong development diagnostic')
    if sha(folder/'calls.jsonl')!=summary['journal_sha256']:raise ValueError('Diagnostic journal SHA mismatch')
    return summary


def fail_rows(folder):
    findings=[]
    with gzip.open(folder/'scored-records.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);flags=[]
            for key,label in [('json_valid','invalid_JSON'),('schema_valid','invalid_schema'),('admitted','rejected'),('complete_state','incomplete_state')]:
                if not row[key]:flags.append(label)
            if row['hit_generation_limit']:flags.append('generation_limit')
            if row['extra_slot_flags']:flags.append('extra_current_turn_slots')
            if row['forbidden_slot_flags']:flags.append('forbidden_slots')
            if row['gold_correction'] and not row['correct_correction']:flags.append('incorrect_correction')
            if row['unknown_turn'] and not (row['unknown_no_updates'] and row['unknown_preserved_nonintent']):flags.append('unknown_handling')
            if flags:findings.append({key:row[key] for key in ('key','domain','domain_family','language','category','error_type','extra_slot_flags','forbidden_slot_flags')}|{'findings':flags})
    return findings


def check_sources(config,repo):
    if sha(__file__)!=config['followup_sha256']:raise ValueError('Frozen followup source changed')
    for name,expected in config['sources_sha256'].items():
        if sha(repo/name)!=expected:raise ValueError('Frozen experiment source changed: '+name)


def granular_breakdown(folder):
    import statistics
    with gzip.open(folder/'scored-records.jsonl.gz','rt',encoding='utf-8') as stream:
        records=[json.loads(line) for line in stream]
    result={}
    for track in ('component','rollout'):
        result[track]={}
        for domain in sorted({r['domain'] for r in records}):
            items=[r for r in records if r['key'][0]==track and r['domain']==domain]
            tp,fp,fn=[sum(r['confusion'][i] for r in items) for i in range(3)]
            result[track][domain]={'calls':len(items),'tp':tp,'fp':fp,'fn':fn,
                'exact_F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
                'json_valid':sum(r['json_valid'] for r in items),'admitted':sum(r['admitted'] for r in items),
                'extra_slot_flags_including_rejected':sum(r['extra_slot_flags'] for r in items),
                'complete_state_turns':sum(r['complete_state'] for r in items),
                'median_extraction_seconds':statistics.median(r['latency_seconds'] for r in items)}
    return result


def scientific_outputs(config,repo,run,control):
    check_sources(config,repo)
    sys.path.insert(0,str(repo/'training/final_round'))
    import run as worker
    launch=worker.check_config(run);trainer=worker.helper();manifest=trainer.load_manifest(run/'full')
    completion=read(run/'training-completion.json')
    if completion['optimizer_steps']!=2234 or launch['expected_optimizer_steps']!=2234:raise ValueError('Unexpected completed training schedule')
    checkpoint_steps=[]
    for path in sorted((run/'full').glob('checkpoint-*'),key=trainer.checkpoint_step):
        trainer.verify_checkpoint(path,manifest['run_fingerprint']);checkpoint_steps.append(trainer.checkpoint_step(path))
    expected_steps=list(range(10,2234,10))+[2234]
    if checkpoint_steps!=expected_steps or checkpoint_steps!=completion['checkpoint_steps']:raise ValueError('Missing/unverified retained checkpoint')
    for name,wanted in completion['adapter_files_sha256'].items():
        if sha(run/'full/best-adapter'/name)!=wanted:raise ValueError('Completed adapter hash changed')
    summaries={'v8':validate_runtime(run/'development-evaluation')}
    v6=config.get('v6');v6_status={'status':'unavailable','reason':'Verified v6 files were not present at launch'}
    if v6:
        adapter=Path(v6['path'])
        for name,wanted in v6['sha256'].items():
            if sha(adapter/name)!=wanted:raise ValueError('Frozen v6 adapter changed')
        output=run/'v6-same-development-diagnostic'
        if not (output/'summary.json').exists():
            write(control/'followup-status.json',{'status':'running_v6_development','run':str(run),'output':str(output)})
            command=[launch['python'],'-B','-u',str(repo/'training/final_round/evaluate_development.py'),
                '--fpy',launch['fpy'],'--adapter',str(adapter),'--output',str(output)]
            with (run/'v6-development-console.log').open('ab',buffering=0) as log:
                subprocess.run(command,cwd=repo,env=worker.execution_environment(launch),stdout=log,stderr=subprocess.STDOUT,check=True)
        summaries['v6']=validate_runtime(output)
        left,right=read(run/'development-evaluation/freeze.json'),read(output/'freeze.json')
        for key in ('version','corpus_manifest_sha256','cases_sha256','scenario_ids','generation','policy','base_revision','runner_sha256','data_tools_sha256','provider_sources'):
            if left[key]!=right[key]:raise ValueError('Development comparison is unmatched: '+key)
        v6_status={'status':'matched_development_complete','adapter':v6}
    result=repo/config['results_relative'];result.mkdir(parents=True,exist_ok=True)
    (result/'.gitattributes').write_text('* -text whitespace=trailing-space,space-before-tab,cr-at-eol\n',encoding='ascii')
    sources={'inventory.json':run/'inventory.json','preflight.json':run/'preflight.json',
        'launch-config.json':run/'launch-config.json','launch-config.sha256.json':run/'launch-config.sha256.json',
        'training-completion.json':run/'training-completion.json','training-run-manifest.json':run/'full/run-manifest.json',
        'training-evaluation.json':run/'full/evaluation.json','training-events.jsonl':run/'full/events.jsonl',
        'followup.py':Path(__file__),'workflow-config.json':control/'workflow-config-r4.json'}
    for name,path in sources.items():stable_copy(path,result/name)
    (result/'ENVIRONMENT_FREEZE.txt').write_text(manifest['runtime']['pip_freeze'],encoding='utf-8')
    tests=Path(config['tests_log']).read_text(encoding='utf-16')
    (result/'PREPARATION_TESTS.txt').write_text(tests,encoding='utf-8')
    rows=[];failures={}
    for arm,summary in summaries.items():
        folder=run/('development-evaluation' if arm=='v8' else 'v6-same-development-diagnostic')
        for name in ('summary.json','freeze.json','runtime.json'):stable_copy(folder/name,result/arm/name)
        archive=checked_zip(result/(arm+'-development-raw.zip'),{name:folder/name for name in
            ('calls.jsonl','scored-records.jsonl.gz','freeze.json','runtime.json','summary.json')})
        write(result/arm/'raw-archive-receipt.json',archive)
        failures[arm]=fail_rows(folder)
        write(result/arm/'failure-findings.json',{'scope':'Automated development triage, not independent human adjudication','calls':failures[arm]})
        for track,metrics in summary['tracks'].items():
            row={'arm':arm,'track':track,'calls':metrics['calls'],'exact_f1':metrics['nonintent_exact']['f1'],
                'precision':metrics['nonintent_exact']['precision'],'recall':metrics['nonintent_exact']['recall']}
            row.update({key:metrics[key] for key in ('json_valid','schema_valid','admitted','extra_slot_flags_including_rejected',
                'forbidden_slot_flags_including_rejected','correct_correction_calls','complete_state_after_correction',
                'complete_state_turns','unknown_successful_empty_calls','unknown_preserved_nonintent_state','median_extraction_seconds')})
            rows.append(row)
    with (result/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    regressions=[]
    if 'v6' in summaries:
        for track in ('component','rollout'):
            for scope in ('overall','en','ur','mixed','weather','economics','combined'):
                a=summaries['v8']['tracks'][track];b=summaries['v6']['tracks'][track]
                if scope in ('en','ur','mixed'):a,b=a['languages'][scope],b['languages'][scope]
                elif scope!='overall':a,b=a['domains'][scope],b['domains'][scope]
                changes={key:a[key]-b[key] for key in ('json_valid','schema_valid','admitted','extra_slot_flags_including_rejected','forbidden_slot_flags_including_rejected','complete_state_turns')}
                changes['exact_f1']=a['nonintent_exact']['f1']-b['nonintent_exact']['f1']
                regressions.append({'track':track,'scope':scope,'v8_minus_v6':changes,
                    'F1_regressed':changes['exact_f1']<0,'extra_flags_increased':changes['extra_slot_flags_including_rejected']>0})
    write(result/'comparison.json',{'scope':'v8-development-diagnostic-1; inspected synthetic development, not final test',
        'rows':rows,'models':summaries,'v6_status':v6_status,'subgroup_differences':regressions,
        'model_winner_established':False,'confidence_intervals_implemented':False,'independent_human_review_complete':False,'paid_API_calls':0})
    granular={arm:granular_breakdown(run/('development-evaluation' if arm=='v8' else 'v6-same-development-diagnostic')) for arm in summaries}
    write(result/'granular-domain-breakdown.json',{'scope':'Prespecified development subgroup description; all 20 domains, no significance test','models':granular})
    adapter=run/'full/best-adapter';adapter_readme=run/'ADAPTER_PACKAGE_README.md'
    adapter_readme.write_text('# Experimental final-step v8 LoRA adapter\n\n'
        'Base: Qwen/Qwen3.5-2B, revision 15852e8c16360a2fea060d615a32b45270f8a8fc. '
        'Dataset: v8-final-20261018-r1; fresh LoRA; two epochs; seed 42; 2,234 optimizer steps. '
        'The folder name best-adapter denotes the final step, not behavioral selection. '
        'Synthetic development results do not establish a final-test or OpenAI winner.\n\n'
        'Runtime: Python 3.12, CUDA PyTorch 2.11.0+cu128, transformers 5.15.0, peft 0.20.0, '
        'accelerate 1.14.0, safetensors 0.8.0; parser pydantic 2.13.5, jsonschema 4.26.0. '
        'Use the repository training/final_round README for pinned fpy and development commands. '
        'Load on the pinned stock base; do not stack adapters. This archive excludes base weights, '
        'optimizer state, credentials and caches. Full resumable checkpoints remain in the original run.\n',encoding='utf-8')
    archive_path=Path.home()/'Downloads'/(run.name+'-adapter.zip')
    archive_info=checked_zip(archive_path,{'best-adapter/adapter_model.safetensors':adapter/'adapter_model.safetensors',
        'best-adapter/adapter_config.json':adapter/'adapter_config.json','best-adapter/selection.json':adapter/'selection.json',
        'training-completion.json':run/'training-completion.json','README.md':adapter_readme})
    write(result/'adapter-package.json',archive_info|{'location_on_training_PC':str(archive_path),'optimizer_checkpoints_included':False,'uploaded_to_normal_Git':False})
    lines=['# Fixed v8 training and local development diagnostic','',
        'Completed the authorized fixed experiment. These are new inspected synthetic DEVELOPMENT results. '
        'No final-test labels or paid APIs were used; independent human annotation remains incomplete. '
        'There is no statistical significance test or established model winner.','',
        '| Model | Track | Exact F1 | Precision | Recall | JSON | Admitted | Extra flags | Median seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['arm']} | {row['track']} | {row['exact_f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['json_valid']}/{row['calls']} | {row['admitted']}/{row['calls']} | {row['extra_slot_flags_including_rejected']} | {row['median_extraction_seconds']:.3f} |")
    lines+=['','Run: `'+str(run)+'`. GPU: '+manifest['runtime']['devices'][0]['name']+'. '
        'Completed 2,234/2,234 optimizer steps; '+str(len(checkpoint_steps))+' complete SHA-verified checkpoints retained. '
        'Seed 42; fresh rank-16 LoRA; two epochs; pinned stock base; effective batch 16; no truncation.','',
        'The isolated environment reuses the existing verified CUDA packages through a read-only site-packages path; '
        'only missing JSON Schema dependencies were installed in the new environment. Preserve both environments for resume. '
        'launch-config records the actual dedicated pinned fpy checkout; the historical trainer also logs a sibling fpy location '
        'as incidental repository metadata, which is not the dependency selected by this diagnostic.','',
        'comparison.json contains English/Roman Urdu/mixed and domain-family summaries, seeded-repair exclusion, '
        'and v8-minus-v6 differences with regression flags. granular-domain-breakdown.json adds all 20 individual domains. Per-arm failure-findings.json lists all structural, grounding/admission, '
        'extra/forbidden-slot, correction, unknown and accumulated-state failures. Extra flags include rejected/repeated emissions. '
        'Do not compare these scores directly with old cohorts or use training loss as extraction F1.','',
        'Adapter ZIP and checksum are local to the training PC Downloads, documented in adapter-package.json. '
        'They contain no base weights or optimizer state. They have not been uploaded as release assets. '
        'Raw development journals are compressed here with member SHA manifests. Final-step naming does not establish a best model.','',
        'Resume the SAME run with the supplied training/final_round/run.py start command after inspecting a failure; '
        'never edit global_step. The worker selects only a complete hash-verified checkpoint. '
        'This followup refuses duplicate supervision and preserves training failures for inspection.']
    (result/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    commands={'prepare':config['prepare_command'],'start':config['start_command'],
        'v6_diagnostic':[launch['python'],'-B',str(repo/'training/final_round/evaluate_development.py'),'--fpy',launch['fpy'],
            '--adapter',v6['path'] if v6 else '<verified-v6-missing>','--output',str(run/'v6-same-development-diagnostic')],
        'followup':[launch['python'],'-B',str(Path(__file__)),'--config',str(control/'workflow-config-r4.json')]}
    write(result/'EXACT_COMMANDS.json',commands)
    write(result/'completion.json',{'status':'verified_training_and_local_development_complete','optimizer_steps':2234,
        'verified_checkpoint_steps':checkpoint_steps,'run':str(run),'adapter_files_sha256':completion['adapter_files_sha256'],
        'v6_status':v6_status['status'],'new_paid_API_calls':0,'final_test_accessed':False,'model_winner_established':False})
    token=re.compile(r'(?<![A-Za-z0-9])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,})')
    for path in result.rglob('*'):
        if path.is_file() and path.suffix in {'.txt','.md','.json','.py','.jsonl','.csv'}:
            if token.search(path.read_text(encoding='utf-8-sig')):raise ValueError('Credential-shaped literal in result record')
    for path in result.glob('*.zip'):
        with zipfile.ZipFile(path) as package:
            for name in package.namelist():
                content=package.read(name)
                if name.endswith('.gz'):content=gzip.decompress(content)
                if token.search(content.decode('utf-8-sig')):
                    raise ValueError('Credential-shaped literal in compressed development record')
    files={p.relative_to(result).as_posix():sha(p) for p in result.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'}
    write(result/'artifact-manifest.json',{'files_sha256':files,'weights_datasets_optimizer_states_included':False})
    return result


def publish(config,repo,result):
    env=dict(os.environ);env.pop('OPENAI_API_KEY',None);env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    def git_at(folder,*args):
        return subprocess.check_output(['git','-c','credential.interactive=never',*args],cwd=folder,env=env,text=True,stderr=subprocess.STDOUT).strip()
    branch='experiment/v5-grounded-20260919'
    git_at(repo,'fetch','origin',branch)
    remote=git_at(repo,'rev-parse','origin/'+branch)
    staged=git_at(repo,'diff','--cached','--name-only').splitlines()
    clean_index=not any(not p.startswith(config['results_relative']+'/') for p in staged)
    upstream_ancestor=subprocess.run(['git','merge-base','--is-ancestor',remote,'HEAD'],cwd=repo).returncode==0
    normal=(git_at(repo,'branch','--show-current')==branch and clean_index and upstream_ancestor)
    publication=repo
    if not normal:
        publication=Path(config['control'])/('publication-worktree-'+str(time.time_ns()))
        git_at(repo,'worktree','add','--detach',str(publication),remote)
        target=publication/config['results_relative']
        for path in result.rglob('*'):
            if path.is_file():stable_copy(path,target/path.relative_to(result))
        write(Path(config['control'])/'publication-worktree.json',{
            'path':str(publication),'parent':remote,'reason':'Preserve unrelated staged work/current branch and publish after remote advancement',
            'original_checkout_unchanged':True})
    git_at(publication,'add','--',config['results_relative']);git_at(publication,'diff','--cached','--check')
    if git_at(publication,'diff','--cached','--name-only'):
        git_at(publication,'commit','-m','Save fixed v8 training and matched local development results','--',config['results_relative'])
    commit=git_at(publication,'rev-parse','HEAD')
    git_at(publication,'push','origin','HEAD:'+branch)
    git_at(publication,'fetch','origin',branch)
    if subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/'+branch],cwd=publication).returncode:
        raise ValueError('Published commit is not contained in the verified remote branch')
    return commit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    config=read(args.config);repo=Path(config['repo']);run=Path(config['run']);control=Path(config['control'])
    check_sources(config,repo);sys.path.insert(0,str(repo/'training/final_round'))
    from run import run_lock
    with run_lock(control/'followup.lock'):
        awake=ctypes.windll.kernel32.SetThreadExecutionState
        if not awake(0x80000001):raise RuntimeError('Followup sleep inhibition failed')
        try:
            while True:
                status=read(run/'status.json')
                if status['status']=='failed':raise RuntimeError('Supplied worker failed in '+status.get('phase','unknown')+'; inspect logs and resume same verified run')
                if status['status']=='complete':break
                write(control/'followup-status.json',{'status':'waiting_for_supplied_worker','worker':status,
                    'heartbeat_UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'new_paid_API_calls':0})
                time.sleep(30)
            result=scientific_outputs(config,repo,run,control)
            write(control/'followup-status.json',{'status':'scientific_results_verified_publication_pending','results':str(result)})
            commit=publish(config,repo,result)
            write(control/'followup-status.json',{'status':'complete_and_pushed','results':str(result),'commit':commit,
                'completed_at_UTC':datetime.now(timezone.utc).isoformat(),'new_paid_API_calls':0})
        except BaseException as error:
            write(control/'followup-status.json',{'status':'stopped_preserved','error_type':type(error).__name__,
                'reason':str(error),'at_UTC':datetime.now(timezone.utc).isoformat(),
                'resume':'Inspect the actual failure; preserve this run and all existing checkpoints. No restart from zero.'})
            raise
        finally:awake(0x80000000)


if __name__=='__main__':main()
