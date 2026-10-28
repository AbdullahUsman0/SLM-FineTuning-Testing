"""Verify completed matched evidence and create a portable, credential-checked ZIP."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import zipfile


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deliver(run,repo,code,secret,scope='complete'):
    import sys
    sys.path.insert(0,str(code))
    from controls import Ledger,dumps
    import pipeline as p
    run=Path(run);repo=Path(repo);code=Path(code)
    protocol=json.loads((run/'protocol.json').read_bytes())
    ledger=Ledger(protocol['shared_budget_ledger']);balance=ledger.summary()
    if balance['reserved_or_uncertain_USD']:
        scope='blocked_uncertain_dispatch'
    cases=json.loads((run/'main-cases.json').read_bytes())
    gold={(c['scenario_id'],t['turn_id']):(c,t) for c in cases for t in c['turns']}
    records=[]
    if (run/'main-scored-records.jsonl').exists():
        records=[json.loads(line) for line in (run/'main-scored-records.jsonl').read_bytes().splitlines()]
    expected=4*sum(len(c['turns']) for c in cases)
    if scope=='complete':
        assert len(records)==expected==404
        assert len({tuple(r['key']) for r in records})==404
        keysets={}
        for arm in ('v8','openai'):
            for track in ('component','rollout'):
                rows=[r for r in records if r['key'][1:3]==[arm,track]]
                assert len(rows)==101
                keysets[arm,track]={tuple(r['key'][3:]):r for r in rows}
                assert set(keysets[arm,track])==set(gold)
        for key in gold:
            assert keysets['v8','component'][key]['identity']==keysets['openai','component'][key]['identity'],'Component input/gold mismatch'
        main_summary=json.loads((run/'main-summary.json').read_bytes())
        for arm in ('v8','openai'):
            for track in ('component','rollout'):
                assert p.prf(list(keysets[arm,track].values()))==main_summary[arm][track]['raw_exact']
        main_receipt=json.loads((run/'main-completion.json').read_bytes())
        assert main_receipt['status']=='complete' and main_receipt['protocol_sha256']==sha(run/'protocol.json')
    else:
        main_summary={}
    result=repo/'research-checkpoints'/(run.name+'-verified-export-r1')
    if result.exists():raise FileExistsError('Preserve prior delivery; use a new delivery version')
    result.mkdir(parents=True)
    for name in ['protocol.json','selection.json','main-cases.json','pilot-cases.json','adjudication-authorization.json','adjudication-changes.json','annotation-review-queue.json','label-audit.json','offline-tests.json','offline-tests.log','historical-openai-audit.json','server-auth-latest.json','credential-sources.json','bundle-verification.json','server-auth-offline-tests.log','budget-reconciliation.json','local-runtime.json','pilot-summary.json','main-summary.json','pilot-completion.json','main-completion.json','calls.jsonl','pilot-scored-records.jsonl','main-scored-records.jsonl','infrastructure-errors.jsonl','status.json']:
        source=run/name
        if source.exists():
            if name.endswith('.jsonl') or name.endswith('.log'):
                (result/(name+'.gz')).write_bytes(gzip.compress(source.read_bytes(),mtime=0))
            else:shutil.copyfile(source,result/name)
    for folder in ['hosted-responses','bundle']:
        if (run/folder).exists():shutil.copytree(run/folder,result/folder)
    shutil.copytree(code,result/'evaluator',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['supervisor.py','delivery.py','supervisor-config.json']:
        if (run/name).exists():shutil.copyfile(run/name,result/name)
    for path in run.glob('local-smoke-*.json'):shutil.copyfile(path,result/path.name)
    for path in run.glob('*worker*.log'):
        (result/(path.name+'.gz')).write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    write(result/'budget-ledger-export.json',balance)
    backup=sqlite3.connect(result/'budget-ledger-snapshot.sqlite3')
    ledger.db.backup(backup);assert backup.execute('PRAGMA integrity_check').fetchone()[0]=='ok';backup.close();ledger.db.close()
    failures=[];slots={};groups={}
    for r in records:
        c,t=gold[tuple(r['key'][3:])]
        flags=[]
        for field in ['json_valid','schema_valid','raw_contract_admitted','reducer_admitted','complete_pipeline_state']:
            if not r[field]:flags.append(field+'=false')
        if r.get('capped'):flags.append('capped')
        if r.get('refused'):flags.append('refused')
        if r['extra_slot_flags']:flags.append('extra_slots')
        if r['forbidden_slot_flags']:flags.append('forbidden_slots')
        if r['gold_correction'] and not r['correct_correction']:flags.append('raw_correction_failure')
        if flags:failures.append({'key':r['key'],'flags':flags,'domain':r['domain'],'language':r['language'],'recovered_slot_ids':r['provenance']['recovered_slot_ids']})
        expected_tuples=Counter(p.tuples(t['gold_extraction']));pred=Counter(p.tuples(json.loads(r['raw_output']))) if r['json_valid'] else Counter()
        match=expected_tuples&pred if r['raw_contract_admitted'] else Counter()
        for sid in {v[0] for v in expected_tuples}|{v[0] for v in pred}:
            key='/'.join(r['key'][1:3])+'/'+sid
            row=slots.setdefault(key,{'tp':0,'fp':0,'fn':0,'support':0})
            tp=sum(n for v,n in match.items() if v[0]==sid);g=sum(n for v,n in expected_tuples.items() if v[0]==sid);v=sum(n for t_,n in pred.items() if t_[0]==sid)
            row['tp']+=tp;row['fp']+=v-tp;row['fn']+=g-tp;row['support']+=g
        groups[c['scenario_id']]={'entity_group':c['entity_group'],'cluster_id':c['cluster_id'],'template_variant':c.get('template_variant'),'category':c['category'],'language':c['language'],'domain':c['domain']}
    for row in slots.values():
        tp,fp,fn=row['tp'],row['fp'],row['fn'];row.update(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    write(result/'per-slot-results.json',slots);write(result/'failure-findings.json',failures);write(result/'synthetic-group-structure.json',groups)
    completion={'status':'complete_matched_development' if scope=='complete' else scope,'run':str(run),'main_planned_conversations':20,'main_completed_scored_calls':len(records),'main_planned_calls':404,'calls_per_model_track':101,'pilot_planned_calls_per_model_track':9,'paid_budget':balance,'protocol_sha256':sha(run/'protocol.json'),'independent_human_review_complete':False,'universal_winner_established':False,'sealed_final_accessed':False,'scope':'inspected synthetic current-pipeline development; fixed script; descriptive only','component_contexts_matched':scope=='complete','raw_extraction_distinct_from_recovery':True,'published_to_GitHub':False}
    write(result/'completion.json',completion)
    lines=['# Current-pipeline v8 versus OpenAI development evaluation','',
        'Status: '+completion['status']+'. Independent human review remains incomplete; this is not a final test or a universal winner claim.','',
        'Current fpy: '+protocol['fpy_commit']+' plus patch '+protocol['bundle_patch_sha256']+'. '
        'The same 20 conversation IDs selected before inference were retained. The user approved marking target_column="true" invalid in the application state for labor_employment/0178; supplied prior/expected states carry that status forward. Raw extraction gold is unchanged. The changes and approval are archived.','',
        '| System | Track | Raw exact F1 | Precision | Recall | JSON | Schema | Raw contract | Current reducer | Exact pipeline state, turns | Recovery calls | Median seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in main_summary:
        for track,m in main_summary[arm].items():
            e=m['raw_exact'];n=m['calls'];lines.append(f"| {arm} | {track} | {e['f1']:.6f} | {e['precision']:.6f} | {e['recall']:.6f} | {m['json_valid']}/{n} | {m['schema_valid']}/{n} | {m['raw_contract_admitted']}/{n} | {m['reducer_admitted_without_recovery']}/{n} | {m['exact_pipeline_state_turns']}/{n} | {m['recovery_calls']} | {m['median_seconds']:.3f} |")
    lines+=['',f"Shared task API spending: actual ${balance['actual_USD']:.6f}, reserved/uncertain ${balance['reserved_or_uncertain_USD']:.6f}. User ceiling $1; internal dispatch ceiling $0.90. Usage cost is an application estimate, not an invoice. No budget renewal on key changes or restart.",'',
        'Raw exact scoring uses non-intent slot + decoded candidate value + status, one wire decode, duplicate-preserving multiset matching. Invalid/capped/refused contracts receive zero true positives. This raw score precedes recovery and reducer normalization. Reducer admission means no exception, and does not mean all updates were grounded/accepted or that the resulting state is correct. The current reducer can discard individual unsupported evidence and retain other updates; trace and before-recovery state disclose this. Pipeline exact state compares all non-null slot values, statuses and confirmation flags, including intent, excluding timestamps/evidence/confidence. Counts are turns, not successful whole conversations.','',
        'Deterministic recovery is attributed to the pipeline only. main-summary.json includes correction/unknown, extra/forbidden, language/domain and seeded-intervention exclusion results. Scored records include raw responses, prior/reduced/pipeline states and trace provenance. Extra-slot flags do not count every hallucination or wrong value. per-slot-results.json records exact support and confusion.','',
        'Both systems received current fpy semantic instructions and fixed scripted user messages. Component inputs and labels were checked identical; rollout uses each system\'s own accepted state. No declared assistant question is fed to either system; static generated questions are recorded and removed before the next fixed turn. Current fpy still ranks a pending slot. This is not a real adaptive user interview. The oracle audit identifies deterministic recovery failures even with gold extraction; it is not a model score.','',
        'Local decoding: active final-step v8 LoRA on pinned stock Qwen3.5-2B, CUDA BF16, greedy, thinking disabled, max 1024 new tokens. Hosted: exact gpt-5-mini-2025-08-07, native Structured Outputs, minimal reasoning, max 2048 output tokens, retries disabled. Native schema enforcement and transport differ. Unscored local warmup fixtures and runtime manifests are archived separately. No claim of energy/compute efficiency follows from cross-system latency.','',
        'The 20 cases are an inspected synthetic subset of the earlier public V8-D set, with shared template/entity structure; 16 English, two Roman Urdu, two mixed, one per 20 domains. Support is not all 79 slots. No significance/equivalence analysis is claimed. Do not compare scores numerically with old 31-case OpenAI or 100-case v8 tables. The offline historical arithmetic audit reproduced the hosted scores; unavailable historical parser implementation limits a full admission re-audit. No new calls were used for that audit.','',
        'No retraining, private prospective cohort, sealed final labels, paid judges, paid question generation or automatic study expansion was used. Do not replace the existing paper or fabricate a manuscript absent on this server. Update it only from completed matched receipts and keep earlier cohorts separate.']
    (result/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    handoff=f'''# Local Codex handoff

GPU run: {run}
Research branch: experiment/v5-grounded-20260919; source handoff commit 0ae1e0a1215c061a5a5e4bb02bb99f4ccbc92554.
Result folder: {result}
Completion: {completion['status']}; main {len(records)}/404 scored calls; 20 planned conversations, 101 turns per model/track; pilots are excluded from main scores.
API actual ${balance['actual_USD']:.6f}; outstanding ${balance['reserved_or_uncertain_USD']:.6f}. The existing shared budget was not reset.

Read completion.json, protocol.json, README.md, main-summary.json (if present), adjudication-authorization.json, adjudication-changes.json, failure-findings.json, label-audit.json, historical-openai-audit.json and the compressed raw/scored journals. Preserve existing paper/edit backups. This is new inspected development, independent human review incomplete, not an untouched final test. No universal superiority/equivalence/deployment claim is supported. Raw model scores and pipeline recovery must stay separate. Previous cohorts and archived scores remain separate; the historical hosted arithmetic reproduced.

Base: Qwen/Qwen3.5-2B at 15852e8c16360a2fea060d615a32b45270f8a8fc. The adapter hashes and actual GPU/runtime appear in protocol.json/local-runtime.json. This ZIP includes no weights, optimizer, caches, credentials or .env files. The adapter remains available from the existing v8-adapter-20261022 GitHub prerelease. Do not assume training-PC D: paths exist on the other PC.

Verify the adjacent ZIP checksum and every ARCHIVE_MANIFEST.json member SHA before use. Exact source bundle is in bundle/: run verify_bundle.py. Recreate fpy at {protocol['fpy_commit']} plus its verified bundled patch/working copies in a NEW checkout. Use all source hashes in protocol.json. Do not substitute fpy main or an installed package. Preserve the raw extraction labels and the separately approved state-status adjudication.

For this server, reproduction/resume uses the existing interpreter and SAME ledger:
  D:\\SLM\\FYP-environments\\v8-final-20261018T230255Z\\Scripts\\python.exe -B {code / 'runner.py'} --run {run} --repo {repo} --stage pilot
then --stage main only when the pilot receipt and affordability gate pass. Set PYTHONPATH to {run / 'offline-deps'} and {code}; load the NEW Windows user credential explicitly into the worker; use https://api.openai.com/v1 and clear inherited organization/project overrides for the worker. Do not copy credentials to the other PC or reset the ledger. The frozen source and case hashes must match on resume. A spent/uncertain dispatch must not be replayed. Correct paths for a new machine only under a separately documented reproduction version and budget decision.

Tests: offline-tests.json/log contain the exact command, versions and outcome. CUDA smoke tests and pilot outputs are separate from main. All full SDK responses/usage are in hosted-responses; request JSON includes synthetic prompts/schema, never authorization headers. Do not treat metadata HTTP 200 as proof that generation succeeded; use completed receipts and usage.

Update the existing paper's current-pipeline methods/results/limitations only if completion.json indicates a complete matched study. Otherwise document the precise blocker/partial coverage and leave unmeasured results blank. Return edited paper paths and a traceable results table; do not invent a replacement manuscript. No additional paid evaluation, training or sealed-test access is authorized by this handoff.
'''
    (result/'LOCAL_CODEX_HANDOFF.md').write_text(handoff,encoding='utf-8')
    from credential_scan import scan_payload
    scan_exceptions=[]
    for path in result.rglob('*'):
        if not path.is_file():continue
        if path.name.startswith('.env'):raise ValueError('Environment file in public results')
        body=path.read_bytes();scan=gzip.decompress(body) if path.suffix=='.gz' else body
        exception=scan_payload(scan,secret)
        if exception:scan_exceptions.append({'path':path.relative_to(result).as_posix(),**exception})
    write(result/'credential-scan-exceptions.json',{'policy':'Exact-file/opaque-field false positive only; configured-key matching never exempted; raw bytes preserved','exceptions':scan_exceptions})
    hashes={path.relative_to(result).as_posix():sha(path) for path in result.rglob('*') if path.is_file()}
    write(result/'artifact-manifest.json',{'files_sha256':hashes,'credential_scan_passed':True,'weights_optimizer_caches_included':False})
    hashes['artifact-manifest.json']=sha(result/'artifact-manifest.json')
    archive=run/(run.name+'-export-r1-results.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(hashes):z.write(result/name,name)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps({'files_sha256':hashes},sort_keys=True,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,wanted in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==wanted
    destination=Path.home()/'Downloads'/archive.name
    if destination.exists():raise FileExistsError('Refusing to overwrite Downloads archive')
    shutil.copyfile(archive,destination);assert sha(archive)==sha(destination)
    checksum=sha(archive);destination.with_suffix('.zip.sha256').write_text(checksum+'  '+destination.name+'\n',encoding='ascii')
    roundtrip=run/'delivery-export-r1-roundtrip';roundtrip.mkdir()
    with zipfile.ZipFile(destination) as z:
        for name in z.namelist():assert (roundtrip/name).resolve().is_relative_to(roundtrip.resolve())
        z.extractall(roundtrip)
    for name,wanted in hashes.items():assert sha(roundtrip/name)==wanted
    receipt={'status':'verified_delivery','results':str(result),'ZIP':str(destination),'sha256':checksum,'bytes':destination.stat().st_size,'member_count':len(hashes)+1,'CRC_and_member_SHA256_verified':True,'Downloads_copy_and_extracted_roundtrip_verified':True,'credential_scan_passed':True,'published_to_GitHub':False}
    write(run/'delivery-verification.json',receipt)
    return receipt
