"""Verify and package the bounded R4 rerun; never generates model responses."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import winreg
import zipfile

RUN=Path(__file__).resolve().parent
C=json.loads((RUN/'config.json').read_bytes());REPO=Path(C['repo']);CODE=Path(C['code'])
sys.path[:0]=[str(RUN/'offline-deps'),str(CODE)]
from controls import AMENDMENT,atomic,digest,sha
from runner import validate_protocol
import pipeline as p


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    protocol=validate_protocol(RUN,REPO)
    assert json.loads((RUN/'main-completion.json').read_bytes())['status']=='complete'
    assert json.loads((RUN/'pilot-completion.json').read_bytes())['status']=='complete'
    with (RUN/'recorded-main-replay.log').open('wb') as log:
        subprocess.run([sys.executable,'-B',str(RUN/'verify_recorded_main.py')],cwd=RUN,stdout=log,stderr=subprocess.STDOUT,check=True)
    replay=json.loads((RUN/'recorded-main-replay-verification.json').read_bytes());assert replay['passed'] and replay['raw_responses_rescored']==404
    result=REPO/'research-checkpoints'/RUN.name
    root=REPO/'research-checkpoints/current-pipeline-v8-openai-r4-20261028'
    assert not result.exists() and not root.exists(),'Preserve earlier exports; use a new export version'
    result.mkdir();root.mkdir()
    # Copy only declared study evidence, excluding caches, keys and model weights.
    names=['config.json','protocol.json','selection.json','main-cases.json','pilot-cases.json','adjudication-authorization.json','adjudication-changes.json','adjudication-scope-verification.json','label-audit.json','bundle-verification.json','historical-openai-audit.json','local-runtime.json','interface-correction.json','local-technical-gate.json','slm-reuse-verification.json','budget-at-start.json','budget-authorization-amendment.json','budget-amendment-installation.json','budget-before-amendment.sqlite3','server-auth-latest.json','offline-tests.json','offline-tests-first-attempt.json','pilot-summary.json','main-summary.json','pilot-completion.json','main-completion.json','background-launch.json','supervisor.py','supervisor-source-freeze.json','recorded-main-replay-verification.json','verify_recorded_main.py','analyze_completed.py','complete_and_package.py','run-offline-tests.py','pricing-recheck-20261028.json','renderer-resume-verification.json','renderer-only-source-audit.json','prepare_continuation.py']
    for name in names:
        if (RUN/name).exists():shutil.copyfile(RUN/name,result/name)
    for path in RUN.iterdir():
        if path.is_file() and path.suffix in {'.jsonl','.log'}:
            (result/(path.name+'.gz')).write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    shutil.copytree(CODE,result/'evaluator',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(RUN/'hosted-responses',result/'hosted-responses')
    shutil.copytree(Path(C['prior_run'])/'bundle',result/'bundle')
    blocked=Path(C['resume_from']);blocked_result=result/'previous-blocked-r4';blocked_result.mkdir()
    for name in ['config.json','protocol.json','status.json','supervisor-status.json','offline-tests.json','offline-tests-first-attempt.json']:
        if (blocked/name).exists():shutil.copyfile(blocked/name,blocked_result/name)
    for path in blocked.glob('*.log'):(blocked_result/(path.name+'.gz')).write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    audited=load_module('r4_analysis',RUN/'analyze_completed.py').main(result)
    billing=audited['billing'];summary=json.loads((RUN/'main-summary.json').read_bytes())
    db=sqlite3.connect('file:'+Path(C['shared_ledger']).as_posix()+'?mode=ro',uri=True)
    old_rows=[]
    all_rows=db.execute('SELECT key,context,bound,actual,state,response_sha,usage FROM calls ORDER BY key').fetchall()
    for row in all_rows:
        key=json.loads(row[0])
        if len(key)!=6 or key[0]!=protocol['dispatch_namespace']:old_rows.append(row)
    assert len(old_rows)==440 and len(all_rows)==660 and all(r[4]=='reconciled' for r in all_rows)
    assert digest(old_rows)==json.loads((RUN/'budget-at-start.json').read_bytes())['old_call_rows_sha256']
    original=sqlite3.connect(RUN/'budget-before-amendment.sqlite3')
    assert db.execute('SELECT * FROM authorization').fetchall()==original.execute('SELECT * FROM authorization').fetchall();original.close()
    assert json.loads(db.execute('SELECT config FROM budget_amendments WHERE id=?',(AMENDMENT['id'],)).fetchone()[0])==AMENDMENT
    backup=sqlite3.connect(result/'budget-ledger-snapshot.sqlite3');db.backup(backup);assert backup.execute('PRAGMA integrity_check').fetchone()[0]=='ok';backup.close();db.close()
    atomic(result/'budget-ledger-export.json',{'authorization_amendment':AMENDMENT,'columns':['key','context','bound','actual','state','response_sha','usage'],'calls':all_rows,'prior_440_rows_unchanged':True,'all_reconciled':True})
    records=[json.loads(line) for line in (RUN/'main-scored-records.jsonl').read_bytes().splitlines()]
    cases=json.loads((RUN/'main-cases.json').read_bytes());gold={(c['scenario_id'],t['turn_id']):(c,t) for c in cases for t in c['turns']}
    review=[]
    for case_turn,(case,turn) in gold.items():
        matched=[r for r in records if tuple(r['key'][3:])==case_turn]
        review.append({'scenario_id':case_turn[0],'turn_id':case_turn[1],'message':turn['message'],'supplied_raw_gold':turn['gold_extraction'],'supplied_expected_state':turn['expected_state_after'],'responses':[{'model':r['key'][1],'track':r['key'][2],'raw_output':r['raw_output'],'native_admitted':r['raw_contract_admitted'],'confusion':r['confusion'],'pipeline_state':r['pipeline_state']} for r in matched],'independent_review_status':'pending'})
    atomic(result/'paired-human-review-queue.json',{'complete':False,'scope':'All 101 public development turns, both models/tracks; prepared for future review, not adjudicated and not a blinded annotation study','items':review})
    for case_turn in gold:
        component=[r for r in records if r['key'][2]=='component' and tuple(r['key'][3:])==case_turn]
        assert len(component)==2 and component[0]['identity']==component[1]['identity']
    per_slot={};failures=[]
    for row in records:
        case,turn=gold[tuple(row['key'][3:])]
        expected=Counter(p.tuples(turn['gold_extraction']))
        emitted=Counter(p.tuples(json.loads(row['raw_output']))) if row['json_valid'] else Counter()
        correct=expected&emitted if row['raw_contract_admitted'] else Counter()
        for sid in {t[0] for t in expected}|{t[0] for t in emitted}:
            key='/'.join(row['key'][1:3])+'/'+sid;stat=per_slot.setdefault(key,{'tp':0,'fp':0,'fn':0,'support':0})
            tp=sum(v for t,v in correct.items() if t[0]==sid);g=sum(v for t,v in expected.items() if t[0]==sid);n=sum(v for t,v in emitted.items() if t[0]==sid)
            stat['tp']+=tp;stat['fp']+=n-tp;stat['fn']+=g-tp;stat['support']+=g
        flags=[name for name in ['json_valid','schema_valid','raw_contract_admitted','reducer_admitted','complete_pipeline_state'] if not row[name]]
        if row['extra_slot_flags']:flags.append('extra_actionable_slots')
        if row['forbidden_slot_flags']:flags.append('forbidden_actionable_slots')
        if row['duplicate_slots']:flags.append('duplicate_slot_entries')
        if row['gold_correction'] and not row['correct_correction']:flags.append('incorrect_correction_updates')
        if flags:failures.append({'key':row['key'],'flags':flags,'confusion':row['confusion'],'domain':row['domain'],'language':row['language'],'recovered_slots':row['provenance']['recovered_slot_ids']})
    atomic(result/'per-slot-results.json',per_slot);atomic(result/'failure-findings.json',failures)
    table=['| Model | Track | Precision | Recall | Exact actionable F1 | Native admitted | Exact pipeline turns | Median seconds |','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('v8','openai'):
        for track in ('component','rollout'):
            m=summary[arm][track];e=m['raw_exact']
            table.append(f"| {arm} | {track} | {e['precision']:.6f} | {e['recall']:.6f} | {e['f1']:.6f} | {m['raw_contract_admitted']}/101 | {m['exact_pipeline_state_turns']}/101 | {m['median_seconds']:.3f} |")
    table='\n'.join(table)
    text='''# R4 native-admission actionable-update comparison

Completed matched corrective DEVELOPMENT comparison: 20 frozen conversations,
101 turns per model/track, 404 main scored calls. OpenAI main 202/202 calls are
FRESH. SLM main 202/202 responses are REUSED from R3 after proving exact identity
of all component/rollout inputs and application trajectories under native parsing.
Both models are rescored under the same NEW R4 rules. The separate nine-turn
pilot contributes 18 fresh OpenAI and 18 reused SLM calls and is excluded from
main metrics. Do not describe both arms as fresh inference.

The cohort retains one conversation per 20 domains: six weather, twelve economic
and two combined domains; sixteen English, two Roman Urdu and two mixed cases.
There are 257 gold actionable non-intent updates per track and positive support
for 39 of the 79 slot types. These are not all-domain/all-slot accuracy claims.

The common prompt/context and gold remain identical to R3. Both arms now use
the native Pydantic candidate decoder once, including its ordinary-string fallback.
Ignored statuses unmentioned/dont_care are excluded from actionable predictions
for both arms, with their counts and literal empty-list output reported separately.
Native admission, JSON/schema validity, duplicate entries, evidence/reducer
admission and resulting state are distinct. Duplicate actionable updates retain
multiset penalties. No paraphrase/synonym repair or numeric semantic normalization
was introduced. Exact slot/value/status F1 is not general semantic accuracy.

Raw scoring precedes reducer normalization and deterministic recovery. Component
uses supplied gold prior; rollout uses its own accepted state except the explicitly
declared synthetic interventions. Current fpy plus the verified bundled fixes is
used, with the unchanged common augmented extraction adapter. Static generated
questions are not fed into later scripted turns. This is not an unmodified native
provider-prompt or adaptive-interview benchmark. The oracle label audit still
records application recovery mismatches even when given gold extraction.

All frozen source/case/adapter hashes were checked. The 299-test suite passed;
the first test collection attempt lacked explicit fpy import roots and is preserved
beside the successful command/log. All 404 raw scores, inputs and pipeline states
were replayed offline after completion. All 220 fresh hosted SDK responses and
usage receipts match the durable ledger, and all 440 prior call rows and the
original authorization row remain unchanged. The additional-dollar approval is
an append-only amendment, not a spending reset. No automatic paid retries.

A versioned R4.1 continuation handles only a local static-question renderer
length validation error. R4 stopped after 132 paid calls; all were copied and
replayed with exactly identical inputs, under the SAME paid-request namespace,
then only the remaining 88 were dispatched. Native extraction/admission, scoring,
prompts, gold and decoding did not change. The original blocked run and source
are preserved; renderer failures are separately recorded and do not repair model
output. This is a fixed-script evaluation wrapper correction, not a production
application fix.

This is inspected, related synthetic development, not independent confirmation.
The 20 cases share earlier public templates; independent human gold/disagreement
review remains INCOMPLETE. No significance, equivalence, general superiority,
deployment readiness or paper-wide validation claim follows from these estimates.
Do not combine this hosted row with SLM scores from another scoring protocol.
No new training, model substitution, sealed/private-label access or paid judge.
No model weights, caches, optimizer state or credentials are included.
'''
    cost=f"\nR4 usage-rate cost estimate: ${billing['usage_rate_estimate_USD']}; cumulative task usage estimate: ${billing['cumulative_task_usage_rate_estimate_USD']}; cumulative conservative ledger accounting: ${billing['cumulative_task_conservative_ledger_USD']}. Reserved/uncertain $0. Rates include reasoning once; cached input affects the estimate but never dispatch permission. This is not a provider invoice. Total authorization $2; task dispatch ceiling $1.90; this rerun's new-spend ceiling $0.90.\n"
    (result/'README.md').write_text(text+'\n'+table+'\n'+cost,encoding='utf-8')
    completion={'status':'complete_matched_corrective_development','main_scored_calls':404,'planned_main_scored_calls':404,'fresh_hosted_main_calls':202,'reused_SLM_main_calls':202,'fresh_hosted_pilot_calls':18,'reused_SLM_pilot_calls':18,'component_inputs_identical':True,'SLM_inputs_and_state_trajectories_verified_unchanged':True,'native_admission_applied_both_models':True,'all_404_raw_scores_and_states_replayed':True,'offline_tests':299,'prior_440_ledger_rows_unchanged':True,'task_conservative_cost_USD':billing['cumulative_task_conservative_ledger_USD'],'task_usage_cost_estimate_USD':billing['cumulative_task_usage_rate_estimate_USD'],'reserved_USD':0,'uncertain_USD':0,'independent_human_review_complete':False,'universal_winner_established':False,'final_labels_accessed':False,'protocol_sha256':sha(RUN/'protocol.json'),'sealed_before_publication':True}
    atomic(result/'completion.json',completion)
    handoff='''# Local Codex: add the R4 matched development comparison

Safely pull experiment/v5-grounded-20260919 in AbdullahUsman0/SLM-FineTuning-Testing;
preserve local paper edits. Read this publication README, completion.json,
protocol.json, interface-correction.json, slm-reuse-verification.json,
PAPER_UPDATE_TABLE.csv, supplementary-contract-audit.json, failure-findings.json,
billing-usage-analysis.json and offline/replay receipts before updating the actual
existing paper draft. Keep a backup; do not create a replacement manuscript.

Use BOTH R4 rows, never a new hosted row alongside an old strict-scoring SLM row.
OpenAI is fresh; SLM responses were previously generated and were reused only after
verifying identical inputs and native-parser state trajectories. Both were rescored
with the same native admission/actionable-update rules. Report exact precision,
recall/F1, structure/admission, corrections/unknowns, extra actionable slots,
recovery/state success and latency with their denominators. Keep raw extraction
separate from deterministic state recovery and describe the common augmented
adapter and fixed scripted question context. The older R2/R3 conditions remain
historical and must not be erased or called fresh R4 evidence.

This supports a bounded direct matched DEVELOPMENT comparison of configured
systems. It does not establish independent/general model superiority, an
unmodified-production-provider benchmark, real-user adaptive conversation quality,
deployment readiness or independent human annotation. Keep related templates,
previously inspected cases, no confidence intervals, decoder differences and
incomplete human review explicit. Do not infer a new high accuracy from a parser
acceptance rate. Use the measured results regardless of which model scores higher.

The original shared $1 authorization and 440 paid calls remain intact; a separately
recorded user-approved additional $1 raised total authorization to $2. The exported
ledger is evidence, not a new spending account. No further paid calls, training,
sealed labels or automatic expansion are authorized by this completed handoff.

Exact model: gpt-5-mini-2025-08-07. SLM: pinned Qwen/Qwen3.5-2B base and final-step
v8 adapter hashes in protocol.json, with original RTX A4000 CUDA BF16 runtime in
local-runtime.json. Existing adapter release v8-adapter-20261022 remains separate;
this ZIP contains results/source only. Do not assume server D: paths exist locally.

Verify the adjacent ZIP checksum and all ARCHIVE_MANIFEST.json member hashes.
For offline reproduction, restore the exact fpy commit plus verified bundle patch,
use the frozen evaluator/source hashes and offline-tests.json command/environment.
verify_recorded_main.py reproduces the completed raw/state evidence without model
calls in the original run layout. Do not replay completed paid stages or reset
the ledger; source/layout changes on another PC require a documented reproduction.
'''
    (result/'LOCAL_CODEX_HANDOFF.md').write_text(handoff+'\n'+table+'\n'+cost,encoding='utf-8')
    (result/'PAPER_UPDATE_PROPOSAL.md').write_text('Add a separate R4 configured-system matched development cohort using the methods and limitations in README.md. Retain older cohorts and the integration-failure audit. Human review remains incomplete.\n\n'+table+'\n'+cost,encoding='utf-8')
    # Exact configured-key matching applies to every byte, including opaque fields.
    # Generic token patterns cannot meaningfully scan random encrypted reasoning;
    # exclude only that provider field from the pattern view and log its SHA.
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:secret=winreg.QueryValueEx(key,'OPENAI_API_KEY')[0].encode()
    pattern=re.compile(rb'(?i)(?<![A-Za-z0-9])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,})')
    opaque=[]
    for path in result.rglob('*'):
        if not path.is_file():continue
        assert not path.name.startswith('.env') and path.suffix not in {'.safetensors','.pt','.bin','.pyc'}
        body=path.read_bytes();body=gzip.decompress(body) if path.suffix=='.gz' else body
        assert secret not in body,'Credential found; content withheld'
        view=body
        if path.parent.name=='hosted-responses' and path.suffix=='.json' and not path.name.endswith('.request.json'):
            obj=json.loads(body);response=obj.get('response',{});assert response.get('model')==protocol['hosted_model']
            for index,output in enumerate(response.get('output',[])):
                encrypted=output.get('encrypted_content')
                if output.get('type')=='reasoning' and isinstance(encrypted,str) and encrypted:
                    opaque.append({'file':path.relative_to(result).as_posix(),'file_sha256':sha(path),'field':f'response.output[{index}].encrypted_content','ciphertext_sha256':hashlib.sha256(encrypted.encode()).hexdigest(),'generic_pattern_hit':bool(pattern.search(encrypted.encode())),'raw_bytes_modified':False})
                    output['encrypted_content']='[OPAQUE ENCRYPTED REASONING: EXCLUDED ONLY FROM GENERIC PATTERN VIEW]'
            view=json.dumps(obj,ensure_ascii=False).encode()
        assert not pattern.search(view),'Unadjudicated credential-shaped plaintext; content withheld'
    atomic(result/'credential-scan.json',{'configured_key_absent_in_all_bytes':True,'generic_plaintext_pattern_scan_passed':True,'opaque_reasoning_field_exclusions':opaque,'raw_responses_modified':False,'credentials_environment_weights_caches_included':False})
    hashes={path.relative_to(result).as_posix():sha(path) for path in result.rglob('*') if path.is_file()}
    atomic(result/'artifact-manifest.json',{'files_sha256':hashes,'credential_scan_passed':True});hashes['artifact-manifest.json']=sha(result/'artifact-manifest.json')
    archive=root/'r4-native-comparison-results.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(hashes):z.write(result/name,name)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps({'files_sha256':hashes},indent=2,sort_keys=True))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,wanted in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==wanted
    checksum=sha(archive);archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n',encoding='ascii')
    destination=Path.home()/'Downloads'/(RUN.name+'-results.zip');assert not destination.exists();shutil.copyfile(archive,destination);assert sha(destination)==checksum
    destination.with_suffix('.zip.sha256').write_text(checksum+'  '+destination.name+'\n',encoding='ascii')
    roundtrip=RUN/'delivery-roundtrip';roundtrip.mkdir()
    with zipfile.ZipFile(destination) as z:
        for name in z.namelist():assert (roundtrip/name).resolve().is_relative_to(roundtrip.resolve())
        z.extractall(roundtrip)
    for name,wanted in hashes.items():assert sha(roundtrip/name)==wanted
    shutil.copyfile(result/'LOCAL_CODEX_HANDOFF.md',root/'LOCAL_CODEX_HANDOFF.md')
    with (root/'LOCAL_CODEX_HANDOFF.md').open('a',encoding='utf-8') as stream:stream.write('\nPortable results ZIP: '+archive.name+'\nSHA-256: '+checksum+'\n')
    (root/'README.md').write_text(text+'\nResult folder: ../'+result.name+'\n\n'+table+'\n'+cost+'\nZIP SHA-256: '+checksum+'\n',encoding='utf-8')
    receipt={'status':'verified_ready_to_publish','results':str(result),'publication_root':str(root),'ZIP':str(archive),'Downloads_ZIP':str(destination),'sha256':checksum,'bytes':archive.stat().st_size,'archive_members':len(hashes)+1,'all_CRC_member_SHA_copy_extraction_roundtrip_verified':True,'credential_scan_passed':True,'fresh_hosted_calls':220,'reused_SLM_calls':220,'task_conservative_USD':billing['cumulative_task_conservative_ledger_USD'],'new_usage_estimate_USD':billing['usage_rate_estimate_USD'],'task_usage_estimate_USD':billing['cumulative_task_usage_rate_estimate_USD'],'reserved_USD':0,'uncertain_USD':0,'independent_human_review_complete':False}
    atomic(root/'DELIVERY.json',receipt);atomic(RUN/'DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
