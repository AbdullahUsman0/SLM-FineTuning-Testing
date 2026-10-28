"""Package completed corrective development evidence; no inference or paid calls."""
from datetime import datetime,timezone
import gzip
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import winreg
import zipfile

RUN=Path(__file__).resolve().parent
CONFIG=json.loads((RUN/'audit-config.json').read_bytes())
REPO=Path(CONFIG['repo']);CODE=Path(CONFIG['code'])
sys.path[:0]=[str(RUN/'offline-deps'),str(CODE)]
from controls import atomic,sha
from runner import validate_protocol


def main():
    protocol=validate_protocol(RUN,REPO)
    base=json.loads((RUN/'delivery-verification.json').read_bytes())
    assert json.loads((RUN/'main-completion.json').read_bytes())['status']=='complete'
    replay=json.loads((RUN/'recorded-main-replay-verification.json').read_bytes())
    assert replay['passed'] and replay['script_sha256']==sha(RUN/'verify_recorded_main.py') and replay['scored_records_sha256']==sha(RUN/'main-scored-records.jsonl')
    final=REPO/'research-checkpoints'/(RUN.name+'-paper')
    if final.exists():raise FileExistsError('Preserve earlier packet; choose an explicit new export version')
    shutil.copytree(base['results'],final)
    spec=importlib.util.spec_from_file_location('corrective_analysis',RUN/'analyze_completed.py');analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis);audited=analysis.main(final)
    billing=audited['billing'];summary=json.loads((RUN/'main-summary.json').read_bytes())
    names=['ROOT_CAUSE_AUDIT.md','PAPER_TEST_REQUIREMENTS.md','interface-correction.json','local-technical-gate.json','budget-at-start.json','golden-scorer-audit.json','legacy-probe-summary.json','explicit-probe-summary.json','legacy-probe-freeze.json','explicit-probe-freeze.json','analyze_completed.py','finish_paper_evidence.py','audit-config.json','run-offline-tests.py','adjudication-scope-verification.json','hardware-snapshot.json','runtime-packages.json']
    for name in names:
        if (RUN/name).exists():shutil.copyfile(RUN/name,final/name)
    for name in ['verify_recorded_main.py','recorded-main-replay-verification.json','recorded-main-replay.log','credential_scan.py','delivery_export_recovery.py','export-recovery-authorization.json']:
        if (RUN/name).exists():shutil.copyfile(RUN/name,final/name)
    for name in ['legacy-probe-calls.jsonl','explicit-probe-calls.jsonl','legacy-probe.log','explicit-probe.log','offline-full-tests.log','offline-full-tests-correct-cwd.log']:
        if (RUN/name).exists():(final/(name+'.gz')).write_bytes(gzip.compress((RUN/name).read_bytes(),mtime=0))
    shutil.copyfile(RUN/'archived-v8-replay/summary.json',final/'archived-v8-replay-summary.json')
    prior=Path(CONFIG['prior_run'])
    prior_summary=json.loads((prior/'main-summary.json').read_bytes())
    atomic(final/'failed-native-r2-summary.json',{'scope':'Preserved failed native interface; do not headline as v8 general capability','run':str(prior),'models':prior_summary,'primary_scoring_not_changed':True})
    atomic(final/'paper-test-receipt.json',{'offline_suite':json.loads((RUN/'offline-tests.json').read_bytes()),'archived_996_call_replay_reproduced_original_metrics':True,'golden_scorer_TP':257,'golden_scorer_FP':0,'golden_scorer_FN':0,'fresh_local_probe_calls':18,'matched_pilot_calls':36,'matched_main_calls':404,'same_main_case_hash_as_failed_run':protocol['main_case_sha256']==json.loads((prior/'protocol.json').read_bytes())['main_case_sha256'],'same_primary_scoring_AST_test_passed':True,'all_final_source_hashes_verified':True,'independent_human_review_complete':False,'sealed_final_accessed':False})
    table=['| System | Track | P | R | Exact F1 | JSON | Schema | Raw admitted | Pipeline exact turns | Median s |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['v8','openai']:
        for track,m in summary[arm].items():
            e=m['raw_exact'];table.append(f"| {arm} | {track} | {e['precision']:.4f} | {e['recall']:.4f} | {e['f1']:.4f} | {m['json_valid']}/101 | {m['schema_valid']}/101 | {m['raw_contract_admitted']}/101 | {m['exact_pipeline_state_turns']}/101 | {m['median_seconds']:.3f} |")
    table='\n'.join(table)
    narrative='''# Corrective r3 development comparison and paper handoff

The original native-interface zero run is preserved as an integration-failure diagnostic. Its prompt omitted the trained update-entry contract and typed slot metadata; only OpenAI received a machine-readable output schema. The pilot admitted only two of nine v8 component responses, yet the first runner lacked a successful positive-extraction gate before main inference. This was an evaluation setup/gating mistake, not proof that v8 lacks semantic extraction capability.

The corrective common adapter retains native current application context, exposes the SAME complete response schema and typed definitions to BOTH models, and explicitly supplies the unchanged annotation/value contract. Local greedy decoding versus hosted native Structured Outputs still differs and is disclosed. Fresh local positive controls and the full offline suite passed before the corrective main run. No output repair, label-dependent case selection, scorer relaxation, adapter change or retraining was used.

This is a post-hoc corrective DEVELOPMENT revision informed by failed integration, not an independent confirmatory test. The same 20 main conversations and raw gold are retained. The previously user-approved invalid state for column `true` is unchanged. Raw model exact slot/value/status scores precede deterministic recovery/normalization; current application state is reported separately. Fixed-script question context remains None for both arms, and static questions are not fed into later turns. This does not measure adaptive interview quality.

All automated current-application/evaluator checks (303) passed with API credentials removed and outbound networking blocked except loopback. The original 996-call archived v8 replay reproduced F1 0.963953 in both tracks. The known-answer scorer produced 257 TP and zero FP/FN on all 101 main turns. Fresh legacy-interface GPU control on nine disjoint pilot turns reached F1 0.809524; the explicit-contract probe reached F1 1.0, with all nine admitted and 21 TP. These small controls are not main comparison results.

The 20-case main has 101 turns and 257 gold non-intent updates per model/track, 39 positive slot types out of 79, and 16 English/two Roman Urdu/two mixed conversations. Cases were previously inspected public synthetic development and share generation/template structure. No significance/equivalence or universal-superiority conclusion is established. Independent human annotation remains incomplete, and sealed/private final labels were not accessed. Keep the older 31-case OpenAI, 100-case V8-D, failed native r2 and corrective r3 results as separate conditions.

Application recovery can introduce changes even with gold model extraction: the oracle audit records 20 mismatching turns under the declared fixed-script fixtures. Recovery is not raw model credit. Exact state projects non-null values/status/confirmation flags including intent, excluding evidence/confidence/timestamps. Component state uses gold prior; a correct final component snapshot is not a completed successful conversation. Inspect rollout state and the supplementary contract audit for full-conversation conclusions. Extra/forbidden flags do not enumerate every invented fact or wrong value, especially unreadable rejected structures.

No further model calls are authorized by this handoff. The authoritative task budget is shared across old and corrected revisions; changing credentials/revision does not renew it. The exported SQLite snapshot is analysis evidence, not a fresh dispatch ledger on another PC. No model weights, optimizer state, credentials, .env files or caches are included. The same verified v8 weights remain in the existing v8-adapter-20261022 release.
'''
    replay=json.loads((RUN/'recorded-main-replay-verification.json').read_bytes())
    assert replay['passed'] and replay['raw_responses_rescored']==replay['current_pipeline_states_replayed']==404
    test_receipt=json.loads((final/'paper-test-receipt.json').read_bytes());test_receipt['post_run_offline_replay']=replay;atomic(final/'paper-test-receipt.json',test_receipt)
    narrative+='\nAll 404 main raw responses were rescored offline, and all 404 current-pipeline states and component/rollout inputs replayed exactly from the saved journals. Outbound network access was blocked except loopback; no additional model calls or ledger mutations were used. See recorded-main-replay-verification.json and its saved script/log.\n'
    cost='\n\nWhole-task usage-rate estimate: $'+billing['cumulative_task_usage_rate_estimate_USD']+'. Whole-task conservative ledger accounting: $'+billing['cumulative_task_conservative_ledger_USD']+'. Corrective revision alone: $'+billing['usage_rate_estimate_USD']+' usage estimate. All previous spending is included; reserved/uncertain $0; user cap $1, internal ceiling $0.90. Usage-based estimates are not a provider invoice.\n'
    (final/'README.md').write_text(narrative+'\n\n'+table+cost,encoding='utf-8')
    (final/'PAPER_UPDATE_PROPOSAL.md').write_text('# Proposed additions to the existing paper\n\n'+table+'\n\nAdd a separate corrective-development cohort, with the methods and limitations in README.md. Disclose the failed native-interface configuration and the diagnostic controls; do not erase it or replace historical scores. Use PAPER_UPDATE_TABLE.csv for complete corrections/unknown/recovery/state/latency counts and billing-usage-analysis.json for per-revision/cumulative cost. Do not fabricate or replace the paper source, claim independent human review, treat component final snapshots as whole-conversation success, or claim a universal winner from these point estimates.\n'+cost,encoding='utf-8')
    completion=json.loads((final/'completion.json').read_bytes());completion.update(status='complete_corrective_development_verified',posthoc_corrective_revision=True,previous_failed_configuration_preserved=True,current_revision_hosted_requests=220,cumulative_task_hosted_requests=billing['cumulative_task_hosted_requests'],cumulative_task_usage_estimate_USD=billing['cumulative_task_usage_rate_estimate_USD'],cumulative_task_guard_accounted_USD=billing['cumulative_task_conservative_ledger_USD'],reserved_USD=0,uncertain_USD=0,independent_human_review_complete=False)
    atomic(final/'completion.json',completion)
    handoff='''# Local Codex: update the existing FYP paper from the corrected evidence

Repository: AbdullahUsman0/SLM-FineTuning-Testing
Branch: experiment/v5-grounded-20260919

Preserve local edits; fetch/fast-forward safely or use a new checkout. Read the publication folder containing this handoff, then the corrective result README, completion.json, protocol.json, ROOT_CAUSE_AUDIT.md, PAPER_TEST_REQUIREMENTS.md, PAPER_UPDATE_TABLE.csv, supplementary-contract-audit.json, billing-usage-analysis.json and automated failure findings. The raw journals and full SDK responses/usage are archived with checksums.

The previous 0/0/0 v8 result came from a mismatched native interface and inadequate pilot gate. Preserve it as an integration-failure condition; do not headline it as model capability. The archived 100-case v8 score reproduces; fresh legacy/explicit local probes establish correct positive extraction. The new r3 main gives both models the same complete visible output schema, typed slots and unchanged annotation contract while retaining the current fpy snapshot plus bundled fixes. It is a corrective inspected-development revision, not independent validation. No scorer relaxation, model substitution or raw-gold change was used.

Update the ACTUAL existing paper draft with backup. Keep earlier N/M/P/V8-D and historical hosted results separate, and add this corrective cohort with precise raw-versus-pipeline definitions. Record the setup mistake, controls, exact matched cases, source hashes, model/decoder differences, unsupported/recovery failures and honest generalization limits. Human annotation remains incomplete; do not claim a universal winner, semantic equivalence or deployment readiness. Use main results only; pilots/probes are excluded from the main denominator. Do not invent a replacement manuscript on this server.

The same pinned Qwen/Qwen3.5-2B base revision 15852e8c16360a2fea060d615a32b45270f8a8fc and final-step v8 adapter are used. Weight SHA-256 c83e0f1a458125970eb70ff360ebead02618f2f48b5bbb59e2ca91a5b7c4c0df; config SHA-256 a1ce14418c5c22375ccfd773338ef1b09117ec9b9e72db034c71ade886b4664c. Download the adapter from the existing verified v8-adapter-20261022 prerelease if needed; do not assume GPU-server D: paths exist on the other PC. This results ZIP excludes weights and credentials.

No new training, paid requests, sealed/private labels or automatic study expansion is authorized by this handoff. The existing shared $1 ledger covers both failed and corrected runs and must never be reset. For offline reproduction, use the source bundle, exact fpy revision/patch and evaluator/source hashes in protocol.json; set CURRENT_PIPELINE_RUN and PYTHONPATH as recorded in offline-tests.json and use its exact command. Do not rerun completed paid stages. The raw/scored journals permit read-only audit without inference.
'''
    (final/'LOCAL_CODEX_HANDOFF.md').write_text(handoff+'\n\n'+table+cost,encoding='utf-8')
    root=REPO/'research-checkpoints/current-pipeline-v8-openai-paper-audit-20261027'
    root.mkdir(exist_ok=False)
    shutil.copyfile(final/'LOCAL_CODEX_HANDOFF.md',root/'LOCAL_CODEX_HANDOFF.md')
    shutil.copyfile(RUN/'ROOT_CAUSE_AUDIT.md',root/'ROOT_CAUSE_AUDIT.md')
    (root/'README.md').write_text('# Start here: audited and corrected v8/OpenAI comparison\n\nRead LOCAL_CODEX_HANDOFF.md and ROOT_CAUSE_AUDIT.md first. Corrective result folder: ../'+final.name+'/ . The prior failed native-interface run is retained at ../'+Path(CONFIG['prior_run']).name+'-final/ ; its zero score is not a reliable model-capability headline. The corrective main results are below; both conditions remain public for transparency.\n\n'+table+cost+'\nIndependent human review remains incomplete.\n',encoding='utf-8')
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as k:secret=winreg.QueryValueEx(k,'OPENAI_API_KEY')[0].encode()
    from credential_scan import scan_payload
    scan_exceptions=[]
    for path in final.rglob('*'):
        if not path.is_file():continue
        b=path.read_bytes();b=gzip.decompress(b) if path.suffix=='.gz' else b
        exception=scan_payload(b,secret)
        if exception:scan_exceptions.append({'path':path.relative_to(final).as_posix(),**exception})
        if path.name.startswith('.env'):raise ValueError('Environment file in evidence')
    atomic(final/'credential-scan-exceptions.json',{'policy':'Exact-file SHA and exact opaque reasoning field only; all configured-key matches prohibited; raw bytes unchanged','exceptions':scan_exceptions})
    hashes={p.relative_to(final).as_posix():sha(p) for p in final.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'}
    atomic(final/'artifact-manifest.json',{'files_sha256':hashes,'credential_scan_passed':True,'weights_optimizer_caches_included':False});hashes['artifact-manifest.json']=sha(final/'artifact-manifest.json')
    archive=root/'corrected-r3-results.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(hashes):z.write(final/name,name)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps({'files_sha256':hashes},sort_keys=True,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,wanted in hashes.items():assert __import__('hashlib').sha256(z.read(name)).hexdigest()==wanted
    checksum=sha(archive);archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n',encoding='ascii')
    destination=Path.home()/'Downloads'/(RUN.name+'-paper-results.zip');shutil.copyfile(archive,destination);assert sha(destination)==checksum
    destination.with_suffix('.zip.sha256').write_text(checksum+'  '+destination.name+'\n',encoding='ascii')
    roundtrip=RUN/'paper-delivery-roundtrip';roundtrip.mkdir()
    with zipfile.ZipFile(destination) as z:
        for name in z.namelist():assert (roundtrip/name).resolve().is_relative_to(roundtrip.resolve())
        z.extractall(roundtrip)
    for name,wanted in hashes.items():assert sha(roundtrip/name)==wanted
    with (root/'LOCAL_CODEX_HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\nPortable ZIP: corrected-r3-results.zip\nSHA-256: '+checksum+'\nThe ZIP internal handoff references the adjacent checksum; this standalone handoff includes the archive checksum.\n')
    atomic(root/'DELIVERY.json',{'ZIP':'corrected-r3-results.zip','sha256':checksum,'bytes':archive.stat().st_size,'CRC_member_hashes_copy_roundtrip_verified':True,'credential_scan_passed':True,'result_folder':'../'+final.name,'prior_failed_configuration_preserved':True})
    receipt={'status':'complete_ready_to_push','results':str(final),'publication_root':str(root),'ZIP':str(destination),'Git_ZIP':str(archive),'sha256':checksum,'handoff':str(root/'LOCAL_CODEX_HANDOFF.md'),'main_calls':404,'pilot_calls':36,'cumulative_task_hosted_requests':billing['cumulative_task_hosted_requests'],'cumulative_guard_USD':billing['cumulative_task_conservative_ledger_USD'],'cumulative_usage_estimate_USD':billing['cumulative_task_usage_rate_estimate_USD'],'published_to_GitHub':False}
    atomic(RUN/'PAPER_DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
