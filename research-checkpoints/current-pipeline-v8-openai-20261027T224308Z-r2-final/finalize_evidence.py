"""Post-inference reporting/export only; never calls a model or changes primary scores."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import winreg
import zipfile

RUN=Path(__file__).resolve().parent
CONFIG=json.loads((RUN/'supervisor-config.json').read_bytes())
REPO=Path(CONFIG['repo']);RESULT=REPO/'research-checkpoints'/(RUN.name+'-final')
sys.path[:0]=[str(RUN/'offline-deps'),CONFIG['code']]
from controls import atomic,sha
from runner import validate_protocol


def main():
    protocol=validate_protocol(RUN,REPO)
    summary=json.loads((RUN/'main-summary.json').read_bytes())
    supplementary=json.loads((RESULT/'supplementary-contract-audit.json').read_bytes())
    billing=json.loads((RESULT/'billing-usage-analysis.json').read_bytes())
    base=json.loads((RUN/'delivery-verification.json').read_bytes())
    with zipfile.ZipFile(base['ZIP']) as z:
        assert z.testzip() is None
        for name,wanted in json.loads(z.read('ARCHIVE_MANIFEST.json'))['files_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==wanted
    assert sha(base['ZIP'])==base['sha256']
    for name in ['analyze_completed.py','finalize_evidence.py','delivery_recovery.py','DELIVERY_RECOVERY.json','adjudication-scope-verification.json','hardware-snapshot.json','runtime-packages.json','background-launch.json']:
        shutil.copyfile(RUN/name,RESULT/name)
    atomic(RESULT/'source-verification-after-inference.json',{'checked_at_UTC':datetime.now(timezone.utc).isoformat(),'evaluator_fpy_provider_adapter_and_case_hashes_match_frozen_protocol':True,'protocol_sha256':sha(RUN/'protocol.json'),'matched_main_calls':404,'separate_pilot_calls':36,'hosted_requests_reconciled':220,'new_inference_calls_during_export':0})
    rows=[]
    for arm in ['v8','openai']:
        for track,m in summary[arm].items():
            e=m['raw_exact'];rows.append(f"| {arm} | {track} | {e['precision']:.4f} | {e['recall']:.4f} | {e['f1']:.4f} | {m['json_valid']}/101 | {m['schema_valid']}/101 | {m['raw_contract_admitted']}/101 | {m['exact_pipeline_state_turns']}/101 | {m['median_seconds']:.3f} |")
    table='\n'.join(['| System | Track | Precision | Recall | Raw exact F1 | JSON | Schema | Raw contract | Exact pipeline state, turns | Median seconds |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']+rows)
    findings='''# Findings under the frozen current-pipeline configuration

This is an inspected synthetic DEVELOPMENT comparison of configured extraction systems, not a general model ranking. The primary scoring, prompts, cases and generation settings were not changed after main inference began. Independent human review remains incomplete.

V8's zero operational raw exact F1 is dominated by output-contract failures. For example, main/v8/component/commodities/0032 t1 emits update entries with aspect/dimension/slot_id/value and intent_confidence 100; the contract requires candidate_value/status/confidence/evidence_text and confidence at most 1. Other turns emit dictionaries of slot names or aspect/new_value pairs instead of the required patch format. These outputs can mention relevant facts but are not admitted. Zero operational F1 is not evidence that the model has no semantic knowledge.

OpenAI satisfies the declared outer schema in all main calls, but outer schema validity does not enforce JSON inside candidate_value, duplicate-slot uniqueness, evidence, correct statuses or canonical value shapes. Recorded examples include a scalar horizon instead of a periods/unit object, a slot/value wrapper instead of the expected column string, unmentioned updates where omission is expected, and confirmed versus provided correction status. These cause exact-match or admission failures. No extra decoding, wrapper removal, numeric/synonym normalization or output repair was introduced to improve scores.

The evaluator uses the actual current ElicitationEngine with a COMMON STRICT comparison admission adapter before reducer/recovery. This adapter enforces JSON-encoded candidate values and duplicate rejection. The stock fpy Pydantic provider can tolerate ordinary text candidate strings; its tolerance, retries and persistence are not the evaluated configuration. The primary results must be described with this adapter policy, not as an unrestricted production-provider benchmark. Native Structured Outputs is supplied only to the hosted arm, while the local arm uses the same semantic instructions/input with recorded greedy decoding. Representation constraints differ by design.

Both rollout arms have zero exactly complete state turns and zero exactly complete final conversation states. This does not mean every slot is wrong. Deterministic recovery frequently runs and must not be credited to the raw model. The pre-inference oracle audit found 20 main turns whose pipeline state differs from the expected state even when gold extraction is supplied; examples map multi-fact messages or unknown-answer sentences into an unrelated pending output-granularity slot. These are application/recovery findings, separate from model extraction quality.

V8's zero primary extra-slot flags does NOT establish absence of invented requirements: many rejected noncanonical updates cannot be read as contract tuples. Extra/forbidden flags for either model are not a count of every hallucination, wrong value or repeated state. Inspect raw records and automated failure findings, then independently adjudicate.

Component state uses supplied correct prior state. A correct final component snapshot does not mean the system completed that entire conversation. Use rollout final-state counts for accumulated-state conclusions; PAPER_UPDATE_TABLE.csv and supplementary-contract-audit.json distinguish these denominators.

The cohort has 20 conversations, 101 calls per model/track, 257 gold non-intent updates per track, 39 positive slot types out of a 79-slot schema, and 16 English/two Roman Urdu/two mixed conversations. It is selected from already inspected public V8-D development. Shared templates/entities and tiny language subgroups limit generalization. No significance/equivalence test or universal winner is established. Older 31-case OpenAI and 100-case v8 scores use different conditions and must remain separate tables.

Next research work should review the common response/value contract and deterministic recovery before considering another training round. No new prompt tuning, parser repair, retraining, paid evaluation or sealed-test access was performed after this study. Any subsequent experiment requires a separately frozen version and an explicit scope/budget decision; remaining authorization is not an automatic experiment expansion.
'''
    (RESULT/'FINDINGS_AND_LIMITATIONS.md').write_text(findings,encoding='utf-8')
    proposal='# Proposed paper update; not a replacement manuscript\n\nAdd a separate current-pipeline development cohort table, keeping existing N/M/P/V8-D and historical OpenAI results separate.\n\n'+table+'\n\nMethods: 20 frozen conversations, 101 calls per model/track, plus two separate pilot conversations (nine calls per model/track), one candidate per turn, exact non-intent slot/value/status scoring before recovery, current fpy commit '+protocol['fpy_commit']+' plus bundled fixes and the common strict admission adapter. Local RTX A4000 CUDA BF16 and the pinned final-step v8 adapter; hosted exact gpt-5-mini-2025-08-07 with native Structured Outputs/minimal reasoning. Disclosure: local cap 1024, hosted cap 2048; decoding/transport differ and arms ran serially.\n\nOne user-adjudicated state contract change was frozen BEFORE inference: target_column="true" remains a provided raw extraction but is invalid under the unchanged current validator, and that status is carried through expected/supplied prior states. Case IDs and raw extraction gold are unchanged; all eight changed state-status fields are inventoried. Do not describe this as completed independent human annotation.\n\nResults/discussion: contract compatibility and value/status representations dominate the low scores; deterministic recovery is not raw model credit, and neither rollout reaches an exactly complete final state. See FINDINGS_AND_LIMITATIONS.md and the detailed CSV/JSON. Avoid claims about general model capability or a universal winner.\n\nTotal hosted usage-rate estimate including cache rates: $'+billing['usage_rate_estimate_USD']+'; conservative guard-accounted cost without cache discounts: $'+billing['conservative_ledger_accounted_USD']+'. Reserved/uncertain $0; provider invoice unverified. All 220 hosted requests, including pilot and failed semantic outputs, are included.\n\nValidation: 91 evaluator/application tests and three offline server-authentication tests; source/adapter/case hashes verified; raw SDK responses, usage, context hashes, per-slot/domain/language results and recovery traces archived. Human review remains incomplete; this is inspected synthetic development and a fixed script, not an adaptive interview or untouched final test.\n'
    (RESULT/'PAPER_UPDATE_PROPOSAL.md').write_text(proposal,encoding='utf-8')
    completion=json.loads((RESULT/'completion.json').read_bytes())
    completion.update(final_evidence_verified_at_UTC=datetime.now(timezone.utc).isoformat(),usage_rate_estimate_USD=billing['usage_rate_estimate_USD'],conservative_budget_accounted_USD=billing['conservative_ledger_accounted_USD'],reserved_USD=0,uncertain_USD=0,independent_human_review_complete=False,packaging_recovery_only=True,source_and_primary_scoring_unchanged_after_main=True)
    atomic(RESULT/'completion.json',completion)
    note='\n\nAdditional verified reporting: read FINDINGS_AND_LIMITATIONS.md, PAPER_UPDATE_TABLE.csv, PAPER_UPDATE_PROPOSAL.md, supplementary-contract-audit.json, billing-usage-analysis.json and adjudication-scope-verification.json. The common strict comparison adapter is part of this measured configuration; raw contract failures must not be described as absence of semantic knowledge. Component final snapshots have supplied gold priors and are not complete-conversation success.\n\nThe authoritative shared ledger remains on this GPU server. Its export/snapshot is for analysis, not a fresh budget on another PC. Do not rerun completed stages or issue additional paid requests. Incomplete-stage recovery must use the SAME run, frozen source/cases and authoritative shared ledger under its exclusive lock. Fresh credentials must be loaded explicitly from Windows User environment into the worker; .env is ignored and is not read by this evaluator. SDK retries remain zero and provider URL is https://api.openai.com/v1.\n\nThe first export failed only because focused cases omit optional template_variant metadata; the recovery treats absent metadata as null and preserves the failed export folder. No inference was repeated, and primary scores were not changed. The supplied recovery source and receipt document this.\n'
    handoff=(RESULT/'LOCAL_CODEX_HANDOFF.md').read_text(encoding='utf-8')+note+'\nMain result table:\n\n'+table+'\n\nWhole-study usage-rate estimate $'+billing['usage_rate_estimate_USD']+'; guard-accounted $'+billing['conservative_ledger_accounted_USD']+'; reserved/uncertain $0.\n'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=Path.home()/'Downloads'/(RUN.name+'-verified-evidence-'+stamp+'.zip')
    handoff+='\nFinal ZIP on GPU PC: '+str(destination)+'. Its SHA-256 is in the adjacent .zip.sha256 file and the standalone handoff returned with it; an archive cannot embed its own checksum without circularity.\n'
    (RESULT/'LOCAL_CODEX_HANDOFF.md').write_text(handoff,encoding='utf-8')
    with (RESULT/'README.md').open('a',encoding='utf-8') as stream:
        stream.write('\n\nThe final verified packet adds FINDINGS_AND_LIMITATIONS.md, PAPER_UPDATE_TABLE.csv and PAPER_UPDATE_PROPOSAL.md. Cache-adjusted usage-rate estimate: $'+billing['usage_rate_estimate_USD']+'; conservative ledger accounting: $'+billing['conservative_ledger_accounted_USD']+'. Both include pilots and every hosted generation request, with zero reserved/uncertain requests. These are usage estimates, not an invoice. The original export and primary results are preserved.\n')
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as keyref:secret=winreg.QueryValueEx(keyref,'OPENAI_API_KEY')[0].encode()
    pattern=re.compile(rb'(?i)(?<![A-Za-z0-9])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,})')
    for path in RESULT.rglob('*'):
        if not path.is_file():continue
        body=path.read_bytes();body=gzip.decompress(body) if path.suffix=='.gz' else body
        assert not (secret in body or pattern.search(body)),'Credential-shaped content in final evidence'
        assert not path.name.startswith('.env')
    files={path.relative_to(RESULT).as_posix():sha(path) for path in RESULT.rglob('*') if path.is_file() and path.name!='artifact-manifest.json'}
    atomic(RESULT/'artifact-manifest.json',{'files_sha256':files,'credential_scan_passed':True,'weights_optimizer_caches_included':False})
    files['artifact-manifest.json']=sha(RESULT/'artifact-manifest.json')
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(files):z.write(RESULT/name,name)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps({'files_sha256':files},indent=2,sort_keys=True))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        for name,wanted in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==wanted
    checksum=sha(destination);destination.with_suffix('.zip.sha256').write_text(checksum+'  '+destination.name+'\n',encoding='ascii')
    copy=RUN/destination.name;shutil.copyfile(destination,copy);assert sha(copy)==checksum
    roundtrip=RUN/('final-roundtrip-'+stamp);roundtrip.mkdir()
    with zipfile.ZipFile(copy) as z:
        for name in z.namelist():assert (roundtrip/name).resolve().is_relative_to(roundtrip.resolve())
        z.extractall(roundtrip)
    for name,wanted in files.items():assert sha(roundtrip/name)==wanted
    standalone=RUN/'LOCAL_CODEX_HANDOFF.md';standalone.write_text(handoff+'\nVerified final ZIP SHA-256: '+checksum+'\n',encoding='utf-8')
    shutil.copyfile(standalone,destination.parent/(RUN.name+'-LOCAL_CODEX_HANDOFF.md'))
    final={'status':'complete_verified_evidence','results':str(RESULT),'ZIP':str(destination),'sha256':checksum,'bytes':destination.stat().st_size,'members':len(files)+1,'standalone_handoff':str(standalone),'CRC_member_hashes_copy_roundtrip_credential_scan_verified':True,'main_calls':404,'pilot_calls':36,'hosted_requests_reconciled':220,'usage_rate_estimate_USD':billing['usage_rate_estimate_USD'],'conservative_budget_accounted_USD':billing['conservative_ledger_accounted_USD'],'reserved_USD':0,'uncertain_USD':0,'published_to_GitHub':False}
    atomic(RUN/'FINAL_DELIVERY_RECEIPT.json',final)
    shutil.copyfile(RUN/'supervisor-status.json',RUN/'supervisor-status-before-export-recovery.json')
    atomic(RUN/'supervisor-status.json',{'status':'complete_verified_evidence','at_UTC':datetime.now(timezone.utc).isoformat(),'delivery':final,'no_more_inference_or_paid_requests':True,'export_recovery_documented':True})
    print(json.dumps(final))


if __name__=='__main__':main()
