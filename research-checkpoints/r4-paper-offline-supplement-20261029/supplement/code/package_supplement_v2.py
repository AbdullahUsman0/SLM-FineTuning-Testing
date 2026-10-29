"""Local-only sealed delivery; no publishing, inference, ledger access or credentials loaded."""
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];C=json.loads((ROOT/'workspace.json').read_bytes());OUT=ROOT/'deliverables';E=Path(C['evidence']);REPO=Path(C['repo'])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main():
    assert sha(C['original_ZIP'])==C['original_ZIP_sha256']
    source_manifest=json.loads((E/'ARCHIVE_MANIFEST.json').read_bytes())['files_sha256']
    assert all(sha(E/name)==wanted for name,wanted in source_manifest.items())
    ledger_checks={name:{'before':wanted,'after':sha(name),'unchanged':sha(name)==wanted} for name,wanted in C['original_ledger_hashes'].items()}
    assert all(x['unchanged'] for x in ledger_checks.values())
    write(OUT/'ORIGINAL_LEDGER_IMMUTABILITY.json',{'shared_ledger_opened_by_analysis':False,'files':ledger_checks,'all_unchanged':True})
    assert json.loads((OUT/'BASELINE_REPRODUCTION.json').read_bytes())['passed']
    assert json.loads((OUT/'human-review/REVIEW_KIT_STATUS.json').read_bytes())['reviewer_decisions_filled']==0
    for path in ROOT.glob('logs/*.receipt.json'):assert json.loads(path.read_bytes())['returncode']==0
    # Freeze only tracked PUBLIC fpy source/test fixtures, no .env, .git or caches.
    fpy=ROOT/'runtime/fpy';tracked=subprocess.run(['git','ls-files'],cwd=fpy,check=True,capture_output=True,text=True).stdout.splitlines()
    selected=[n for n in tracked if n.startswith(('src/','tests/','evaluation/','scripts/')) or n=='pyproject.toml']
    assert not any('sealed' in n.lower() or 'private' in n.lower() or n.startswith('.env') for n in selected)
    repro=OUT/'reproduction';repro.mkdir(exist_ok=True);source_dir=repro/'fpy';source_dir.mkdir(exist_ok=True)
    hashes={}
    for name in selected:
        target=source_dir/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(fpy/name)
        else:shutil.copyfile(fpy/name,target)
        hashes[name]=sha(target)
    protocol=json.loads((E/'protocol.json').read_bytes())
    assert all(hashes[name]==wanted for name,wanted in protocol['fpy_source_sha256'].items())
    write(repro/'FPY_SOURCE_MANIFEST.json',{'commit':protocol['fpy_commit'],'bundled_fix_patch_sha256':protocol['bundle_patch_sha256'],'files_sha256':hashes,'includes_two_bundled_modified_test_files':True,'private_or_sealed_FYP_labels_included':False})
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    write(repro/'ENVIRONMENT_VERSIONS.json',{'python':sys.version,'packages':dict(sorted(versions.items())),'environment_values_or_credentials_included':False,'installed_packages_changed':False})
    completion={'status':'offline_supplement_completed_human_review_pending','scope':'separate versioned POST-HOC supplement to immutable R4/R4.1 evidence','required_publication':C['required_commit'],'original_ZIP_sha256':C['original_ZIP_sha256'],'baseline_tests_passed':299,'main_scores_inputs_states_reproduced':404,'supplement_control_tests_passed':10,'main_conversations':20,'main_turns':101,'known_nonintent_slots_reported':78,'positive_support_slots':39,'paired_bootstrap_resamples':10000,'paired_conversation_units':20,'script_family_sensitivity_units':10,'oracle_mismatches_reproduced':20,'oracle_mismatch_calls_correct_before_recovery':20,'worked_examples':6,'review_forms':{'gold_rows':101,'paired_response_rows':202,'second_reviewer_rows':101,'adjudication_rows':101,'filled_human_decisions':0,'mapping':'coordinator only'},'completed_offline':['evidence/runtime verification','299 tests and 404 offline replays','human-review kit and validated proposal-only importer','trace-backed taxonomy/oracle/errors','all-slot/conversation/subgroup/intervention/paired exploratory analysis','fairness and A-F scope audit','future protocol and blank forms','paper paragraphs/tables/six-page allocation'],'prepared_pending_human':['reviewer independence and identities','gold-first review and second review','blinded response review and adjudication','approval of any proposed label changes'],'unexecuted_future':['participant recruitment and collection','prospective independent labels','new inference or training','future paid experiment'],'new_model_calls':0,'new_paid_API_calls':0,'new_training_runs':0,'new_adapters':0,'sealed_labels_accessed':False,'shared_spending_ledger_modified':False,'original_gold_prompts_parser_scores_modified':False,'independent_human_review_complete':False,'whole_research_complete':False,'paper_manuscript_modified':False,'replacement_prose_is_proposal_only':True,'published_or_pushed':False,'legacy_primary_receipt_gaps':['historical A','historical B','one-conversation D','full E rule report'],'unchanged_R4_F1':{'v8_component':0.9686274509803922,'openai_component':0.6953405017921147,'v8_rollout':0.9568627450980393,'openai_rollout':0.6938053097345133},'unchanged_final_rollout_conversations':{'v8':9,'openai':1,'denominator':20},'human_decisions_remaining':['assign independent humans and staged gold-first/A-B review','adjudicate exact span/value/status and application-state disagreements; proposals do not auto-apply','supply missing legacy public raw receipts if full A/B/D/E re-verification is desired','approve prospective writer/reviewer/ethics/privacy/cohort/metric/resource design before any collection or inference','provide separate explicit publishing authorization if desired']}
    write(OUT/'ANALYSIS_COMPLETION.json',completion)
    (OUT/'LOCAL_CODEX_HANDOFF.md').write_text('''# Local Codex handoff: offline R4 paper evidence supplement

This is a LOCAL, UNPUBLISHED post-hoc supplement. The research branch remains at
6c51ee2c32e16d3237fce78523a07e04cdb29667. No paid APIs, generation, training,
sealed labels, ledger changes or publishing were performed in this phase.
The latest paper edits are on the main PC; no manuscript was replaced here.

Verify the adjacent supplement ZIP checksum and every PACKAGE_MANIFEST.json
member hash. The nested source-evidence ZIP retains the original R4 checksum
4d4a33bd6ec4ac1bab28b47d37ee324a315177035826119e36a230cd71c143ae;
its 529 archive members and CRC were checked. Read ANALYSIS_COMPLETION.json,
POSTHOC_ANALYSIS_PLAN.md, paper/, analysis/, fairness-and-methods/ and test logs.

Preserve the published R4 point estimates and their cohort boundaries: local
component/rollout exact actionable F1 0.9686274509803922/0.9568627450980393;
hosted 0.6953405017921147/0.6938053097345133; complete final rollout state 9/20
versus 1/20. Hosted calls were fresh; local emissions were verified reused. Both
were rescored under the same R4 rules. Do not mix cohorts or treat component
final snapshots as successful full conversations. The main contribution is the
separation of structure, actionable extraction and application-state reliability.

New offline findings: all 20 gold-oracle mismatch calls are correct before recovery,
then differ after trace-reported recovery. Unknown-turn local abstention is 20/20,
but prior state is preserved 13/20; hosted preservation is 15/20. See the six full
trace-backed examples. Coverage is 39/78 positive non-intent slots; the other slots
and unknown-ID FP are retained. Explore conditional paired intervals only under
the saved POST-HOC plan; 18/20 cases share a template variant. Do not overwrite
the original no-CI receipt or claim population independence/significance.

For human review, give only 01-gold-first-review-kit.zip first. After gold decisions
are locked, release 02-blinded-response-review-kit.zip. Withhold coordinator/,
the full supplement and source evidence containing model identities and scores.
All review fields remain blank. A/B mapping is stable per conversation. Use the
proposal-only importer; it validates immutable inputs and schema/evidence, never
applies changes or certifies human independence. Keep original labels and proposed
adjudications separate. Independent human review is INCOMPLETE.

The A-F matrix records different models/cohorts/policies, not a two-model-by-six-
method factorial study. The one-case slot-wise pilot, configured XGrammar result
and historical size runs have explicit limits. Missing old primary A/B/D/full-E
receipts are identified rather than fabricated as reverified results.

Update the actual existing main-PC paper with backup, using paper/ replacement
paragraphs and the suggested six-page main/supplement split. Keep N/M/P/V8-D
history separate and report raw versus pipeline results. Return actual edited
paths/build checks when that authorized editing is performed on the main PC.

future-independent-evaluation/ contains ONLY a proposal and blank collection/
annotation/approval forms. No independent requests or participants were generated.
Human, institutional/privacy, cohort/metric and any new inference/budget approvals
remain required. Old spending approvals do not authorize another experiment.

Do not push, publish, run models, call paid APIs, retrain or access sealed labels
without separate authorization. This handoff does not authorize those actions.
''',encoding='utf-8')
    (OUT/'README.md').write_text('''# R4 paper evidence supplement — offline, post-hoc, unpublished

Start with ANALYSIS_COMPLETION.json and LOCAL_CODEX_HANDOFF.md.

| Deliverable | Location |
|---|---|
| Baseline 299 tests / 404 replays | BASELINE_REPRODUCTION.json; logs/ in package |
| Human gold-first and blinded forms | human-review/ staged ZIPs and reviewers/ |
| Coordinator key and importer | human-review/coordinator/; human-review/import_reviews.py |
| Error taxonomy, oracle and six examples | analysis/ERROR_TAXONOMY_GUIDE.md, ORACLE_MECHANISM.md, WORKED_EXAMPLES.md |
| All slots, conversation/subgroup/intervention results | analysis/*.csv |
| Exploratory paired intervals | POSTHOC_ANALYSIS_PLAN.md; analysis/exploratory_paired_intervals.json |
| Fairness/A-F scope and receipt limits | fairness-and-methods/ |
| Future independent protocol and blank forms | future-independent-evaluation/ |
| Paper prose/tables and six-page allocation | paper/ |
| Frozen source/dependencies | reproduction/ |

All human decisions remain blank. Give reviewers only the staged review ZIPs,
not this full package or coordinator mapping. Future study resources are proposals,
not executed or approved experiments. No original evidence/labels/scores or shared
spending ledger were changed. Nothing was pushed or published.
''',encoding='utf-8')
    delivery=ROOT/'delivery-r2';delivery.mkdir();package=delivery/'package';package.mkdir()
    shutil.copytree(OUT,package/'deliverables');shutil.copytree(ROOT/'scripts',package/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'logs',package/'logs');(package/'source-evidence').mkdir();shutil.copyfile(C['original_ZIP'],package/'source-evidence/r4-native-comparison-results.zip')
    shutil.copytree(source_dir,package/'reproduction/fpy')
    shutil.copyfile(OUT/'LOCAL_CODEX_HANDOFF.md',package/'LOCAL_CODEX_HANDOFF.md');shutil.copyfile(OUT/'ANALYSIS_COMPLETION.json',package/'ANALYSIS_COMPLETION.json')
    (package/'README.md').write_text('Read LOCAL_CODEX_HANDOFF.md and deliverables/README.md.\nThis package is local/unpublished and contains a coordinator-only blinding key.\nDistribute only the staged reviewer ZIPs to reviewers.\n\nFor offline reproduction, read REPRODUCTION.md.\n',encoding='utf-8')
    (package/'REPRODUCTION.md').write_text('''# Offline reproduction

The original evidence is nested under source-evidence/ and retains its exact SHA.
Code is under code/, outputs under deliverables/, logs under logs/. The copied
public fpy source/test snapshot is under reproduction/fpy (67 tracked files, with
the four original bundled fixes). No .env, model weights, caches or live ledger.

Use the recorded Python/dependency versions; do not replace CUDA packages. Existing
tzdata supplies Asia/Karachi. The two modified bundled test files are included.
The human-review importer is standalone standard-library Python and needs no model.

Prepare a NEW workspace (no inference or network):

    python code/prepare_analysis_workspace.py --output <NEW-folder> --repo <research-checkout-at-required-commit> --offline-deps <existing-isolated-dependencies>

Then run code from that prepared workspace's scripts directory, e.g.:

    python <NEW-folder>/scripts/run_stage.py coverage_and_intervals.py
    python <NEW-folder>/scripts/run_stage.py build_review_kit.py
    python <NEW-folder>/scripts/run_stage.py error_and_oracle_analysis.py
    python <NEW-folder>/scripts/run_stage.py fairness_and_methods.py
    python <NEW-folder>/scripts/run_stage.py write_paper_and_future.py
    python <NEW-folder>/scripts/run_stage.py test_supplement.py

Analysis/form generation does not load models or require weights. The byte-identical
original baseline replay verifier additionally HASHES the adapter at its recorded
path and provider files in the research checkout. On another PC use the already
verified main-PC path mapping or explicitly map only to the same hash-verified
adapter. Do not edit the frozen protocol, substitute weights or claim changed
contexts are a fresh experiment. Run scripts/reproduce_baseline.py only in a new
workspace with the correct paths/dependencies; it performs tests and saved replay,
not inference. No paid stages or spending-ledger access are allowed.

To import real human forms, use deliverables/human-review/import_reviews.py with
the coordinator reference manifest and a NEW output directory. Original labels
are never changed. Blank fields are pending; software does not certify reviewers.
''',encoding='utf-8')
    # Smoke the shipped source layout, not just the development directory.
    smoke=delivery/'portable-smoke';env=dict(os.environ)
    for name in list(env):
        if any(s in name.upper() for s in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL','OPENAI_ORG','OPENAI_PROJECT','OPENAI_BASE_URL']):env.pop(name,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    setup=[C['python'],'-B',str(ROOT/'scripts/offline_guard.py'),str(package/'code/prepare_analysis_workspace.py'),'--package',str(package),'--output',str(smoke),'--repo',C['repo'],'--offline-deps',C['offline_deps']]
    with (ROOT/'logs/portable-setup.log').open('wb') as log:subprocess.run(setup,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'logs/portable-baseline.log').open('wb') as log:subprocess.run([C['python'],'-B',str(smoke/'scripts/offline_guard.py'),str(smoke/'scripts/reproduce_baseline.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    smoke_result=json.loads((smoke/'deliverables/REPRODUCED_BASELINE.json').read_bytes());assert smoke_result['passed']
    for path in [ROOT/'logs/portable-setup.log',ROOT/'logs/portable-baseline.log',smoke/'logs/baseline-tests-reproduction.log',smoke/'logs/baseline-replay-reproduction.log']:shutil.copyfile(path,package/'logs'/path.name)
    write(package/'deliverables/PORTABLE_REPRODUCTION.json',{'passed':True,'setup_command':setup,'baseline_tests':299,'main_replays':404,'no_model_calls':True,'no_network_calls':True,'same_shipped_source_layout':True})
    # No API credential is loaded: scan plaintext and retain the exact original
    # ciphertext exclusions already documented in the sealed evidence.
    oldscan=json.loads((E/'credential-scan.json').read_bytes());opaque={r['file_sha256'] for r in oldscan['opaque_reasoning_field_exclusions']}
    pattern=re.compile(rb'(?i)(?<![A-Za-z0-9])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,}|(?:AKIA|ASIA)[A-Z0-9]{16})')
    scanned=0;exclusions=[]
    def scan(data,name):
        nonlocal scanned
        scanned+=1
        if name.endswith('.gz'):return scan(gzip.decompress(data),name[:-3])
        if name.endswith('.zip'):
            import io
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                assert z.testzip() is None
                for member in z.namelist():scan(z.read(member),name+'!'+member)
            return
        view=data;h=hashlib.sha256(data).hexdigest()
        if h in opaque:
            obj=json.loads(data)
            for output in obj['response']['output']:
                if output.get('type')=='reasoning' and isinstance(output.get('encrypted_content'),str):output['encrypted_content']='[OPAQUE FIELD EXCLUDED ONLY FROM PATTERN VIEW]'
            view=json.dumps(obj,ensure_ascii=False).encode();exclusions.append({'member':name,'original_file_sha256':h,'reason':'Exactly preserved original provider encrypted-reasoning field; original raw bytes unchanged'})
        if pattern.search(view):raise ValueError('Credential-shaped plaintext detected; details withheld')
    for path in package.rglob('*'):
        if path.is_file():
            assert not path.name.startswith('.env') and path.suffix not in {'.safetensors','.pt','.bin','.pyc'}
            scan(path.read_bytes(),path.relative_to(package).as_posix())
    write(package/'CREDENTIAL_SCAN.json',{'passed':True,'payloads_scanned':scanned,'API_credentials_loaded_for_analysis_or_scanning':False,'opaque_original_evidence_exclusions':exclusions,'original_sealed_evidence_preserved':True,'no_environment_files_weights_or_caches':True})
    assert sha(C['original_ZIP'])==C['original_ZIP_sha256']
    assert all(sha(name)==wanted for name,wanted in C['original_ledger_hashes'].items())
    hashes={p.relative_to(package).as_posix():sha(p) for p in package.rglob('*') if p.is_file()}
    write(package/'PACKAGE_MANIFEST.json',{'files_sha256':hashes,'published':False,'independent_human_review_complete':False})
    hashes['PACKAGE_MANIFEST.json']=sha(package/'PACKAGE_MANIFEST.json')
    archive=delivery/'r4-paper-offline-supplement-20261028-r2.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(hashes):z.write(package/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(hashes)
        for name,wanted in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==wanted
    checksum=sha(archive);archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n',encoding='ascii')
    download=Path.home()/'Downloads'/archive.name;assert not download.exists();shutil.copyfile(archive,download);assert sha(download)==checksum
    download.with_suffix('.zip.sha256').write_text(checksum+'  '+download.name+'\n',encoding='ascii')
    roundtrip=delivery/'extracted-roundtrip';roundtrip.mkdir()
    with zipfile.ZipFile(download) as z:
        for name in z.namelist():assert (roundtrip/name).resolve().is_relative_to(roundtrip.resolve())
        z.extractall(roundtrip)
    assert all(sha(roundtrip/name)==wanted for name,wanted in hashes.items())
    receipt={'status':'local_offline_supplement_complete_verified','workspace':str(ROOT),'deliverables':str(OUT),'package':str(package),'ZIP':str(download),'sha256':checksum,'bytes':download.stat().st_size,'members':len(hashes),'CRC_all_member_hashes_copy_extraction_verified':True,'credential_scan_passed':True,'portable_baseline_299_tests_404_replays_passed':True,'original_evidence_and_ledger_hashes_unchanged':True,'new_paid_calls':0,'new_model_calls':0,'new_training':0,'human_review_complete':False,'published_or_pushed':False}
    write(delivery/'DELIVERY_VERIFICATION.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
