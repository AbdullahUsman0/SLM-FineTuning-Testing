"""Seal verified local export with exact public-test-fixture scan exceptions."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1];C=json.loads((ROOT/'workspace.json').read_bytes());delivery=ROOT/'delivery-r2';package=delivery/'package';E=Path(C['evidence'])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    portable=json.loads((package/'deliverables/PORTABLE_REPRODUCTION.json').read_bytes());assert portable['passed']
    assert portable['baseline_tests']==299 and portable['main_replays']==404
    assert sha(C['original_ZIP'])==C['original_ZIP_sha256']
    assert all(sha(name)==wanted for name,wanted in C['original_ledger_hashes'].items())
    shutil.copyfile(__file__,package/'code/seal_existing_package.py')
    opaque={r['file_sha256'] for r in json.loads((E/'credential-scan.json').read_bytes())['opaque_reasoning_field_exclusions']}
    fixture_files={'13ab24630fc4fb52d02a75dd61622db544701daba5ba38de48e605343c4a7ee2':'test_extractor_prompt.py','a1ff7e3e3c8b4a5eb23501a4631f235603c7797b3d81ecddfc8af97c94ea67bb':'test_validation.py'}
    fixture_token_sha='1a5d44a2dca19669d72edf4c4f1c27c4c1ca4b4408fbb17f6ce4ad452d78ddb3'
    pattern=re.compile(rb'(?i)(?<![A-Za-z0-9])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,}|(?:AKIA|ASIA)[A-Z0-9]{16})')
    checked=0;exclusions=[];fixtures=[]
    def scan(data,name):
        nonlocal checked
        checked+=1
        if name.endswith('.gz'):return scan(gzip.decompress(data),name[:-3])
        if name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                assert z.testzip() is None
                for member in z.namelist():scan(z.read(member),name+'!'+member)
            return
        view=data;h=hashlib.sha256(data).hexdigest()
        if h in opaque:
            obj=json.loads(data)
            for item in obj['response']['output']:
                if item.get('type')=='reasoning' and isinstance(item.get('encrypted_content'),str):item['encrypted_content']='[OPAQUE FIELD EXCLUDED ONLY FROM PATTERN VIEW]'
            view=json.dumps(obj,ensure_ascii=False).encode();exclusions.append({'member':name,'original_file_sha256':h,'reason':'Byte-identical original encrypted-reasoning field; original raw bytes preserved'})
        if h in fixture_files:
            assert name.endswith('/tests/unit/'+fixture_files[h])
            matches=list(pattern.finditer(view));assert len(matches)==2 and all(hashlib.sha256(m.group()).hexdigest()==fixture_token_sha for m in matches)
            view=pattern.sub(b'[DOCUMENTED PUBLIC UNIT-TEST FIXTURE: SCANNER VIEW ONLY]',view)
            fixtures.append({'member':name,'source_file_sha256':h,'fixture_token_sha256':fixture_token_sha,'occurrences':len(matches),'reason':'Dummy public AWS-key fixture used to test redaction/rejection, not a loaded live credential','raw_source_modified':False})
        if pattern.search(view):raise ValueError('Unadjudicated credential-shaped plaintext detected; contents withheld')
    for path in package.rglob('*'):
        if path.is_file():
            assert not path.name.startswith('.env') and path.suffix not in {'.safetensors','.pt','.bin','.pyc'}
            scan(path.read_bytes(),path.relative_to(package).as_posix())
    write(package/'CREDENTIAL_SCAN.json',{'passed':True,'payloads_scanned':checked,'live_API_credentials_loaded':False,'opaque_original_evidence_exclusions':exclusions,'public_unit_test_fixture_exceptions':fixtures,'all_exceptions_exact_file_or_original_member_SHA_bound':True,'original_sources_and_evidence_unchanged':True,'environment_files_models_weights_caches_included':False})
    hashes={p.relative_to(package).as_posix():sha(p) for p in package.rglob('*') if p.is_file()}
    write(package/'PACKAGE_MANIFEST.json',{'files_sha256':hashes,'published':False,'independent_human_review_complete':False});hashes['PACKAGE_MANIFEST.json']=sha(package/'PACKAGE_MANIFEST.json')
    archive=delivery/'r4-paper-offline-supplement-20261028-r2.zip';assert not archive.exists()
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
    assert sha(C['original_ZIP'])==C['original_ZIP_sha256'] and all(sha(name)==wanted for name,wanted in C['original_ledger_hashes'].items())
    receipt={'status':'local_offline_supplement_complete_verified','workspace':str(ROOT),'deliverables':str(ROOT/'deliverables'),'package':str(package),'ZIP':str(download),'sha256':checksum,'bytes':archive.stat().st_size,'members':len(hashes),'CRC_all_member_hashes_copy_extraction_verified':True,'credential_scan_passed':True,'public_dummy_test_fixture_exceptions_documented':True,'baseline_tests_passed':299,'main_replays_passed':404,'supplement_tests_passed':10,'fresh_shipped_layout_reproduction_passed':True,'original_evidence_and_ledger_hashes_unchanged':True,'new_paid_calls':0,'new_model_calls':0,'new_training':0,'independent_human_review_complete':False,'published_or_pushed':False}
    write(delivery/'DELIVERY_VERIFICATION.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
