"""Prepare a NEW offline reproduction workspace from the delivered supplement."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--offline-deps',type=Path);args=parser.parse_args()
    package=args.package.resolve();root=args.output.resolve();assert not root.exists(),'Use a NEW directory';root.mkdir(parents=True)
    archive=package/'source-evidence/r4-native-comparison-results.zip';assert hashlib.sha256(archive.read_bytes()).hexdigest()=='4d4a33bd6ec4ac1bab28b47d37ee324a315177035826119e36a230cd71c143ae'
    evidence=root/'verified-evidence';evidence.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('ARCHIVE_MANIFEST.json'))['files_sha256']
        for name,wanted in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==wanted
        for name in z.namelist():assert (evidence/name).resolve().is_relative_to(evidence.resolve())
        z.extractall(evidence)
    runtime=root/'runtime';runtime.mkdir();(runtime/'tmp').mkdir();(root/'logs').mkdir();(root/'deliverables').mkdir()
    shutil.copytree(package/'code',root/'scripts');shutil.copytree(package/'reproduction/fpy',runtime/'fpy');shutil.copytree(evidence/'evaluator',runtime/'evaluator')
    for path in evidence.iterdir():
        if path.is_file() and path.suffix in {'.json','.py'}:shutil.copyfile(path,runtime/path.name)
    for name in ['calls.jsonl','main-scored-records.jsonl','pilot-scored-records.jsonl']:(runtime/name).write_bytes(gzip.decompress((evidence/(name+'.gz')).read_bytes()))
    deps=str(args.offline_deps.resolve()) if args.offline_deps else str(runtime/'offline-deps')
    if args.offline_deps:
        if os.name=='nt':subprocess.run(['cmd','/c','mklink','/J',str(runtime/'offline-deps'),deps],check=True,capture_output=True)
        else:(runtime/'offline-deps').symlink_to(deps,target_is_directory=True)
    config={'root':str(root),'repo':str(args.repo.resolve()),'evidence':str(evidence),'python':sys.executable,'offline_deps':deps,'required_commit':'6c51ee2c32e16d3237fce78523a07e04cdb29667','original_ZIP':str(archive),'original_ZIP_sha256':'4d4a33bd6ec4ac1bab28b47d37ee324a315177035826119e36a230cd71c143ae','original_ledger_hashes':{'D:/SLM/FYP-experiment-control/current-pipeline-v8-openai-20261023/budget.sqlite3':'not_accessed_in_portable_reproduction'}}
    (root/'workspace.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8');(runtime/'config.json').write_text(json.dumps({'code':str(runtime/'evaluator'),'repo':config['repo']},indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(package/'deliverables/POSTHOC_ANALYSIS_PLAN.md',root/'deliverables/POSTHOC_ANALYSIS_PLAN.md')
    (root/'deliverables/human-review').mkdir();shutil.copyfile(package/'deliverables/human-review/ANNOTATION_GUIDE.md',root/'deliverables/human-review/ANNOTATION_GUIDE.md')
    print(json.dumps({'prepared_workspace':str(root),'model_calls':0,'network_requests':0,'note':'Analysis does not require model weights. Original baseline verifier also hashes the adapter at its recorded path; use the main PC verified path mapping. Do not edit sealed evidence.'}))


if __name__=='__main__':main()
