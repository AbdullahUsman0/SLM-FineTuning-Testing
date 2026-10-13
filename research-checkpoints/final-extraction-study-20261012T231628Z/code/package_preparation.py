"""Create distinct public preparation and private review ZIPs; exclusive, checksum-verified."""
from pathlib import Path
import re
import zipfile
import runtime as rt


def checked_zip(target,root,files,manifest):
    root=root.resolve()
    inventory={path.relative_to(root).as_posix():rt.sha(path) for path in files}
    if any(not path.resolve().is_relative_to(root) for path in files):raise ValueError('Archive member outside intended root')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as package:
        for path in sorted(files):package.write(path,path.relative_to(root).as_posix())
        package.writestr('PACKAGE_MANIFEST.json',rt.dumps({**manifest,'files_sha256':inventory}))
    with zipfile.ZipFile(target) as package:
        if package.testzip() is not None or len(package.namelist())!=len(inventory)+1:
            raise ValueError('Archive CRC/member inventory failed')
        import hashlib
        for name,expected in inventory.items():
            if hashlib.sha256(package.read(name)).hexdigest()!=expected:raise ValueError('Archived bytes changed')
    checksum=rt.sha(target)
    with target.with_suffix(target.suffix+'.sha256').open('x',encoding='ascii') as stream:
        stream.write(checksum+'  '+target.name+'\n')
    return {'filename':target.name,'sha256':checksum,'bytes':target.stat().st_size,
        'member_count':len(inventory)+1,'CRC_and_all_member_SHA256_verified':True}


def main():
    checkpoint=rt.CHECKPOINT
    private=Path(rt.read(checkpoint/'workspace-intake.json')['private'])
    if rt.read(checkpoint/'EXECUTION_DECISION.json')['new_paid_budget_usd']!=0:raise ValueError('Unexpected changed execution scope')
    rt.verify_sources();rt.verify_models(rt.read(private/'runtime-config.json'))
    private_out=private/'packages';private_out.mkdir(exist_ok=False)
    review_readme=private/'REVIEW_PACKAGE_README.md'
    with review_readme.open('x',encoding='utf-8') as stream:
        stream.write('# Private proposed evaluation review\n\n'
            'Agent-authored candidate and proposed labels; independent human review remains pending. '
            'Current user decision: use existing results; no new scored cohort runs and $0 new paid budget. '
            'The budget template is an archived proposal, not approval. '
            'Transfer securely via AnyDesk if review resumes. Do not upload these labels or their generator publicly. '
            'No weights, credentials or caches are included. The runtime configuration has training-PC paths; '
            'configure paths explicitly on another PC. Do not treat any of this material as training data.\n')
    private_files=[review_readme,private/'runtime-config.json']
    for name in ('candidate','review','preparation-code'):
        private_files.extend(p for p in (private/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    private_info=checked_zip(private_out/'final-extraction-study-20261012-private-review.zip',private,private_files,
        {'scope':'Private proposed final-cohort review; NOT approved or scored','public_upload_authorized':False,
         'candidate_sha256':rt.sha(private/'candidate/conversations.jsonl')})
    rt.atomic(checkpoint/'PRIVATE_PACKAGE_RECEIPT.json',{**private_info,'directory_on_training_PC':str(private_out),
        'public_upload_authorized':False,'transfer_method':'Secure AnyDesk transfer if review resumes'},exclusive=True)
    public_files=[p for p in checkpoint.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    allowed={'.py','.json','.md','.txt','.csv'}
    if any(p.suffix not in allowed for p in public_files):raise ValueError('Unexpected public artifact type')
    forbidden={'make_candidate.py','conversations.jsonl','review-sheet.csv','adapter_model.safetensors',
        'adapter_config.json','.env','budget-approval.json'}
    if any(p.name in forbidden for p in public_files):raise ValueError('Protected artifact in public package')
    # Tokens in documentation like OPENAI_API_KEY are safe; actual credential-shaped literals are not.
    credential=re.compile(r'(?<![A-Za-z0-9])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,})')
    for path in public_files:
        if credential.search(path.read_text(encoding='utf-8-sig')):raise ValueError('Credential-shaped literal found')
    public_info=checked_zip(checkpoint/'final-extraction-study-20261012-offline.zip',checkpoint,public_files,
        {'scope':'Verified existing evidence and offline preparation only','new_paid_budget_usd':0,
         'new_final_cohort_included':False,'models_included':False,'private_review_included':False,
         'dependencies_external':'Pinned r8 research bundle, parser/runtime pins and historical repository inputs'})
    rt.atomic(checkpoint/'PUBLIC_PACKAGE_RECEIPT.json',public_info,exclusive=True)
    print(rt.dumps({'public':public_info,'private':private_info,'paid_API_calls':0}))


if __name__=='__main__':main()
