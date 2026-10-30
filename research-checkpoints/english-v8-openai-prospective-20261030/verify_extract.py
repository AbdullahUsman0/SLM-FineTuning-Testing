"""Verify the English follow-up input/protocol capsule and extract into a NEW directory.

Standard library only. No models, API requests, credentials or sealed labels.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import uuid
import zipfile

HERE=Path(__file__).resolve().parent
EXPECTED='310f27c64afba23342310fad9742b90032f4fc63a8bc4ae3ae79b5b84a539b95'
def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,default=HERE/'english-protocol-r1.zip')
    parser.add_argument('--output',type=Path,default=Path('runtime')/f'english-protocol-r1-{uuid.uuid4().hex[:10]}')
    args=parser.parse_args()
    raw=args.zip.read_bytes()
    if sha(raw)!=EXPECTED:raise ValueError('ZIP SHA-256 mismatch; do not use the capsule')
    destination=args.output.resolve()
    if destination.exists():raise FileExistsError('Choose a NEW output directory; existing labels/runs will not be overwritten')
    with zipfile.ZipFile(args.zip) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP members')
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '\\' in info.filename or ':' in info.filename or '..' in p.parts or info.is_dir() or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Unsafe or unexpected archive path')
            if not (destination/str(p)).resolve().is_relative_to(destination):raise ValueError('Archive path escapes destination')
        manifest=json.loads(z.read('INPUT_PROTOCOL_SNAPSHOT.json'))['member_sha256']
        if set(manifest)!=set(names)-{'INPUT_PROTOCOL_SNAPSHOT.json'}:raise ValueError('Manifest coverage mismatch')
        for name,expected in manifest.items():
            if sha(z.read(name))!=expected:raise ValueError('Internal hash mismatch: '+name)
        primary=[json.loads(s) for s in z.read('primary-cases.jsonl').decode('utf-8').splitlines()]
        challenge=[json.loads(s) for s in z.read('multi-target-challenge.jsonl').decode('utf-8').splitlines()]
        if len(primary)!=44 or sum(len(x['turns']) for x in primary)!=205:raise ValueError('Primary counts mismatch')
        if len(challenge)!=3 or sum(len(x['turns']) for x in challenge)!=18:raise ValueError('Challenge counts mismatch')
        if any(x['language']!='en' for x in primary+challenge):raise ValueError('Non-English evaluation case')
        readiness=json.loads(z.read('READINESS.json'))
        if readiness['ready_for_inference'] or readiness['new_model_scores'] is not None:raise ValueError('Unexpected completed-evaluation claim')
        destination.mkdir(parents=True,exist_ok=False)
        for name in names:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    for name,expected in manifest.items():
        if sha((destination/name).read_bytes())!=expected:raise ValueError('Extracted file mismatch: '+name)
    print(json.dumps({'verified_zip_sha256':EXPECTED,'members_verified':len(names),'output_directory':str(destination),'primary_conversations':44,'primary_turns':205,'challenge_conversations':3,'ready_for_inference':False,'api_calls':0},indent=2))

if __name__=='__main__':main()
