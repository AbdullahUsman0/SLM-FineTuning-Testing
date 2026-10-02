"""Materialize verified development data after a separately recorded review.

This prepares inputs only. It neither starts a trainer nor accesses final labels.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_slm_lab.corpus_v6 import verify_artifacts
from local_slm_lab.corpus_v5 import file_record


def approval_check(path,manifest_hash):
    data=json.loads(path.read_text(encoding="utf8"))
    if data.get("corpus_manifest_sha256")!=manifest_hash:
        raise ValueError("review approval refers to a different corpus")
    for field in ("reviewer","reviewed_at_utc","review_ledger_sha256"):
        if not isinstance(data.get(field),str) or not data[field].strip():
            raise ValueError(f"missing review provenance: {field}")
    for field in ("training_labels_approved","validation_labels_approved","overlap_review_complete"):
        if data.get(field) is not True:
            raise ValueError(f"review unresolved: {field}")
    if data.get("final_labels_used_for_training_or_selection") is not False:
        raise ValueError("approval must explicitly exclude final labels")
    return data


def prepare(corpus,output,approval):
    corpus=corpus.resolve(); output=output.resolve()
    verify_artifacts(corpus)
    manifest=json.loads((corpus/"manifest.json").read_bytes())
    manifest_hash=file_record(corpus/"manifest.json")["sha256"]
    reviewed=approval_check(approval.resolve(),manifest_hash)
    if output.exists():
        raise FileExistsError("use a new empty input directory")
    if output.is_relative_to(corpus) or corpus.is_relative_to(output):
        raise ValueError("materialize outside the immutable corpus")
    output.mkdir(parents=True)
    records={}
    for source,name in (("sft/train.jsonl.gz","train.jsonl"),("sft/validation.jsonl.gz","validation.jsonl"),
                        ("splits/validation.jsonl.gz","validation-cases.jsonl"),("splits/smoke.jsonl.gz","smoke-cases.jsonl")):
        dest=output/name; temporary=output/("."+name+".partial")
        h=hashlib.sha256()
        with gzip.open(corpus/source,"rb") as inp, temporary.open("xb") as out:
            for block in iter(lambda:inp.read(1024*1024),b""):
                out.write(block); h.update(block)
            out.flush(); os.fsync(out.fileno())
        if h.hexdigest()!=manifest["files"][source]["uncompressed_sha256"]:
            raise ValueError("materialized bytes differ from frozen input")
        os.replace(temporary,dest); records[name]=file_record(dest)
    lock={"corpus_version":manifest["corpus_version"],"corpus_manifest_sha256":manifest_hash,
          "fpy_commit":manifest["fpy_commit"],"review_approval":reviewed,"files":records,
          "final_labels_accessed":False,"status":"prepared_not_trained"}
    (output/"input-lock.json").write_text(json.dumps(lock,indent=2)+"\n",encoding="utf8")
    return lock


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--review-approval",required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(prepare(a.corpus,a.output,a.review_approval),indent=2))
