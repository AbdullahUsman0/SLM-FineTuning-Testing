"""Export readable development examples only; no final data is opened."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_slm_lab.corpus_v6 import read_jsonl_gz,verify_artifacts


def export(corpus,output):
    verify_artifacts(corpus)
    rows=read_jsonl_gz(corpus/"review-sample.jsonl.gz")
    if any(row["split"]!="train" for row in rows):
        raise ValueError("review sample must contain development training cases only")
    if output.exists():
        raise FileExistsError("review export exists; choose another file")
    output.parent.mkdir(parents=True,exist_ok=True)
    text=["# v6 domain review sample", "", "One deterministic training source per subdomain; 20 sources.",
        "This sample is NOT a completed human review. Review full development data and overlap queues before training.",
        "", "For each case check units, calendar, explicit evidence, omissions, corrections, naturalness and question relevance.",
        "Write decisions in a separate ledger with scenario ID, reviewer, UTC time, approved/rejected and substantive reasons.", ""]
    for row in rows:
        text.extend([f"## {row['domain']} — {row['category']}","",f"ID: `{row['scenario_id']}`", "",
                     f"Entity group: `{row['entity_group']}`. Template: `{row['template_family']}`.", ""])
        # Show chronological prior turn plus one variant; the other two are
        # accessible in compressed development component records.
        selected=[t for t in row["turns"] if t["render_category"]=="prior_confirmation" or t["variant"]==0]
        for turn in selected:
            text.extend([f"User ({turn['turn_id']}):", "",turn["message"],"", "Expected extraction:","","```json",
                         json.dumps(turn["gold_extraction"],ensure_ascii=False,indent=2),"```",""])
        text.extend([f"Next question slot: `{row['expected_question_slot']}`", "",row["ideal_question"],"",
                     "Review decision: pending.", ""])
    output.write_bytes(("\n".join(text)+"\n").encode("utf8"))
    return {"output":str(output),"review_sources":len(rows),"final_labels_accessed":False,"status":"pending_human_review"}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(export(a.corpus,a.output),indent=2))
