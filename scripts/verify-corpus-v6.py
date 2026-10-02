"""Verify candidate development records and hashes; never deserialize final labels."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_slm_lab.corpus_v6 import verify_artifacts


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sealed-output",type=Path)
    a=p.parse_args()
    print(json.dumps(verify_artifacts(a.output,a.sealed_output),indent=2))
