"""Build an immutable weather/economics candidate; no training or API calls."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_slm_lab.corpus_v6 import write_artifacts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sealed-output", type=Path, required=True)
    p.add_argument("--per-domain", type=int, default=100)
    args = p.parse_args()
    if args.output.name != args.sealed_output.name or not args.output.name.startswith("v6-"):
        p.error("use matching v6 revision names for public and sealed outputs")
    manifest = write_artifacts(args.output, args.sealed_output, per_domain=args.per_domain, version=args.output.name)
    print(json.dumps({"version":manifest["corpus_version"], "status":manifest["build_status"],
        "scenarios":manifest["statistics"]["source_scenarios"], "examples":manifest["statistics"]["total_examples"],
        "splits":manifest["statistics"]["split_scenarios"], "domains":manifest["statistics"]["domain_family_scenarios"]}, indent=2))


if __name__ == "__main__":
    main()
