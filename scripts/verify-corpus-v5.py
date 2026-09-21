"""Validate development records and hashes; never parse sealed final labels."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_slm_lab.corpus_v5 import verify_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sealed-output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify_artifacts(args.output, args.sealed_output)
    except (ValueError, KeyError, OSError):
        parser.exit(1, "Artifact verification failed; do not alter the frozen revision.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
