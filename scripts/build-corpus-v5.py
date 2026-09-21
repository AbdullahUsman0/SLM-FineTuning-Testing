"""Build a new immutable v5 revision locally, without training or API calls."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_slm_lab.corpus_v5 import write_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sealed-output", required=True, type=Path)
    args = parser.parse_args()
    output, sealed = args.output.resolve(), args.sealed_output.resolve()
    if output.name != sealed.name or not output.name.startswith("v5-"):
        parser.error("public and sealed directories must use the same v5 revision name")
    try:
        manifest = write_artifacts(output, sealed, version=output.name)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps({"output": str(output), "sealed_output": str(sealed),
                      "status": manifest["build_status"],
                      "scenarios": manifest["statistics"]["source_scenarios"],
                      "examples": manifest["statistics"]["total_examples"],
                      "split_scenarios": manifest["statistics"]["split_scenarios"],
                      "sft_examples": manifest["statistics"]["sft_examples"],
                      "public_sft_hashes": {k: v["sha256"] for k, v in manifest["files"].items() if k.startswith("sft/")},
                      "sealed_hashes": manifest["sealed_files"]}, indent=2))


if __name__ == "__main__":
    main()
