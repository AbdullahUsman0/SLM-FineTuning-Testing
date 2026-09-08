"""Fail-fast integrity checks for corpus v4 before a GPU training run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.corpus_v4 import validate_corpus_v4


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    corpus_root = ROOT / "corpus-v4"
    manifest = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = corpus_root / relative
        if not path.is_file():
            raise SystemExit(f"Missing corpus artifact: {path}")
        actual = sha256(path)
        if actual.lower() != expected["sha256"].lower():
            raise SystemExit(
                f"Checksum mismatch for {relative}: expected {expected['sha256']}, got {actual}"
            )

    corpus = read_jsonl(corpus_root / "corpus-v4.jsonl")
    sft = [
        *read_jsonl(corpus_root / "sft" / "train.jsonl"),
        *read_jsonl(corpus_root / "sft" / "validation.jsonl"),
    ]
    validate_corpus_v4(corpus, sft)

    split_ids = {
        split: {
            item["scenario_id"]
            for item in read_jsonl(corpus_root / "splits" / f"{split}.jsonl")
        }
        for split in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_ids[left] & split_ids[right]:
            raise SystemExit(f"{left} and {right} scenario IDs overlap")

    print("Corpus v4 integrity: PASS")
    print(f"Scenarios: {manifest['scenario_count']} {manifest['split_counts']}")
    print(f"SFT examples: {manifest['sft_example_count']} {manifest['sft_split_counts']}")
    print("Tasks: structured extraction only")
    print("Test examples in SFT: 0")


if __name__ == "__main__":
    main()
