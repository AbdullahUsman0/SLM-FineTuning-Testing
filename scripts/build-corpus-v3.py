import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.corpus_v3 import write_artifacts_v3


if __name__ == "__main__":
    manifest = write_artifacts_v3(ROOT / "corpus-v3")
    print(f"Wrote {manifest['scenario_count']} v3 scenarios")
    print(f"SFT examples: {manifest['sft_example_count']}")
    print(f"Splits: {manifest['sft_split_counts']}")
    print(f"Variants: {manifest['sft_variant_counts']}")
