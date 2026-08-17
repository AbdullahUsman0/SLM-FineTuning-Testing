import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.corpus import PROJECT_ROOT, write_artifacts


if __name__ == "__main__":
    manifest = write_artifacts(Path(PROJECT_ROOT) / "corpus")
    print(f"Wrote {manifest['scenario_count']} scenarios")
    print(f"Splits: {manifest['split_counts']}")
    print(f"SFT examples: {manifest['sft_example_counts']}")
