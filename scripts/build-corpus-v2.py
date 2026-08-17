import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.corpus_v2 import write_artifacts_v2


if __name__ == "__main__":
    manifest = write_artifacts_v2(ROOT / "corpus-v2")
    print(f"Wrote {manifest['scenario_count']} v2 scenarios")
    print(f"Splits: {manifest['split_counts']}")
    print(f"SFT tasks: {manifest['sft_task_counts']}")

