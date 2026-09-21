"""Paired source-scenario bootstrap and validation-only adapter recommendation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_slm_lab.v5_eval import (
    paired_bootstrap, safe_error, select_validation_candidate, sha256, strict_json, write_report,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="Reference/base report")
    parser.add_argument("--right", type=Path, required=True, help="Candidate report")
    parser.add_argument("--output", type=Path, required=True, help="New output; never overwritten")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-runtime-difference", action="store_true")
    parser.add_argument("--select", action="store_true", help="Recommend adapter using validation only; does not authorize final")
    parser.add_argument("--candidate", action="append", default=[], metavar="ID=REPORT",
                        help="Additional validation candidates for --select (all must be declared)")
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError("comparison output already exists")
        left_path, right_path = args.left.resolve(), args.right.resolve()
        left = strict_json(left_path.read_text(encoding="utf-8"))
        right = strict_json(right_path.read_text(encoding="utf-8"))
        result = paired_bootstrap(left, right, samples=args.samples, seed=args.seed,
                                  allow_runtime_difference=args.allow_runtime_difference)
        result["reports"] = {"left": {"path": str(left_path), "sha256": sha256(left_path)},
                             "right": {"path": str(right_path), "sha256": sha256(right_path)}}
        result["comparison_script_sha256"] = sha256(Path(__file__))
        if args.candidate and not args.select:
            raise ValueError("candidate list requires validation selection mode")
        if args.select:
            if args.allow_runtime_difference:
                raise ValueError("selection does not permit runtime differences")
            name = right["metadata"].get("input", {}).get("candidate_id") or right_path.stem
            candidates = {name: right}
            result["candidate_reports"] = {name: {"path": str(right_path), "sha256": sha256(right_path)}}
            for declaration in args.candidate:
                name, separator, filename = declaration.partition("=")
                if not separator or not name or name in candidates:
                    raise ValueError("candidate declarations require unique IDs")
                path = Path(filename).resolve()
                candidates[name] = strict_json(path.read_text(encoding="utf-8"))
                result["candidate_reports"][name] = {"path": str(path), "sha256": sha256(path)}
            result["selection"] = select_validation_candidate(left, candidates)
        write_report(result, output)
        print(json.dumps({"output": str(output), "nonintent": result["nonintent"]}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": safe_error(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
