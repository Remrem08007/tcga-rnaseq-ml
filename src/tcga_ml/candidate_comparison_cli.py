from __future__ import annotations

import argparse
import json

from .candidate_comparison import compare_candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a holdout-blind comparison of development-CV candidates."
    )
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    payload = compare_candidates(args.evidence, args.outdir)
    print(json.dumps({"n_candidates": len(payload["candidates"]), "selection_made": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
