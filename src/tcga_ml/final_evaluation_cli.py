from __future__ import annotations

import argparse
import json

from .final_evaluation import (
    create_final_evaluation_lock,
    run_final_evaluation,
    verify_final_evaluation_lock,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lock and verify the development-selected final pipeline before "
            "the one-time holdout evaluation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock = subparsers.add_parser(
        "lock",
        help="Create a non-overwriting final pipeline/data/evidence lock.",
    )
    lock.add_argument("--config", required=True)
    lock.add_argument("--matrix", required=True)
    lock.add_argument("--split", required=True)
    lock.add_argument("--genes", required=True)
    lock.add_argument(
        "--evidence",
        action="append",
        required=True,
        help="Development-only result JSON; repeat for every selection input.",
    )
    lock.add_argument("--output", required=True)

    verify = subparsers.add_parser(
        "verify",
        help="Verify the lock digest and all bound input artifacts.",
    )
    verify.add_argument("--lock", required=True)
    verify.add_argument("--matrix", required=True)
    verify.add_argument("--split", required=True)
    verify.add_argument("--genes", required=True)
    verify.add_argument(
        "--evidence",
        action="append",
        default=None,
        help="Optionally re-verify the complete selection-evidence set.",
    )
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run the receipt-guarded final holdout evaluation exactly once.",
    )
    evaluate.add_argument("--lock", required=True)
    evaluate.add_argument("--matrix", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--genes", required=True)
    evaluate.add_argument("--outdir", required=True)
    evaluate.add_argument(
        "--receipt",
        required=True,
        help="Persistent one-time receipt path outside --outdir.",
    )
    evaluate.add_argument(
        "--evidence",
        action="append",
        default=None,
        help="Optionally re-verify the complete selection-evidence set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "lock":
        payload = create_final_evaluation_lock(
            args.config,
            args.matrix,
            args.split,
            args.genes,
            args.evidence,
            args.output,
        )
        print(
            json.dumps(
                {
                    "holdout_status": payload["holdout_status"],
                    "lock_sha256": payload["lock_sha256"],
                    "output": args.output,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "verify":
        payload = verify_final_evaluation_lock(
            args.lock,
            args.matrix,
            args.split,
            args.genes,
            evidence_paths=args.evidence,
        )
        print(
            json.dumps(
                {
                    "holdout_status": payload["holdout_status"],
                    "lock_sha256": payload["lock_sha256"],
                    "verified": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    payload = run_final_evaluation(
        args.lock,
        args.matrix,
        args.split,
        args.genes,
        args.outdir,
        args.receipt,
        evidence_paths=args.evidence,
    )
    print(
        json.dumps(
            {
                "candidate_id": payload["candidate_id"],
                "holdout_used": payload["holdout_used"],
                "metrics": payload["metrics"],
                "outputs": payload["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
