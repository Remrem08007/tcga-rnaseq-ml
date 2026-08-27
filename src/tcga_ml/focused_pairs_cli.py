from __future__ import annotations

import argparse
import json

from .feature_budget import FEATURE_BUDGET_MODELS
from .focused_pairs import FOCUSED_PAIRS, run_focused_pair_studies
from .splitting import DEFAULT_SEED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run development-only focused TCGA cancer-pair studies."
    )
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--pair",
        action="append",
        choices=tuple(FOCUSED_PAIRS),
        help="Focused pair to run; repeat for multiple pairs. Defaults to both locked pairs.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=FEATURE_BUDGET_MODELS,
        help="Linear model to run; repeat for multiple models. Defaults to elastic_net.",
    )
    parser.add_argument("--gene-budget", default="1000")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--negative-policy", choices=("error", "clip"), default="error")
    parser.add_argument("--scaler", choices=("standard", "robust"), default="standard")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_focused_pair_studies(
        args.matrix,
        args.split,
        args.genes,
        args.outdir,
        pairs=args.pair or tuple(FOCUSED_PAIRS),
        models=args.model or ("elastic_net",),
        gene_budget=args.gene_budget,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        negative_policy=args.negative_policy,
        scaler=args.scaler,
        seed=args.seed,
    )
    compact = {
        "evaluation_scope": payload["evaluation_scope"],
        "holdout_used": payload["holdout_used"],
        "ranking_hardest_first": payload["ranking_hardest_first"],
        "outputs": payload["outputs"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
