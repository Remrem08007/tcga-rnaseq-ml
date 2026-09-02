from __future__ import annotations

import argparse

from .feature_budget import (
    DEFAULT_GENE_BUDGETS,
    FEATURE_BUDGET_MODELS,
    parse_gene_budget,
    run_feature_budget,
)
from .splitting import DEFAULT_SEED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe development-CV gene-budget and feature-stability "
            "experiments."
        )
    )
    parser.add_argument("--matrix", required=True, help="Sample x gene .npy cache.")
    parser.add_argument("--split", required=True, help="Frozen split_manifest.tsv.")
    parser.add_argument("--genes", required=True, help="genes.tsv from the cache builder.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--model",
        choices=FEATURE_BUDGET_MODELS,
        default="elastic_net",
        help="Linear model used for the gene-budget study (default: elastic_net).",
    )
    parser.add_argument(
        "--gene-budget",
        action="append",
        default=None,
        help=(
            "Repeatable positive integer or 'all'. If omitted, use the locked "
            "20/50/100/200/500/1000/5000/all roadmap budgets."
        ),
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help=(
            "0=auto from SLURM_CPUS_PER_TASK/os.cpu_count, capped at CV folds."
        ),
    )
    parser.add_argument(
        "--negative-policy",
        choices=("error", "clip"),
        default="error",
    )
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust"),
        default="standard",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable fold progress and the one-minute running heartbeat.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    budgets = (
        DEFAULT_GENE_BUDGETS
        if args.gene_budget is None
        else [parse_gene_budget(value) for value in args.gene_budget]
    )
    payload = run_feature_budget(
        args.matrix,
        args.split,
        args.genes,
        args.outdir,
        model_name=args.model,
        gene_budgets=budgets,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        negative_policy=args.negative_policy,
        scaler=args.scaler,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    print(
        f"feature-budget results: {args.outdir}/feature_budget.json "
        f"({len(payload['budgets'])} budgets)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
