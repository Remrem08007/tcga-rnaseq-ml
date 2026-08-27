from __future__ import annotations

import argparse

from .feature_budget import parse_gene_budget
from .splitting import DEFAULT_SEED
from .xgboost_benchmark import probe_cuda, run_compute_scaling, run_xgboost_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run development-only XGBoost nonlinear/GPU benchmarks.")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="Probe whether XGBoost can actually train on a visible CUDA device.")
    probe.set_defaults(command="probe")

    cv = sub.add_parser("cv", help="Run leakage-safe XGBoost cross-validation.")
    _common_data_args(cv)
    cv.add_argument("--genes", required=True)
    cv.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    cv.add_argument("--threads", type=int, default=0)
    cv.add_argument("--fold-jobs", type=int, default=1)
    _model_args(cv)

    scaling = sub.add_parser("scale", help="Measure CPU-thread scaling and optionally a CUDA run.")
    _common_data_args(scaling)
    scaling.add_argument("--cpu-threads", action="append", type=int, required=True)
    scaling.add_argument("--include-gpu", action="store_true")
    scaling.add_argument("--require-gpu", action="store_true")
    _model_args(scaling)
    return parser


def _common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--outdir", required=True)


def _model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gene-budget", default="1000")
    parser.add_argument("--negative-policy", choices=("error", "clip"), default="error")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        probe = probe_cuda()
        print(f"cuda_available={str(probe.available).lower()}")
        print(f"resolved={probe.resolved}")
        print(f"reason={probe.reason}")
        return 0

    budget = parse_gene_budget(args.gene_budget)
    if args.command == "cv":
        payload = run_xgboost_benchmark(
            args.matrix,
            args.split,
            args.genes,
            args.outdir,
            requested_device=args.device,
            threads=args.threads,
            fold_jobs=args.fold_jobs,
            gene_budget=budget,
            negative_policy=args.negative_policy,
            cv_folds=args.cv_folds,
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
        result = payload["benchmark"]
        print(
            f"xgboost device={result['resolved_device']} "
            f"macro_f1={result['metrics']['macro_f1']['mean']:.4f} "
            f"wall={result['wall_seconds']:.3f}s"
        )
        return 0

    payload = run_compute_scaling(
        args.matrix,
        args.split,
        args.outdir,
        cpu_threads=args.cpu_threads,
        include_gpu=args.include_gpu,
        require_gpu=args.require_gpu,
        gene_budget=budget,
        negative_policy=args.negative_policy,
        cv_folds=args.cv_folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    print(f"compute-scaling runs={len(payload['runs'])} output={args.outdir}/compute_scaling.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
