from __future__ import annotations

import argparse

from .benchmark import run_benchmark
from .models import MODEL_NAMES
from .splitting import DEFAULT_SEED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-validate classical TCGA RNA-seq models on the development set only.")
    parser.add_argument("--matrix", required=True, help="expression.float32.npy cache.")
    parser.add_argument("--split", required=True, help="Frozen split_manifest.tsv.")
    parser.add_argument("--outdir", default="results/benchmarks/classical")
    parser.add_argument("--model", action="append", choices=MODEL_NAMES, dest="models")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=0, help="Outer CV workers; 0 derives from SLURM_CPUS_PER_TASK/CPU count.")
    parser.add_argument("--negative-policy", choices=["error", "clip"], default="error")
    parser.add_argument("--scaler", choices=["standard", "robust"], default="standard")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable model/fold progress and the one-minute running heartbeat.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_benchmark(
        args.matrix,
        args.split,
        args.outdir,
        models=args.models or MODEL_NAMES,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        negative_policy=args.negative_policy,
        scaler=args.scaler,
        seed=args.seed,
        pca_components=args.pca_components,
        show_progress=not args.no_progress,
    )
    for result in payload["models"]:
        macro = result["metrics"]["macro_f1"]["mean"]
        print(f"{result['model']}\tmacro_f1={macro:.6f}\twall={result['wall_seconds']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
