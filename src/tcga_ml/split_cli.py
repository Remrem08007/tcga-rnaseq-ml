from __future__ import annotations

import argparse

from .splitting import DEFAULT_HOLDOUT_FRACTION, DEFAULT_SEED, make_holdout_split, read_samples_table, write_split_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze a deterministic stratified patient-level development/holdout split.")
    parser.add_argument("--samples", required=True, help="samples.tsv from the expression cache.")
    parser.add_argument("--outdir", default="data/processed", help="Output directory.")
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = read_samples_table(args.samples)
    split = make_holdout_split(samples, holdout_fraction=args.holdout_fraction, seed=args.seed)
    manifest, summary = write_split_outputs(
        split,
        args.outdir,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
    )
    n_holdout = sum(record.split == "holdout" for record in split)
    print(f"development={len(split) - n_holdout} holdout={n_holdout}")
    print(manifest)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
