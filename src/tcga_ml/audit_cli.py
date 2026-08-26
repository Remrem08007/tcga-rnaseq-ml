from __future__ import annotations

import argparse

from .normalization import audit_expression_matrix, write_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the raw cached TCGA expression matrix before normalization.")
    parser.add_argument("--matrix", required=True, help="Path to expression.float32.npy.")
    parser.add_argument("--output", default="data/cache/expression_audit.json", help="Audit JSON output path.")
    parser.add_argument("--max-quantile-values", type=int, default=1_000_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_expression_matrix(args.matrix, max_quantile_values=args.max_quantile_values)
    write_audit(args.output, audit)
    print(f"shape={audit.shape} negative_fraction={audit.negative_fraction:.6g} nonfinite={audit.n_nonfinite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
