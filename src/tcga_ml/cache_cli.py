from __future__ import annotations

import argparse
import json

from .cache import build_expression_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a memory-mappable float32 sample×gene cache from the TCGA expression TSV.")
    parser.add_argument("--expression", required=True, help="PanCancer gene-expression TSV.")
    parser.add_argument("--cohort", required=True, help="cohort.tsv produced by tcga_ml.cohort_cli.")
    parser.add_argument("--outdir", default="data/cache", help="Cache output directory.")
    parser.add_argument("--chunk-genes", type=int, default=256, help="Genes per write block (default: 256).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = build_expression_cache(
        args.expression,
        args.cohort,
        args.outdir,
        chunk_genes=args.chunk_genes,
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
