from __future__ import annotations

import argparse

from .cohort import TARGET_CANCERS, build_cohort, read_expression_barcodes, read_quality_annotations, write_cohort_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic, quality-filtered TCGA RNA-seq cohort manifest.")
    parser.add_argument("--expression", required=True, help="PanCancer gene-expression TSV.")
    parser.add_argument("--quality", required=True, help="merged_sample_quality_annotations.tsv.")
    parser.add_argument("--outdir", default="data/processed", help="Output directory.")
    parser.add_argument(
        "--cancer",
        action="append",
        dest="cancers",
        help="Cancer acronym to include; repeat for multiple. Defaults to the locked 10-cancer cohort.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cancers = args.cancers or list(TARGET_CANCERS)
    barcodes = read_expression_barcodes(args.expression)
    quality = read_quality_annotations(args.quality)
    result = build_cohort(barcodes, quality, target_cancers=cancers)
    manifest, summary = write_cohort_outputs(result, args.outdir)
    print(f"selected={result.n_selected}/{result.n_expression_samples}")
    print(manifest)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
