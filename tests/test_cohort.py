import csv
import json
from pathlib import Path

from tcga_ml.cohort import (
    build_cohort,
    read_expression_barcodes,
    read_quality_annotations,
    write_cohort_outputs,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_read_quality_and_expression_headers():
    barcodes = read_expression_barcodes(FIXTURES / "expression_small.tsv")
    quality = read_quality_annotations(FIXTURES / "quality_small.tsv")
    assert len(barcodes) == 7
    assert len(quality) == 6
    assert quality[0].participant_barcode == "TCGA-AA-0001"
    assert quality[0].cancer_type == "BRCA"


def test_read_expression_header_accepts_official_quoted_fields(tmp_path):
    expression = tmp_path / "quoted.tsv"
    expression.write_text(
        '"gene_id"\t"TCGA-OR-A5J1-01A-11R-A29S-07"\t'
        '"TCGA-OR-A5J2-01A-11R-A29S-07"\n',
        encoding="utf-8",
    )

    assert read_expression_barcodes(expression) == [
        "TCGA-OR-A5J1-01A-11R-A29S-07",
        "TCGA-OR-A5J2-01A-11R-A29S-07",
    ]


def test_read_quality_normalises_compact_sample_and_portion_field(tmp_path):
    quality = tmp_path / "quality.tsv"
    quality.write_text(
        "aliquot_barcode\tcancer type\tplatform\tDo_not_use\n"
        "TCGA-D9-A1X3-06A21-A20M-20\tSKCM\tIlluminaHiSeq_RNASeqV2\tfalse\n",
        encoding="utf-8",
    )

    records = read_quality_annotations(quality)

    assert records[0].aliquot_barcode == "TCGA-D9-A1X3-06A-21-A20M-20"
    assert records[0].sample_barcode == "TCGA-D9-A1X3-06"


def test_build_cohort_filters_and_deduplicates_patient():
    barcodes = read_expression_barcodes(FIXTURES / "expression_small.tsv")
    quality = read_quality_annotations(FIXTURES / "quality_small.tsv")
    result = build_cohort(barcodes, quality)

    assert result.n_expression_samples == 7
    assert result.n_selected == 1
    assert result.samples[0].expression_barcode == "TCGA-AA-0001-01A-11R-0000-01"
    assert result.exclusion_counts == {
        "unmatched_quality": 2,
        "do_not_use": 1,
        "non_primary": 1,
        "non_target_cancer": 1,
        "duplicate_participant": 1,
    }


def test_write_cohort_outputs(tmp_path):
    result = build_cohort(
        read_expression_barcodes(FIXTURES / "expression_small.tsv"),
        read_quality_annotations(FIXTURES / "quality_small.tsv"),
    )
    manifest, summary = write_cohort_outputs(result, tmp_path)
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["participant_barcode"] == "TCGA-AA-0001"
    assert json.loads(summary.read_text())["n_selected"] == 1
