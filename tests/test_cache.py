import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tcga_ml.cache import build_expression_cache, parse_gene_identifier, scan_expression_tsv


FIXTURES = Path(__file__).parent / "fixtures"


def test_scan_expression_and_gene_identifier():
    scan = scan_expression_tsv(FIXTURES / "expression_cache_small.tsv")
    assert scan.n_genes == 3
    assert len(scan.barcodes) == 3
    assert len(scan.sha256) == 64
    assert parse_gene_identifier("A1BG|1") == ("A1BG", "1")
    assert parse_gene_identifier("NO_PIPE") == ("NO_PIPE", "")


def test_build_float32_sample_by_gene_cache(tmp_path):
    metadata = build_expression_cache(
        FIXTURES / "expression_cache_small.tsv",
        FIXTURES / "cohort_cache_small.tsv",
        tmp_path,
        chunk_genes=2,
    )
    matrix = np.load(tmp_path / "expression.float32.npy", mmap_mode="r")
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(matrix[0], [1.0, -0.01, 10.0])
    np.testing.assert_allclose(matrix[1], [3.0, 8.5, 30.0])
    assert metadata["shape"] == [2, 3]
    assert metadata["n_nonfinite_values"] == 0
    assert metadata["transformation"].startswith("none")

    with (tmp_path / "genes.tsv").open() as handle:
        genes = list(csv.DictReader(handle, delimiter="\t"))
    assert genes[0]["symbol"] == "A1BG"
    assert genes[1]["entrez_id"] == "100133144"

    payload = json.loads((tmp_path / "cache_metadata.json").read_text())
    assert payload["dtype"] == "float32"


def test_cache_preserves_nan_when_selected(tmp_path):
    cohort = tmp_path / "cohort.tsv"
    cohort.write_text(
        "matrix_index\texpression_barcode\tparticipant_barcode\tcancer_type\n"
        "1\tTCGA-BB-0002-01A-11R-0000-01\tTCGA-BB-0002\tLUAD\n"
    )
    metadata = build_expression_cache(
        FIXTURES / "expression_cache_small.tsv",
        cohort,
        tmp_path / "cache",
    )
    matrix = np.load(tmp_path / "cache" / "expression.float32.npy", mmap_mode="r")
    assert np.isnan(matrix[0, 1])
    assert metadata["n_nonfinite_values"] == 1


def test_cache_rejects_barcode_index_mismatch(tmp_path):
    cohort = tmp_path / "cohort.tsv"
    cohort.write_text(
        "matrix_index\texpression_barcode\tparticipant_barcode\tcancer_type\n"
        "0\tTCGA-BB-0002-01A-11R-0000-01\tTCGA-BB-0002\tLUAD\n"
    )
    with pytest.raises(ValueError, match="barcode/index mismatch"):
        build_expression_cache(FIXTURES / "expression_cache_small.tsv", cohort, tmp_path / "cache")
