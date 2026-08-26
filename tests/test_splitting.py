import json
from pathlib import Path

import pytest

from tcga_ml.splitting import (
    SampleRecord,
    make_development_cv,
    make_holdout_split,
    read_samples_table,
    split_manifest_sha256,
    write_split_outputs,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_holdout_split_is_stratified_and_disjoint():
    samples = read_samples_table(FIXTURES / "samples_split_small.tsv")
    split = make_holdout_split(samples, holdout_fraction=0.2, seed=20260825)
    development = {row.participant_barcode for row in split if row.split == "development"}
    holdout = {row.participant_barcode for row in split if row.split == "holdout"}
    assert not development & holdout
    assert len(development) == 12
    assert len(holdout) == 3
    assert {row.cancer_type for row in split if row.split == "holdout"} == {"BRCA", "LUAD", "LUSC"}


def test_split_is_independent_of_input_row_order():
    samples = read_samples_table(FIXTURES / "samples_split_small.tsv")
    forward = make_holdout_split(samples, seed=42)
    reverse = make_holdout_split(list(reversed(samples)), seed=42)
    forward_map = {row.participant_barcode: row.split for row in forward}
    reverse_map = {row.participant_barcode: row.split for row in reverse}
    assert forward_map == reverse_map
    assert split_manifest_sha256(forward) == split_manifest_sha256(reverse)


def test_split_outputs_have_frozen_hash(tmp_path):
    samples = read_samples_table(FIXTURES / "samples_split_small.tsv")
    split = make_holdout_split(samples, seed=7)
    manifest, summary = write_split_outputs(split, tmp_path, seed=7)
    payload = json.loads(summary.read_text())
    assert payload["manifest_sha256"] == split_manifest_sha256(split)
    assert payload["n_samples"] == 15
    assert manifest.read_text().startswith("cache_index\tparticipant_barcode")


def test_duplicate_participant_rejected():
    record = SampleRecord(0, 0, "A", "TCGA-AA-0001", "BRCA")
    duplicate = SampleRecord(1, 1, "B", "TCGA-AA-0001", "BRCA")
    with pytest.raises(ValueError, match="duplicate participants"):
        make_holdout_split([record, duplicate])


def test_cv_is_reproducible():
    cv1 = make_development_cv(n_splits=3, seed=9)
    cv2 = make_development_cv(n_splits=3, seed=9)
    assert cv1.random_state == cv2.random_state == 9
    assert cv1.n_splits == cv2.n_splits == 3
