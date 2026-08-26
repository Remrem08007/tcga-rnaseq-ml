import pytest

from tcga_ml.barcodes import parse_tcga_barcode, participant_barcode, sample_barcode


def test_parse_aliquot_barcode():
    parsed = parse_tcga_barcode("TCGA-02-0001-01C-01D-0182-01")
    assert parsed.participant == "TCGA-02-0001"
    assert parsed.sample == "TCGA-02-0001-01"
    assert parsed.sample_type_code == "01"
    assert parsed.vial == "C"
    assert parsed.is_primary_solid_tumor


def test_participant_and_sample_helpers():
    value = "TCGA-A1-A0SD-11A-11R-A114-13"
    assert participant_barcode(value) == "TCGA-A1-A0SD"
    assert sample_barcode(value) == "TCGA-A1-A0SD-11"
    assert not parse_tcga_barcode(value).is_primary_solid_tumor


def test_participant_only_barcode():
    parsed = parse_tcga_barcode("TCGA-AB-1234")
    assert parsed.sample is None
    with pytest.raises(ValueError):
        sample_barcode("TCGA-AB-1234")


def test_rejects_non_tcga_barcode():
    with pytest.raises(ValueError):
        parse_tcga_barcode("sample-01")
