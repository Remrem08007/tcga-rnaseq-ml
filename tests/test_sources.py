import pytest

from tcga_ml.sources import SOURCES, get_source


def test_official_source_registry_contains_m1_inputs():
    assert SOURCES["expression"].gdc_uuid == "3586c0da-64d0-4b74-a449-5ff4d9136611"
    assert SOURCES["sample_quality"].gdc_uuid == "1a7d7be8-675d-4e60-a105-19d4121bdebf"
    assert SOURCES["expression"].url.startswith("https://api.gdc.cancer.gov/data/")


def test_unknown_source_fails_cleanly():
    with pytest.raises(KeyError):
        get_source("not-a-source")
