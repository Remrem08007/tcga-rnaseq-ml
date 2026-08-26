from __future__ import annotations

from dataclasses import dataclass


GDC_DATA_BASE = "https://api.gdc.cancer.gov/data"


@dataclass(frozen=True)
class DataSource:
    key: str
    filename: str
    gdc_uuid: str
    description: str

    @property
    def url(self) -> str:
        return f"{GDC_DATA_BASE}/{self.gdc_uuid}"


SOURCES: dict[str, DataSource] = {
    "expression": DataSource(
        key="expression",
        filename="EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv",
        gdc_uuid="3586c0da-64d0-4b74-a449-5ff4d9136611",
        description="TCGA PanCancer Atlas batch-corrected RNA expression matrix",
    ),
    "sample_quality": DataSource(
        key="sample_quality",
        filename="merged_sample_quality_annotations.tsv",
        gdc_uuid="1a7d7be8-675d-4e60-a105-19d4121bdebf",
        description="PanCancer Atlas merged analyte/sample quality annotations",
    ),
    "clinical_followup": DataSource(
        key="clinical_followup",
        filename="clinical_PANCAN_patient_with_followup.tsv",
        gdc_uuid="0fc78496-818b-4896-bd83-52db1f533c5c",
        description="PanCancer Atlas clinical data with follow-up",
    ),
    "clinical_cdr": DataSource(
        key="clinical_cdr",
        filename="TCGA-CDR-SupplementalTableS1.xlsx",
        gdc_uuid="1b5f413e-a8d1-4d10-92eb-7c4ae739ed81",
        description="TCGA Clinical Data Resource supplemental table",
    ),
}


def get_source(key: str) -> DataSource:
    try:
        return SOURCES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(SOURCES))
        raise KeyError(f"unknown source {key!r}; choose one of: {choices}") from exc
