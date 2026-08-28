from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

from .barcodes import parse_tcga_barcode


TARGET_CANCERS: tuple[str, ...] = (
    "BRCA",
    "LUAD",
    "LUSC",
    "KIRC",
    "KIRP",
    "LIHC",
    "STAD",
    "THCA",
    "UCEC",
    "COAD",
)
RNA_PLATFORM = "IlluminaHiSeq_RNASeqV2"
PRIMARY_SOLID_TUMOR_CODE = "01"


@dataclass(frozen=True)
class QualityRecord:
    aliquot_barcode: str
    sample_barcode: str
    participant_barcode: str
    cancer_type: str
    platform: str
    do_not_use: bool


@dataclass(frozen=True)
class CohortSample:
    matrix_index: int
    expression_barcode: str
    sample_barcode: str
    participant_barcode: str
    sample_type_code: str
    cancer_type: str


@dataclass(frozen=True)
class CohortResult:
    samples: tuple[CohortSample, ...]
    exclusion_counts: dict[str, int]
    n_expression_samples: int
    n_selected: int


def _normalise_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _resolve_column(fieldnames: Sequence[str], *aliases: str) -> str:
    normalised = {_normalise_column(name): name for name in fieldnames}
    for alias in aliases:
        key = _normalise_column(alias)
        if key in normalised:
            return normalised[key]
    raise ValueError(f"required column not found; expected one of {aliases!r}; found {list(fieldnames)!r}")


def _parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n", ""}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def read_quality_annotations(path: str | Path) -> list[QualityRecord]:
    records: list[QualityRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("sample-quality TSV has no header")
        aliquot_col = _resolve_column(reader.fieldnames, "aliquot_barcode", "aliquot barcode")
        cancer_col = _resolve_column(reader.fieldnames, "cancer type", "cancer_type", "acronym")
        platform_col = _resolve_column(reader.fieldnames, "platform")
        dnu_col = _resolve_column(reader.fieldnames, "Do_not_use", "Do not use", "do_not_use")

        for line_number, row in enumerate(reader, start=2):
            aliquot = (row.get(aliquot_col) or "").strip().upper()
            if not aliquot:
                continue
            try:
                parsed = parse_tcga_barcode(aliquot)
            except ValueError as exc:
                raise ValueError(f"invalid aliquot barcode at {path}:{line_number}: {aliquot!r}") from exc
            if parsed.sample is None:
                raise ValueError(f"quality annotation lacks a sample-level barcode at {path}:{line_number}")
            records.append(
                QualityRecord(
                    aliquot_barcode=aliquot,
                    sample_barcode=parsed.sample,
                    participant_barcode=parsed.participant,
                    cancer_type=(row.get(cancer_col) or "").strip().upper(),
                    platform=(row.get(platform_col) or "").strip(),
                    do_not_use=_parse_bool(row.get(dnu_col) or ""),
                )
            )
    return records


def read_expression_barcodes(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            fields = next(csv.reader(handle, delimiter="\t"))
        except StopIteration:
            fields = []
    if not fields:
        raise ValueError("expression TSV is empty")
    if len(fields) < 2:
        raise ValueError("expression TSV header must contain gene_id and at least one sample")
    if _normalise_column(fields[0]) != "gene_id":
        raise ValueError(f"expected first expression column to be gene_id, found {fields[0]!r}")
    barcodes = [field.strip().upper() for field in fields[1:]]
    if len(set(barcodes)) != len(barcodes):
        raise ValueError("expression matrix contains duplicate column barcodes")
    for barcode in barcodes:
        parse_tcga_barcode(barcode)
    return barcodes


def _quality_index(records: Iterable[QualityRecord], platform: str) -> dict[str, QualityRecord]:
    index: dict[str, QualityRecord] = {}
    for record in records:
        if record.platform != platform:
            continue
        previous = index.get(record.aliquot_barcode)
        if previous is not None and previous != record:
            raise ValueError(f"conflicting quality rows for aliquot {record.aliquot_barcode}")
        index[record.aliquot_barcode] = record
    return index


def build_cohort(
    expression_barcodes: Sequence[str],
    quality_records: Iterable[QualityRecord],
    *,
    target_cancers: Iterable[str] = TARGET_CANCERS,
    platform: str = RNA_PLATFORM,
    primary_sample_code: str = PRIMARY_SOLID_TUMOR_CODE,
) -> CohortResult:
    targets = {value.upper() for value in target_cancers}
    quality = _quality_index(quality_records, platform)
    exclusions = {
        "unmatched_quality": 0,
        "do_not_use": 0,
        "non_primary": 0,
        "non_target_cancer": 0,
        "duplicate_participant": 0,
    }
    candidates: list[CohortSample] = []

    for matrix_index, raw_barcode in enumerate(expression_barcodes):
        barcode = raw_barcode.strip().upper()
        parsed = parse_tcga_barcode(barcode)
        record = quality.get(barcode)
        if record is None:
            exclusions["unmatched_quality"] += 1
            continue
        if record.do_not_use:
            exclusions["do_not_use"] += 1
            continue
        if parsed.sample_type_code != primary_sample_code:
            exclusions["non_primary"] += 1
            continue
        if record.cancer_type not in targets:
            exclusions["non_target_cancer"] += 1
            continue
        if parsed.sample is None:
            exclusions["non_primary"] += 1
            continue
        candidates.append(
            CohortSample(
                matrix_index=matrix_index,
                expression_barcode=barcode,
                sample_barcode=parsed.sample,
                participant_barcode=parsed.participant,
                sample_type_code=parsed.sample_type_code,
                cancer_type=record.cancer_type,
            )
        )

    # One expression profile per participant. Selection is deterministic and does not
    # depend on labels or expression values: lexical aliquot barcode, then matrix index.
    by_participant: dict[str, list[CohortSample]] = {}
    for sample in candidates:
        by_participant.setdefault(sample.participant_barcode, []).append(sample)

    selected: list[CohortSample] = []
    for participant in sorted(by_participant):
        choices = sorted(
            by_participant[participant],
            key=lambda item: (item.expression_barcode, item.matrix_index),
        )
        selected.append(choices[0])
        exclusions["duplicate_participant"] += len(choices) - 1

    selected.sort(key=lambda item: item.matrix_index)
    return CohortResult(
        samples=tuple(selected),
        exclusion_counts=exclusions,
        n_expression_samples=len(expression_barcodes),
        n_selected=len(selected),
    )


def write_cohort_outputs(result: CohortResult, outdir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "cohort.tsv"
    summary = output_dir / "cohort_summary.json"

    fieldnames = [
        "matrix_index",
        "expression_barcode",
        "sample_barcode",
        "participant_barcode",
        "sample_type_code",
        "cancer_type",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sample in result.samples:
            writer.writerow(asdict(sample))

    payload = {
        "n_expression_samples": result.n_expression_samples,
        "n_selected": result.n_selected,
        "exclusion_counts": result.exclusion_counts,
    }
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, summary
