from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


DEFAULT_SEED = 20260825
DEFAULT_HOLDOUT_FRACTION = 0.20


@dataclass(frozen=True)
class SampleRecord:
    cache_index: int
    matrix_index: int
    expression_barcode: str
    participant_barcode: str
    cancer_type: str


@dataclass(frozen=True)
class SplitRecord:
    cache_index: int
    participant_barcode: str
    cancer_type: str
    split: str


def read_samples_table(path: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cache_index", "matrix_index", "expression_barcode", "participant_barcode", "cancer_type"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"samples table must contain columns: {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                cache_index = int(row["cache_index"])
                matrix_index = int(row["matrix_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid integer index at {path}:{line_number}") from exc
            records.append(
                SampleRecord(
                    cache_index=cache_index,
                    matrix_index=matrix_index,
                    expression_barcode=row["expression_barcode"].strip().upper(),
                    participant_barcode=row["participant_barcode"].strip().upper(),
                    cancer_type=row["cancer_type"].strip().upper(),
                )
            )
    _validate_unique_samples(records)
    return records


def _validate_unique_samples(records: Sequence[SampleRecord]) -> None:
    participants = [record.participant_barcode for record in records]
    if len(set(participants)) != len(participants):
        raise ValueError("samples table contains duplicate participants; patient-level split would be ambiguous")
    cache_indices = [record.cache_index for record in records]
    if len(set(cache_indices)) != len(cache_indices):
        raise ValueError("samples table contains duplicate cache indices")


def make_holdout_split(
    records: Sequence[SampleRecord],
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = DEFAULT_SEED,
) -> list[SplitRecord]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if not records:
        raise ValueError("cannot split an empty sample table")
    _validate_unique_samples(records)

    ordered = sorted(records, key=lambda record: record.participant_barcode)
    labels = np.asarray([record.cancer_type for record in ordered], dtype=object)
    classes, counts = np.unique(labels, return_counts=True)
    too_small = {str(c): int(n) for c, n in zip(classes, counts) if n < 2}
    if too_small:
        raise ValueError(f"each class needs at least two participants for stratification: {too_small}")

    indices = np.arange(len(ordered))
    development, holdout = train_test_split(
        indices,
        test_size=holdout_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    holdout_set = set(int(index) for index in holdout)
    development_set = set(int(index) for index in development)
    if holdout_set & development_set:
        raise AssertionError("internal split error: development and holdout overlap")

    return [
        SplitRecord(
            cache_index=record.cache_index,
            participant_barcode=record.participant_barcode,
            cancer_type=record.cancer_type,
            split="holdout" if index in holdout_set else "development",
        )
        for index, record in enumerate(ordered)
    ]


def split_manifest_sha256(records: Sequence[SplitRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.participant_barcode):
        digest.update(
            f"{record.cache_index}\t{record.participant_barcode}\t{record.cancer_type}\t{record.split}\n".encode("utf-8")
        )
    return digest.hexdigest()


def write_split_outputs(
    records: Sequence[SplitRecord],
    outdir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> tuple[Path, Path]:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.tsv"
    summary_path = output_dir / "split_summary.json"

    ordered = sorted(records, key=lambda item: item.participant_barcode)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["cache_index", "participant_barcode", "cancer_type", "split"])
        for record in ordered:
            writer.writerow([record.cache_index, record.participant_barcode, record.cancer_type, record.split])

    class_counts: dict[str, dict[str, int]] = {}
    for record in ordered:
        class_counts.setdefault(record.cancer_type, {"development": 0, "holdout": 0})[record.split] += 1
    summary = {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "n_samples": len(ordered),
        "n_development": sum(record.split == "development" for record in ordered),
        "n_holdout": sum(record.split == "holdout" for record in ordered),
        "class_counts": class_counts,
        "manifest_sha256": split_manifest_sha256(ordered),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, summary_path


def make_development_cv(*, n_splits: int = 5, seed: int = DEFAULT_SEED) -> StratifiedKFold:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
