from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .provenance import sha256_file


@dataclass(frozen=True)
class ExpressionScan:
    barcodes: tuple[str, ...]
    n_genes: int
    sha256: str


@dataclass(frozen=True)
class CohortIndex:
    matrix_index: int
    expression_barcode: str
    participant_barcode: str
    cancer_type: str


def scan_expression_tsv(path: str | Path) -> ExpressionScan:
    digest = hashlib.sha256()
    n_genes = 0
    with Path(path).open("rb") as handle:
        header_bytes = handle.readline()
        if not header_bytes:
            raise ValueError("expression TSV is empty")
        digest.update(header_bytes)
        try:
            header = header_bytes.decode("utf-8-sig").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise ValueError("expression TSV header is not UTF-8") from exc
        fields = header.split("\t")
        if len(fields) < 2 or fields[0].strip().lower() != "gene_id":
            raise ValueError("expression TSV must start with a gene_id column")
        barcodes = tuple(field.strip().upper() for field in fields[1:])
        if len(set(barcodes)) != len(barcodes):
            raise ValueError("expression matrix contains duplicate column barcodes")

        for line in handle:
            digest.update(line)
            if line.strip():
                n_genes += 1

    if n_genes == 0:
        raise ValueError("expression TSV contains no gene rows")
    return ExpressionScan(barcodes=barcodes, n_genes=n_genes, sha256=digest.hexdigest())


def read_cohort_manifest(path: str | Path) -> list[CohortIndex]:
    rows: list[CohortIndex] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"matrix_index", "expression_barcode", "participant_barcode", "cancer_type"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"cohort manifest must contain columns: {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                matrix_index = int(row["matrix_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid matrix_index at {path}:{line_number}") from exc
            rows.append(
                CohortIndex(
                    matrix_index=matrix_index,
                    expression_barcode=row["expression_barcode"].strip().upper(),
                    participant_barcode=row["participant_barcode"].strip().upper(),
                    cancer_type=row["cancer_type"].strip().upper(),
                )
            )
    if not rows:
        raise ValueError("cohort manifest contains no selected samples")
    indices = [row.matrix_index for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError("cohort manifest contains duplicate matrix indices")
    participants = [row.participant_barcode for row in rows]
    if len(set(participants)) != len(participants):
        raise ValueError("cohort manifest contains duplicate participants")
    return rows


def _validate_cohort_against_scan(cohort: Sequence[CohortIndex], scan: ExpressionScan) -> None:
    for row in cohort:
        if row.matrix_index < 0 or row.matrix_index >= len(scan.barcodes):
            raise ValueError(f"cohort matrix index out of range: {row.matrix_index}")
        matrix_barcode = scan.barcodes[row.matrix_index]
        if matrix_barcode != row.expression_barcode:
            raise ValueError(
                f"cohort barcode/index mismatch at index {row.matrix_index}: "
                f"manifest={row.expression_barcode} matrix={matrix_barcode}"
            )


def parse_gene_identifier(value: str) -> tuple[str, str]:
    source = value.strip()
    if "|" not in source:
        return source, ""
    symbol, entrez = source.split("|", 1)
    return symbol.strip(), entrez.strip()


def _parse_values(payload: str, expected: int) -> np.ndarray:
    normalised = payload.replace("NA", "nan")
    values = np.fromstring(normalised, sep="\t", dtype=np.float32)
    if values.size == expected:
        return values

    tokens = payload.split("\t")
    if len(tokens) != expected:
        raise ValueError(f"expression row has {len(tokens)} values; expected {expected}")
    parsed: list[float] = []
    for token in tokens:
        text = token.strip()
        if text.lower() in {"", "na", "nan"}:
            parsed.append(float("nan"))
        else:
            parsed.append(float(text))
    return np.asarray(parsed, dtype=np.float32)


def build_expression_cache(
    expression_path: str | Path,
    cohort_manifest: str | Path,
    outdir: str | Path,
    *,
    chunk_genes: int = 256,
) -> dict[str, object]:
    if chunk_genes < 1:
        raise ValueError("chunk_genes must be >= 1")

    expression = Path(expression_path)
    cohort_path = Path(cohort_manifest)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan = scan_expression_tsv(expression)
    cohort = read_cohort_manifest(cohort_path)
    _validate_cohort_against_scan(cohort, scan)

    selected_indices = np.asarray([row.matrix_index for row in cohort], dtype=np.int64)
    n_samples = len(cohort)
    n_genes = scan.n_genes

    matrix_path = output_dir / "expression.float32.npy"
    genes_path = output_dir / "genes.tsv"
    samples_path = output_dir / "samples.tsv"
    metadata_path = output_dir / "cache_metadata.json"

    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_genes),
        fortran_order=False,
    )

    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["cache_index", "matrix_index", "expression_barcode", "participant_barcode", "cancer_type"])
        for cache_index, row in enumerate(cohort):
            writer.writerow([cache_index, row.matrix_index, row.expression_barcode, row.participant_barcode, row.cancer_type])

    missing_values = 0
    gene_index = 0
    pending_values: list[np.ndarray] = []
    pending_genes: list[tuple[int, str, str, str]] = []

    def flush_block() -> None:
        nonlocal pending_values, pending_genes
        if not pending_values:
            return
        start = pending_genes[0][0]
        stop = pending_genes[-1][0] + 1
        block = np.stack(pending_values, axis=1)
        matrix[:, start:stop] = block
        pending_values = []
        pending_genes = []

    with expression.open("r", encoding="utf-8-sig") as handle, genes_path.open("w", encoding="utf-8", newline="") as gene_handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        if tuple(field.strip().upper() for field in header[1:]) != scan.barcodes:
            raise ValueError("expression header changed between scan and cache pass")
        gene_writer = csv.writer(gene_handle, delimiter="\t", lineterminator="\n")
        gene_writer.writerow(["gene_index", "source_gene_id", "symbol", "entrez_id"])

        for line_number, line in enumerate(handle, start=2):
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            try:
                source_gene_id, payload = stripped.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"malformed expression row at line {line_number}") from exc
            all_values = _parse_values(payload, len(scan.barcodes))
            selected = all_values[selected_indices]
            missing_values += int(np.count_nonzero(~np.isfinite(selected)))
            symbol, entrez = parse_gene_identifier(source_gene_id)
            pending_values.append(selected)
            pending_genes.append((gene_index, source_gene_id, symbol, entrez))
            gene_writer.writerow([gene_index, source_gene_id, symbol, entrez])
            gene_index += 1
            if len(pending_values) >= chunk_genes:
                flush_block()

        flush_block()

    if gene_index != n_genes:
        raise ValueError(f"expression row count changed between passes: scan={n_genes} cache={gene_index}")
    matrix.flush()
    del matrix

    metadata: dict[str, object] = {
        "format_version": 1,
        "matrix_file": matrix_path.name,
        "samples_file": samples_path.name,
        "genes_file": genes_path.name,
        "dtype": "float32",
        "shape": [n_samples, n_genes],
        "n_samples": n_samples,
        "n_genes": n_genes,
        "n_nonfinite_values": missing_values,
        "expression_source": str(expression),
        "expression_sha256": scan.sha256,
        "cohort_manifest": str(cohort_path),
        "cohort_sha256": sha256_file(cohort_path),
        "transformation": "none; source values preserved for downstream audit/normalization",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
