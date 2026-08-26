from __future__ import annotations

from dataclasses import dataclass
import re


_PARTICIPANT_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$", re.IGNORECASE)
_SAMPLE_RE = re.compile(r"^(?P<code>\d{2})(?P<vial>[A-Z])?$", re.IGNORECASE)


@dataclass(frozen=True)
class TCGABarcode:
    raw: str
    participant: str
    sample: str | None
    sample_type_code: str | None
    vial: str | None

    @property
    def is_primary_solid_tumor(self) -> bool:
        return self.sample_type_code == "01"


def parse_tcga_barcode(value: str) -> TCGABarcode:
    raw = value.strip().upper()
    fields = raw.split("-")
    if len(fields) < 3:
        raise ValueError(f"not a TCGA barcode: {value!r}")

    participant = "-".join(fields[:3])
    if not _PARTICIPANT_RE.fullmatch(participant):
        raise ValueError(f"invalid TCGA participant prefix: {value!r}")

    if len(fields) == 3:
        return TCGABarcode(raw=raw, participant=participant, sample=None, sample_type_code=None, vial=None)

    match = _SAMPLE_RE.fullmatch(fields[3])
    if match is None:
        raise ValueError(f"invalid TCGA sample field in barcode: {value!r}")

    code = match.group("code")
    vial = match.group("vial")
    sample = f"{participant}-{code}"
    return TCGABarcode(
        raw=raw,
        participant=participant,
        sample=sample,
        sample_type_code=code,
        vial=vial,
    )


def participant_barcode(value: str) -> str:
    return parse_tcga_barcode(value).participant


def sample_barcode(value: str) -> str:
    parsed = parse_tcga_barcode(value)
    if parsed.sample is None:
        raise ValueError(f"barcode has no sample component: {value!r}")
    return parsed.sample
