from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_provenance_record(
    path: str | Path,
    *,
    source_url: str,
    source_key: str,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    file_path = Path(path)
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    return {
        "source_key": source_key,
        "source_url": source_url,
        "retrieved_at": timestamp,
        "filename": file_path.name,
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def write_provenance(path: str | Path, record: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
