import hashlib
import json

from tcga_ml.provenance import build_provenance_record, sha256_file, write_provenance


def test_sha256_and_record(tmp_path):
    payload = b"tcga\n"
    source = tmp_path / "source.tsv"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(source) == expected

    record = build_provenance_record(
        source,
        source_url="https://example.test/source.tsv",
        source_key="fixture",
        retrieved_at="2026-08-26T00:00:00+00:00",
    )
    assert record["sha256"] == expected
    assert record["size_bytes"] == len(payload)

    output = tmp_path / "source.tsv.provenance.json"
    write_provenance(output, record)
    assert json.loads(output.read_text()) == record
