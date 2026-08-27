from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcga_ml.final_evaluation import (
    DEVELOPMENT_SCOPE,
    FinalEvaluationLockError,
    create_final_evaluation_lock,
    load_final_evaluation_lock,
    verify_final_evaluation_lock,
)


def _write_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    matrix = root / "expression.npy"
    split = root / "split.tsv"
    genes = root / "genes.tsv"
    evidence = root / "development.json"
    config = root / "selection.json"

    matrix.write_bytes(b"synthetic-matrix")
    split.write_text(
        "cache_index\tparticipant_barcode\tcancer_type\tsplit\n"
        "0\tTCGA-AA-0001\tLUAD\tdevelopment\n"
        "1\tTCGA-AA-0002\tLUSC\tholdout\n",
        encoding="utf-8",
    )
    genes.write_text(
        "gene_index\tsource_gene_id\tsymbol\tentrez_id\n"
        "0\tG0|0\tG0\t0\n",
        encoding="utf-8",
    )
    evidence.write_text(
        json.dumps(
            {
                "evaluation_scope": DEVELOPMENT_SCOPE,
                "holdout_used": False,
                "models": [{"model": "elastic_net", "macro_f1": 0.9}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config.write_text(
        json.dumps(
            {
                "candidate_id": "elastic-net-1000",
                "primary_metric": "macro_f1",
                "selection_rationale": (
                    "Highest development macro F1 among the stable interpretable candidates."
                ),
                "pipeline": {
                    "family": "linear_gene_budget",
                    "model": "elastic_net",
                    "gene_budget": 1000,
                    "negative_policy": "clip",
                    "scaler": "standard",
                    "seed": 20260825,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return config, matrix, split, genes, evidence


def test_lock_is_self_verifying_and_binds_development_evidence(
    tmp_path: Path,
) -> None:
    config, matrix, split, genes, evidence = _write_inputs(tmp_path)
    lock = tmp_path / "final.lock.json"
    payload = create_final_evaluation_lock(
        config,
        matrix,
        split,
        genes,
        [evidence],
        lock,
    )

    assert payload["holdout_status"] == "sealed"
    assert len(payload["lock_sha256"]) == 64
    assert payload["selection"]["pipeline"]["gene_budget"] == 1000
    assert payload["selection_evidence"][0]["holdout_used"] is False

    loaded = load_final_evaluation_lock(lock)
    verified = verify_final_evaluation_lock(
        lock,
        matrix,
        split,
        genes,
        evidence_paths=[evidence],
    )
    assert loaded["lock_sha256"] == payload["lock_sha256"]
    assert verified["lock_sha256"] == payload["lock_sha256"]


def test_lock_refuses_holdout_touched_selection_evidence(tmp_path: Path) -> None:
    config, matrix, split, genes, evidence = _write_inputs(tmp_path)
    evidence.write_text(
        json.dumps(
            {
                "evaluation_scope": DEVELOPMENT_SCOPE,
                "holdout_used": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FinalEvaluationLockError, match="holdout_used=false"):
        create_final_evaluation_lock(
            config,
            matrix,
            split,
            genes,
            [evidence],
            tmp_path / "final.lock.json",
        )


def test_lock_detects_changed_artifact_and_changed_lock(tmp_path: Path) -> None:
    config, matrix, split, genes, evidence = _write_inputs(tmp_path)
    lock = tmp_path / "final.lock.json"
    create_final_evaluation_lock(config, matrix, split, genes, [evidence], lock)

    split.write_text(split.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(FinalEvaluationLockError, match="split manifest"):
        verify_final_evaluation_lock(lock, matrix, split, genes)

    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["holdout_status"] = "opened"
    lock.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(FinalEvaluationLockError, match="digest mismatch"):
        load_final_evaluation_lock(lock)


def test_lock_refuses_overwrite_and_unknown_pipeline_fields(tmp_path: Path) -> None:
    config, matrix, split, genes, evidence = _write_inputs(tmp_path)
    lock = tmp_path / "final.lock.json"
    create_final_evaluation_lock(config, matrix, split, genes, [evidence], lock)

    with pytest.raises(FinalEvaluationLockError, match="refusing to overwrite"):
        create_final_evaluation_lock(config, matrix, split, genes, [evidence], lock)

    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["pipeline"]["accidental_option"] = True
    config.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(FinalEvaluationLockError, match="unknown pipeline fields"):
        create_final_evaluation_lock(
            config,
            matrix,
            split,
            genes,
            [evidence],
            tmp_path / "other.lock.json",
        )
