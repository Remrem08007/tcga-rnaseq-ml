from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from tcga_ml.final_evaluation import (
    DEVELOPMENT_SCOPE,
    FinalEvaluationLockError,
    create_final_evaluation_lock,
    load_final_evaluation_lock,
    run_final_evaluation,
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


def _write_evaluation_inputs(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path, np.ndarray, list[int], list[int]]:
    rng = np.random.default_rng(43)
    matrix_rows: list[np.ndarray] = []
    manifest: list[list[object]] = []
    development_indices: list[int] = []
    holdout_indices: list[int] = []
    labels = ("BRCA", "LUAD", "LUSC")
    cache_index = 0

    for class_index, label in enumerate(labels):
        block = rng.lognormal(0.3, 0.12, size=(6, 8)).astype(np.float32)
        block[:, class_index] *= np.float32(8.0)
        block[4:, :] += np.float32(25.0)
        matrix_rows.append(block)
        for local_index in range(6):
            split = "development" if local_index < 4 else "holdout"
            manifest.append(
                [
                    cache_index,
                    f"TCGA-{class_index:02d}-{local_index:04d}",
                    label,
                    split,
                ]
            )
            if split == "development":
                development_indices.append(cache_index)
            else:
                holdout_indices.append(cache_index)
            cache_index += 1

    matrix = np.vstack(matrix_rows)
    matrix_path = root / "x.npy"
    split_path = root / "split.tsv"
    genes_path = root / "genes.tsv"
    evidence_path = root / "development.json"
    config_path = root / "selection.json"
    np.save(matrix_path, matrix)

    with split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["cache_index", "participant_barcode", "cancer_type", "split"]
        )
        writer.writerows(manifest)

    with genes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["gene_index", "source_gene_id", "symbol", "entrez_id"])
        for index in range(matrix.shape[1]):
            writer.writerow([index, f"G{index}|{index}", f"G{index}", index])

    evidence_path.write_text(
        json.dumps(
            {
                "evaluation_scope": DEVELOPMENT_SCOPE,
                "holdout_used": False,
                "models": [{"model": "logistic_l2", "macro_f1": 0.8}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "candidate_id": "logistic-l2-four-genes",
                "primary_metric": "macro_f1",
                "selection_rationale": (
                    "Synthetic fixture choice based only on development CV."
                ),
                "pipeline": {
                    "family": "linear_gene_budget",
                    "model": "logistic_l2",
                    "gene_budget": 4,
                    "negative_policy": "error",
                    "scaler": "standard",
                    "seed": 20260825,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        config_path,
        matrix_path,
        split_path,
        genes_path,
        evidence_path,
        matrix,
        development_indices,
        holdout_indices,
    )


def test_final_evaluation_is_receipt_guarded_and_fits_on_development_only(
    tmp_path: Path,
) -> None:
    (
        config,
        matrix_path,
        split,
        genes,
        evidence,
        matrix,
        development_indices,
        holdout_indices,
    ) = _write_evaluation_inputs(tmp_path)
    lock = tmp_path / "final.lock.json"
    receipt = tmp_path / "final.receipt.json"
    outdir = tmp_path / "final-output"
    create_final_evaluation_lock(
        config,
        matrix_path,
        split,
        genes,
        [evidence],
        lock,
    )

    payload = run_final_evaluation(
        lock,
        matrix_path,
        split,
        genes,
        outdir,
        receipt,
        evidence_paths=[evidence],
    )
    assert payload["evaluation_scope"] == "locked_final_holdout_once"
    assert payload["holdout_used"] is True
    assert payload["n_development_samples"] == len(development_indices)
    assert payload["n_holdout_samples"] == len(holdout_indices)
    assert payload["primary_metric"] == "macro_f1"
    assert (outdir / "final_metrics.json").is_file()
    assert (outdir / "final_predictions.tsv").is_file()
    assert (outdir / "final_confusion.tsv").is_file()
    assert (outdir / "final_pipeline.joblib").is_file()
    assert (outdir / "final_confusion.svg").read_text().startswith("<svg")
    assert (outdir / "final_per_class_f1.svg").read_text().startswith("<svg")

    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["status"] == "completed"
    assert receipt_payload["holdout_used"] is True
    assert len(receipt_payload["outputs"]) == 6

    prediction_rows = list(
        csv.DictReader(
            (outdir / "final_predictions.tsv").open(
                encoding="utf-8",
                newline="",
            ),
            delimiter="\t",
        )
    )
    assert len(prediction_rows) == len(holdout_indices)
    assert all(
        row["participant_barcode"].endswith(("0004", "0005"))
        for row in prediction_rows
    )

    estimator = joblib.load(outdir / "final_pipeline.joblib")
    variance_indices = estimator.named_steps["variance"].get_support(indices=True)
    selected_after_variance = estimator.named_steps["select"].get_support(
        indices=True
    )
    selected_indices = np.asarray(variance_indices)[selected_after_variance]
    transformed = np.log2(matrix + np.float32(1.0))
    development_mean = transformed[development_indices][:, selected_indices].mean(
        axis=0
    )
    whole_data_mean = transformed[:, selected_indices].mean(axis=0)
    np.testing.assert_allclose(
        estimator.named_steps["scale"].mean_,
        development_mean,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.allclose(development_mean, whole_data_mean)

    with pytest.raises(FinalEvaluationLockError, match="already has a receipt"):
        run_final_evaluation(
            lock,
            matrix_path,
            split,
            genes,
            tmp_path / "second-output",
            receipt,
        )


def test_final_evaluation_refuses_existing_output_before_receipt(
    tmp_path: Path,
) -> None:
    config, matrix, split, genes, evidence, _, _, _ = _write_evaluation_inputs(
        tmp_path
    )
    lock = tmp_path / "final.lock.json"
    receipt = tmp_path / "final.receipt.json"
    outdir = tmp_path / "existing"
    outdir.mkdir()
    create_final_evaluation_lock(config, matrix, split, genes, [evidence], lock)

    with pytest.raises(FinalEvaluationLockError, match="existing final output"):
        run_final_evaluation(
            lock,
            matrix,
            split,
            genes,
            outdir,
            receipt,
        )
    assert not receipt.exists()
