from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from tcga_ml.feature_budget import GeneRecord
from tcga_ml.focused_pairs import (
    FOCUSED_PAIRS,
    evaluate_focused_pair,
    parse_pair_name,
    run_focused_pair_studies,
)


def _write_fixture(root: Path) -> tuple[Path, Path, Path, set[str]]:
    rng = np.random.default_rng(29)
    labels = ("LUAD", "LUSC", "KIRC", "KIRP", "BRCA")
    blocks: list[np.ndarray] = []
    rows: list[tuple[str, str, str]] = []
    holdout_participants: set[str] = set()
    cache_index = 0

    for class_index, label in enumerate(labels):
        n_samples = 10 if label != "BRCA" else 6
        block = rng.lognormal(0.3, 0.18, size=(n_samples, 10)).astype(np.float32)
        if label == "LUAD":
            block[:, 0] *= np.float32(7.0)
        elif label == "LUSC":
            block[:, 1] *= np.float32(7.0)
        elif label == "KIRC":
            block[:, 2] *= np.float32(7.0)
        elif label == "KIRP":
            block[:, 3] *= np.float32(7.0)
        else:
            block[:, 4] *= np.float32(7.0)
        blocks.append(block)

        for local_index in range(n_samples):
            is_holdout = local_index >= 8 if label != "BRCA" else local_index >= 4
            participant = (
                f"HOLDOUT-{label}-{local_index:02d}"
                if is_holdout
                else f"TCGA-{label[:2]}-{local_index:04d}"
            )
            if is_holdout:
                holdout_participants.add(participant)
            rows.append(
                (
                    str(cache_index),
                    participant,
                    label,
                    "holdout" if is_holdout else "development",
                )
            )
            cache_index += 1

    matrix_path = root / "x.npy"
    split_path = root / "split.tsv"
    genes_path = root / "genes.tsv"
    np.save(matrix_path, np.vstack(blocks))

    with split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["cache_index", "participant_barcode", "cancer_type", "split"])
        writer.writerows(rows)

    with genes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["gene_index", "source_gene_id", "symbol", "entrez_id"])
        for index in range(10):
            writer.writerow([index, f"G{index}|{index}", f"G{index}", str(index)])

    return matrix_path, split_path, genes_path, holdout_participants


def test_pair_names_are_locked_and_normalized() -> None:
    assert parse_pair_name("LUAD-LUSC") == "luad_lusc"
    assert parse_pair_name("kirc_kirp") == "kirc_kirp"
    assert FOCUSED_PAIRS["luad_lusc"] == ("LUAD", "LUSC")
    with pytest.raises(ValueError, match="unknown focused pair"):
        parse_pair_name("BRCA-LUAD")


def test_focused_pair_studies_use_development_samples_only(tmp_path: Path) -> None:
    matrix, split, genes, holdout = _write_fixture(tmp_path)
    outdir = tmp_path / "out"
    payload = run_focused_pair_studies(
        matrix,
        split,
        genes,
        outdir,
        models=("logistic_l2",),
        gene_budget=4,
        cv_folds=2,
        n_jobs=1,
    )

    assert payload["evaluation_scope"] == "development_cross_validation_only"
    assert payload["holdout_used"] is False
    assert len(payload["studies"]) == 2
    assert all(study["n_samples"] == 16 for study in payload["studies"])
    assert all(sum(sum(row) for row in study["confusion_counts"]) == 16 for study in payload["studies"])

    prediction_text = (outdir / "focused_pair_predictions.tsv").read_text(encoding="utf-8")
    assert not any(participant in prediction_text for participant in holdout)
    assert (outdir / "focused_pairs.json").is_file()
    assert (outdir / "focused_pair_metrics.tsv").is_file()
    assert (outdir / "focused_pair_confusion.tsv").is_file()
    assert (outdir / "focused_pair_errors.tsv").is_file()
    assert (outdir / "focused_pair_genes.tsv").is_file()


def test_pair_outputs_include_class_metrics_and_stable_gene_directions(tmp_path: Path) -> None:
    matrix, split, genes, _ = _write_fixture(tmp_path)
    payload = run_focused_pair_studies(
        matrix,
        split,
        genes,
        tmp_path / "out",
        pairs=("luad_lusc",),
        models=("logistic_l2",),
        gene_budget=4,
        cv_folds=2,
        n_jobs=1,
    )
    study = payload["studies"][0]
    assert {row["class"] for row in study["per_class"]} == {"LUAD", "LUSC"}
    assert 0.0 <= study["metrics"]["macro_f1"] <= 1.0
    assert 0.0 <= study["metrics"]["balanced_accuracy"] <= 1.0
    assert len(study["predictions"]) == 16
    assert study["genes"]
    for row in study["genes"]:
        assert row["direction"] in {"LUAD", "LUSC", "neutral"}
        assert 0.0 <= row["selection_frequency"] <= 1.0
        assert 0.0 <= row["sign_consistency"] <= 1.0


def test_focused_pair_requires_enough_development_examples() -> None:
    X = np.ones((3, 4), dtype=np.float32)
    y = np.asarray(["LUAD", "LUAD", "LUSC"], dtype=object)
    participants = ["A", "B", "C"]
    genes = [GeneRecord(index, f"G{index}|{index}", f"G{index}", str(index)) for index in range(4)]
    with pytest.raises(ValueError, match="at least 2 development samples"):
        evaluate_focused_pair(
            X,
            y,
            participants,
            genes,
            "luad_lusc",
            model_name="logistic_l2",
            gene_budget=2,
            cv_folds=2,
            n_jobs=1,
        )


def test_duplicate_focused_pair_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        run_focused_pair_studies(
            "unused.npy",
            "unused.tsv",
            "unused-genes.tsv",
            "unused-out",
            pairs=("luad_lusc", "luad_lusc"),
            models=("logistic_l2",),
        )


def test_focused_pairs_report_study_and_sequential_fold_progress(tmp_path: Path) -> None:
    matrix, split, genes, _ = _write_fixture(tmp_path)
    progress = io.StringIO()
    run_focused_pair_studies(
        matrix,
        split,
        genes,
        tmp_path / "out",
        models=("logistic_l2",),
        gene_budget=4,
        cv_folds=2,
        n_jobs=1,
        show_progress=True,
        progress_stream=progress,
        progress_heartbeat_seconds=0,
    )

    output = progress.getvalue()
    assert "study 1/2 (luad_lusc, logistic_l2)" in output
    assert "study 2/2 (kirc_kirp, logistic_l2)" in output
    assert output.count("folds 1/2") >= 2
    assert output.count("folds 2/2") >= 2
    assert output.count("complete") == 2
