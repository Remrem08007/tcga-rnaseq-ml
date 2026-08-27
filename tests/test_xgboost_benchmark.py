from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tcga_ml.xgboost_benchmark import (
    BalancedXGBClassifier,
    build_xgboost_pipeline,
    evaluate_xgboost_cv,
    probe_cuda,
    resolve_device,
    resolve_threads,
    run_compute_scaling,
    run_xgboost_benchmark,
)


def _synthetic_expression() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(("BRCA", "LUAD", "LUSC")):
        block = rng.lognormal(0.4, 0.18, size=(12, 9)).astype(np.float32)
        block[:, class_index * 3 : class_index * 3 + 3] *= np.float32(6.0)
        blocks.append(block)
        labels.extend([label] * len(block))
    return np.vstack(blocks), np.asarray(labels, dtype=object)


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    X, y = _synthetic_expression()
    matrix_path = root / "x.npy"
    split_path = root / "split.tsv"
    genes_path = root / "genes.tsv"
    np.save(matrix_path, X)

    with split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["cache_index", "participant_barcode", "cancer_type", "split"])
        class_seen = {"BRCA": 0, "LUAD": 0, "LUSC": 0}
        for index, label in enumerate(y):
            local_index = class_seen[str(label)]
            class_seen[str(label)] += 1
            split = "holdout" if local_index >= 10 else "development"
            writer.writerow([index, f"TCGA-AA-{index:04d}", label, split])

    with genes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["gene_index", "source_gene_id", "symbol", "entrez_id"])
        for index in range(X.shape[1]):
            writer.writerow([index, f"G{index}|{index}", f"G{index}", str(index)])
    return matrix_path, split_path, genes_path


def test_cpu_resolution_and_thread_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert resolve_device("cpu") == ("cpu", None)
    assert resolve_threads(0) == 4
    assert resolve_threads(2) == 2
    with pytest.raises(ValueError, match="threads must be"):
        resolve_threads(-1)


def test_pipeline_keeps_learned_selection_inside_training_fold() -> None:
    pipeline = build_xgboost_pipeline(device="cpu", threads=1, gene_budget=4)
    assert [name for name, _ in pipeline.steps] == [
        "log2p1",
        "impute",
        "variance",
        "select",
        "model",
    ]


def test_balanced_xgboost_classifier_fits_multiclass_cpu() -> None:
    X, y = _synthetic_expression()
    model = BalancedXGBClassifier(
        device="cpu",
        n_estimators=8,
        max_depth=2,
        learning_rate=0.2,
        n_jobs=1,
        random_state=3,
    ).fit(X, y)
    prediction = model.predict(X[:6])
    assert prediction.shape == (6,)
    assert set(prediction).issubset(set(y))
    assert model.resolved_device_ == "cpu"
    assert model.feature_importances_.shape == (X.shape[1],)


def test_development_cv_runs_xgboost_without_cpu_oversubscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    X, y = _synthetic_expression()
    summary, folds = evaluate_xgboost_cv(
        X,
        y,
        requested_device="cpu",
        threads=1,
        fold_jobs=2,
        gene_budget=4,
        cv_folds=2,
        n_estimators=6,
        max_depth=2,
        learning_rate=0.2,
    )
    assert summary["resolved_device"] == "cpu"
    assert summary["threads"] == 1
    assert summary["fold_jobs"] == 2
    assert len(summary["metrics"]["macro_f1"]["fold_values"]) == 2
    assert len(folds) == 2
    assert all(len(fold["gene_indices"]) == 4 for fold in folds)

    with pytest.raises(ValueError, match="oversubscription"):
        evaluate_xgboost_cv(
            X,
            y,
            requested_device="cpu",
            threads=2,
            fold_jobs=2,
            gene_budget=4,
            cv_folds=2,
            n_estimators=2,
            max_depth=2,
        )


def test_xgboost_benchmark_and_scaling_outputs_are_development_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    matrix_path, split_path, genes_path = _write_fixture(tmp_path)

    benchmark_dir = tmp_path / "benchmark"
    payload = run_xgboost_benchmark(
        matrix_path,
        split_path,
        genes_path,
        benchmark_dir,
        requested_device="cpu",
        threads=1,
        fold_jobs=1,
        gene_budget=4,
        cv_folds=2,
        n_estimators=6,
        max_depth=2,
        learning_rate=0.2,
    )
    assert payload["holdout_used"] is False
    assert payload["n_development_samples"] == 30
    assert (benchmark_dir / "xgboost_benchmark.json").is_file()
    assert (benchmark_dir / "xgboost_feature_importance.tsv").is_file()

    scaling_dir = tmp_path / "scaling"
    scaling = run_compute_scaling(
        matrix_path,
        split_path,
        scaling_dir,
        cpu_threads=[1, 2],
        gene_budget=4,
        cv_folds=2,
        n_estimators=4,
        max_depth=2,
        learning_rate=0.2,
    )
    assert scaling["holdout_used"] is False
    assert [run["threads"] for run in scaling["runs"]] == [1, 2]
    assert all(run["device"] == "cpu" for run in scaling["runs"])
    assert (scaling_dir / "compute_scaling.json").is_file()
    assert (scaling_dir / "compute_scaling.tsv").is_file()


def test_cuda_probe_never_reports_cpu_fallback_as_cuda() -> None:
    probe_cuda.cache_clear()
    probe = probe_cuda()
    if probe.available:
        assert probe.resolved is not None
        assert probe.resolved.startswith("cuda")
        assert resolve_device("cuda")[0] == "cuda"
    else:
        assert probe.resolved is None or not probe.resolved.startswith("cuda")
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")
