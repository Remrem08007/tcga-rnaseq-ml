import csv
import io
import json

import numpy as np

from tcga_ml.benchmark import resolve_n_jobs, run_benchmark
from tcga_ml.models import MODEL_NAMES, build_model_pipeline


def _write_synthetic_case(tmp_path):
    rng = np.random.default_rng(7)
    n_per_class = 30
    n_features = 12
    classes = ["BRCA", "LUAD", "LUSC"]
    blocks = []
    labels = []
    for class_index, label in enumerate(classes):
        X = rng.lognormal(mean=0.5, sigma=0.25, size=(n_per_class, n_features)).astype(np.float32)
        X[:, class_index * 3 : class_index * 3 + 3] *= np.float32(8.0)
        blocks.append(X)
        labels.extend([label] * n_per_class)
    matrix = np.vstack(blocks)
    matrix_path = tmp_path / "expression.float32.npy"
    np.save(matrix_path, matrix)

    split_path = tmp_path / "split_manifest.tsv"
    with split_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["cache_index", "participant_barcode", "cancer_type", "split"])
        for index, label in enumerate(labels):
            split = "holdout" if index % 5 == 0 else "development"
            writer.writerow([index, f"TCGA-AA-{index:04d}", label, split])
    return matrix_path, split_path


def test_model_registry_builds_all_pipelines():
    for name in MODEL_NAMES:
        pipeline = build_model_pipeline(name, pca_components=3)
        assert pipeline.steps[-1][0] == "model"


def test_resolve_n_jobs_caps_to_cv(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    assert resolve_n_jobs(0, cv_folds=5) == 5
    assert resolve_n_jobs(2, cv_folds=5) == 2


def test_benchmark_uses_development_only_and_linear_model_beats_dummy(tmp_path):
    matrix, split = _write_synthetic_case(tmp_path)
    payload = run_benchmark(
        matrix,
        split,
        tmp_path / "results",
        models=["dummy", "logistic_l2"],
        cv_folds=3,
        n_jobs=1,
        negative_policy="error",
        seed=11,
    )
    assert payload["holdout_used"] is False
    assert payload["evaluation_scope"] == "development_cross_validation_only"
    assert payload["n_development_samples"] == 72
    scores = {row["model"]: row["metrics"]["macro_f1"]["mean"] for row in payload["models"]}
    assert scores["logistic_l2"] > scores["dummy"] + 0.4
    saved = json.loads((tmp_path / "results" / "benchmark.json").read_text())
    assert saved["holdout_used"] is False


def test_benchmark_reports_model_and_sequential_fold_progress(tmp_path):
    matrix, split = _write_synthetic_case(tmp_path)
    progress = io.StringIO()
    run_benchmark(
        matrix,
        split,
        tmp_path / "results",
        models=["dummy", "logistic_l2"],
        cv_folds=3,
        n_jobs=1,
        show_progress=True,
        progress_stream=progress,
        progress_heartbeat_seconds=0,
    )

    output = progress.getvalue()
    assert "model 1/2 (dummy)" in output
    assert "model 2/2 (logistic_l2)" in output
    assert "folds 1/3" in output
    assert "folds 3/3" in output
    assert output.count("complete") == 2
