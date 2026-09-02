import csv
import io
import json

import numpy as np
import pytest

from tcga_ml.feature_budget import (
    AdaptiveSelectKBest,
    _selected_original_indices,
    build_feature_budget_pipeline,
    read_gene_table,
    run_feature_budget,
)


def _write_gene_table(path, n_features: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["gene_index", "source_gene_id", "symbol", "entrez_id"])
        for index in range(n_features):
            writer.writerow([index, f"G{index}|{index}", f"G{index}", index])


def _write_split(path, labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["cache_index", "participant_barcode", "cancer_type", "split"]
        )
        for index, label in enumerate(labels):
            writer.writerow(
                [
                    index,
                    f"TCGA-AA-{index:04d}",
                    label,
                    "holdout" if index % 5 == 0 else "development",
                ]
            )


def _synthetic_dataset(tmp_path) -> None:
    rng = np.random.default_rng(4)
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(["BRCA", "LUAD", "LUSC"]):
        block = rng.lognormal(0.3, 0.15, size=(20, 10)).astype("float32")
        block[:, class_index * 2 : class_index * 2 + 2] *= 8
        block[:, 9] = 3.0
        blocks.append(block)
        labels.extend([label] * 20)

    np.save(tmp_path / "x.npy", np.vstack(blocks))
    _write_split(tmp_path / "split.tsv", labels)
    _write_gene_table(tmp_path / "genes.tsv", 10)


def test_feature_selection_happens_before_scaling():
    pipeline = build_feature_budget_pipeline(
        "elastic_net",
        5,
        negative_policy="clip",
    )
    names = [name for name, _ in pipeline.steps]
    assert names.index("variance") < names.index("select")
    assert names.index("select") < names.index("scale") < names.index("model")


def test_elastic_net_primary_path_fits_without_future_warning():
    X = np.array(
        [
            [8, 7, 1, 1],
            [7, 8, 1, 1],
            [9, 7, 1, 1],
            [1, 1, 8, 7],
            [1, 1, 7, 8],
            [1, 1, 9, 7],
            [5, 1, 5, 1],
            [6, 1, 6, 1],
            [5, 2, 6, 1],
        ],
        dtype="float32",
    )
    y = np.array(["BRCA"] * 3 + ["LUAD"] * 3 + ["LUSC"] * 3)
    pipeline = build_feature_budget_pipeline(
        "elastic_net",
        2,
        negative_policy="clip",
    )
    pipeline.fit(X, y)
    assert len(_selected_original_indices(pipeline)) == 2


def test_gene_table_requires_matrix_width(tmp_path):
    path = tmp_path / "genes.tsv"
    _write_gene_table(path, 3)
    with pytest.raises(ValueError, match="matrix has 4 features"):
        read_gene_table(path, expected_features=4)


def test_adaptive_selector_caps_k_after_prior_filtering():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.array([0, 1] * 5)
    selector = AdaptiveSelectKBest(10).fit(X, y)
    assert selector.actual_k_ == 3
    assert selector.transform(X).shape[1] == 3


def test_selected_indices_map_through_variance_filter():
    X = np.array(
        [
            [1.0, 5, 0.0],
            [2.0, 5, 0.2],
            [8.0, 5, 0.8],
            [9.0, 5, 1.0],
            [1.4, 5, 0.1],
            [8.6, 5, 0.9],
        ],
        dtype="float32",
    )
    y = np.array(["A", "A", "B", "B", "A", "B"])
    pipeline = build_feature_budget_pipeline(
        "logistic_l2",
        1,
        negative_policy="clip",
    ).fit(X, y)
    selected = _selected_original_indices(pipeline)
    assert len(selected) == 1
    assert int(selected[0]) in {0, 2}
    assert int(selected[0]) != 1


def test_run_feature_budget_is_dev_only_and_writes_interpretation(tmp_path):
    _synthetic_dataset(tmp_path)
    output = tmp_path / "out"
    payload = run_feature_budget(
        tmp_path / "x.npy",
        tmp_path / "split.tsv",
        tmp_path / "genes.tsv",
        output,
        model_name="logistic_l2",
        gene_budgets=[2, 4, "all"],
        cv_folds=3,
        n_jobs=1,
        negative_policy="clip",
    )

    assert payload["holdout_used"] is False
    assert [row["gene_budget"] for row in payload["budgets"]] == [2, 4, "all"]
    assert all(
        row["metrics"]["macro_f1"]["mean"] > 0.7
        for row in payload["budgets"]
    )
    assert (output / "feature_stability.tsv").is_file()
    assert (output / "coefficients.tsv").is_file()

    with (output / "feature_stability.tsv").open(encoding="utf-8") as handle:
        stability = list(csv.DictReader(handle, delimiter="\t"))
    assert stability
    assert max(float(row["selection_frequency"]) for row in stability) == 1.0

    saved = json.loads((output / "feature_budget.json").read_text())
    assert saved["evaluation_scope"] == "development_cross_validation_only"


def test_run_feature_budget_reports_parallel_fold_progress(tmp_path):
    _synthetic_dataset(tmp_path)
    progress = io.StringIO()
    run_feature_budget(
        tmp_path / "x.npy",
        tmp_path / "split.tsv",
        tmp_path / "genes.tsv",
        tmp_path / "out",
        model_name="logistic_l2",
        gene_budgets=[2, "all"],
        cv_folds=3,
        n_jobs=2,
        negative_policy="clip",
        show_progress=True,
        progress_stream=progress,
        progress_heartbeat_seconds=0,
    )

    output = progress.getvalue()
    assert "budget 1/2 (2 genes)" in output
    assert "budget 2/2 (all genes)" in output
    assert "folds 1/3" in output or "folds 2/3" in output
    assert "folds 3/3" in output
    assert output.count("complete") == 2
