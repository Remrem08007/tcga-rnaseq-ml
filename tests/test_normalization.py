import json

import numpy as np
import pytest

from tcga_ml.normalization import (
    PanCancerLog2p1,
    audit_expression_matrix,
    build_standardized_preprocessor,
    write_audit,
)


def test_expression_audit_counts_and_quantiles(tmp_path):
    matrix = np.asarray([[0.0, 1.0, -0.01], [3.0, np.nan, 7.0]], dtype=np.float32)
    path = tmp_path / "x.npy"
    np.save(path, matrix)
    audit = audit_expression_matrix(path, max_quantile_values=100)
    assert audit.shape == (2, 3)
    assert audit.n_values == 6
    assert audit.n_finite == 5
    assert audit.n_nonfinite == 1
    assert audit.n_negative == 1
    assert audit.n_zero == 1
    assert audit.minimum == pytest.approx(-0.01)
    assert audit.maximum == 7.0
    assert audit.quantiles_exact

    output = tmp_path / "audit.json"
    write_audit(output, audit)
    assert json.loads(output.read_text())["n_negative"] == 1


def test_log_transform_requires_explicit_negative_policy():
    X = np.asarray([[0.0, -0.01, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="negative expression"):
        PanCancerLog2p1().fit_transform(X)
    transformed = PanCancerLog2p1(negative_policy="clip").fit_transform(X)
    np.testing.assert_allclose(transformed, [[0.0, 0.0, 2.0]])


def test_standard_scaler_is_fit_on_training_data_only():
    X_train = np.asarray([[0.0, 1.0], [3.0, 7.0], [15.0, 31.0]], dtype=np.float32)
    X_holdout = np.asarray([[1023.0, 2047.0]], dtype=np.float32)
    pre = build_standardized_preprocessor(scaler="standard", negative_policy="error")
    pre.fit(X_train)

    logged_train = np.log2(X_train + 1.0)
    expected_mean = logged_train.mean(axis=0)
    scaler = pre.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-6)

    before = scaler.mean_.copy()
    transformed = pre.transform(X_holdout)
    np.testing.assert_array_equal(scaler.mean_, before)
    assert transformed.shape == (1, 2)


def test_robust_scaler_pipeline_supported():
    X = np.asarray([[0.0, 1.0], [1.0, 2.0], [3.0, 8.0]], dtype=np.float32)
    pre = build_standardized_preprocessor(scaler="robust")
    out = pre.fit_transform(X)
    assert out.shape == X.shape
