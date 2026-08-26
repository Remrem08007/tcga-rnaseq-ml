from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass(frozen=True)
class ExpressionAudit:
    shape: tuple[int, int]
    dtype: str
    n_values: int
    n_finite: int
    n_nonfinite: int
    n_negative: int
    n_zero: int
    negative_fraction: float
    zero_fraction: float
    minimum: float | None
    maximum: float | None
    quantiles: dict[str, float]
    quantiles_exact: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


def audit_expression_matrix(
    matrix_path: str | Path,
    *,
    max_quantile_values: int = 1_000_000,
    row_chunk: int = 256,
) -> ExpressionAudit:
    if max_quantile_values < 1:
        raise ValueError("max_quantile_values must be >= 1")
    if row_chunk < 1:
        raise ValueError("row_chunk must be >= 1")

    matrix = np.load(Path(matrix_path), mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D expression matrix, found shape {matrix.shape}")

    n_values = int(matrix.size)
    n_finite = 0
    n_negative = 0
    n_zero = 0
    minimum = np.inf
    maximum = -np.inf

    for start in range(0, matrix.shape[0], row_chunk):
        block = np.asarray(matrix[start : start + row_chunk])
        finite = np.isfinite(block)
        finite_values = block[finite]
        n_finite += int(finite_values.size)
        if finite_values.size:
            n_negative += int(np.count_nonzero(finite_values < 0))
            n_zero += int(np.count_nonzero(finite_values == 0))
            minimum = min(minimum, float(np.min(finite_values)))
            maximum = max(maximum, float(np.max(finite_values)))

    n_nonfinite = n_values - n_finite
    if n_finite == 0:
        quantiles = {key: float("nan") for key in ("q0", "q01", "q25", "q50", "q75", "q99", "q100")}
        min_value = None
        max_value = None
        exact = True
    else:
        stride = max(1, (n_values + max_quantile_values - 1) // max_quantile_values)
        sampled = np.asarray(matrix.reshape(-1)[::stride])
        sampled = sampled[np.isfinite(sampled)]
        probs = np.asarray([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
        values = np.quantile(sampled, probs)
        quantiles = {
            "q0": float(values[0]),
            "q01": float(values[1]),
            "q25": float(values[2]),
            "q50": float(values[3]),
            "q75": float(values[4]),
            "q99": float(values[5]),
            "q100": float(values[6]),
        }
        min_value = float(minimum)
        max_value = float(maximum)
        exact = stride == 1

    denom = max(n_finite, 1)
    return ExpressionAudit(
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        dtype=str(matrix.dtype),
        n_values=n_values,
        n_finite=n_finite,
        n_nonfinite=n_nonfinite,
        n_negative=n_negative,
        n_zero=n_zero,
        negative_fraction=n_negative / denom,
        zero_fraction=n_zero / denom,
        minimum=min_value,
        maximum=max_value,
        quantiles=quantiles,
        quantiles_exact=exact,
    )


def write_audit(path: str | Path, audit: ExpressionAudit) -> None:
    Path(path).write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class PanCancerLog2p1(BaseEstimator, TransformerMixin):
    """Apply log2(x + 1) with explicit handling of batch-correction negatives.

    The source PanCancer matrix can contain small negative values after batch
    correction. The transform never clips them silently: callers must choose
    `negative_policy="clip"` after reviewing the source audit, or leave the
    safer default `"error"`.
    """

    def __init__(self, *, negative_policy: str = "error") -> None:
        self.negative_policy = negative_policy

    def fit(self, X, y=None):
        self._validate_policy()
        return self

    def transform(self, X):
        self._validate_policy()
        values = np.asarray(X, dtype=np.float32)
        finite_negative = np.isfinite(values) & (values < 0)
        if np.any(finite_negative):
            if self.negative_policy == "error":
                minimum = float(np.min(values[finite_negative]))
                raise ValueError(
                    "negative expression values found before log2p1 "
                    f"(minimum={minimum}); review the audit and explicitly choose negative_policy='clip' if appropriate"
                )
            values = values.copy()
            values[finite_negative] = 0.0
        return np.log2(values + np.float32(1.0)).astype(np.float32, copy=False)

    def _validate_policy(self) -> None:
        if self.negative_policy not in {"error", "clip"}:
            raise ValueError("negative_policy must be 'error' or 'clip'")


def build_standardized_preprocessor(
    *,
    scaler: str = "standard",
    negative_policy: str = "error",
) -> Pipeline:
    if scaler == "standard":
        scaler_step = StandardScaler()
    elif scaler == "robust":
        scaler_step = RobustScaler()
    else:
        raise ValueError("scaler must be 'standard' or 'robust'")

    return Pipeline(
        steps=[
            ("log2p1", PanCancerLog2p1(negative_policy=negative_policy)),
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scale", scaler_step),
        ]
    )
