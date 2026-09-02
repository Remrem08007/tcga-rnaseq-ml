from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Iterable, TextIO

from joblib import parallel_backend
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .benchmark import SCORING, _metric_summary, load_development_data, resolve_n_jobs
from .models import build_model_pipeline
from .normalization import build_standardized_preprocessor
from .progress import ProgressReporter, cross_validate_with_progress
from .splitting import DEFAULT_SEED, make_development_cv

try:  # Unix/HPC only; keep the analysis usable on platforms without resource.
    import resource
except ImportError:  # pragma: no cover - GitHub CI and Alliance clusters are Unix.
    resource = None


FEATURE_BUDGET_MODELS: tuple[str, ...] = (
    "logistic_l2",
    "elastic_net",
    "linear_svm",
)
DEFAULT_GENE_BUDGETS: tuple[int | str, ...] = (
    20,
    50,
    100,
    200,
    500,
    1_000,
    5_000,
    "all",
)


@dataclass(frozen=True)
class GeneRecord:
    gene_index: int
    source_gene_id: str
    symbol: str
    entrez_id: str


class AdaptiveSelectKBest(BaseEstimator, TransformerMixin):
    """Fold-safe SelectKBest that caps k after fold-specific variance filtering.

    A training fold can lose features at the preceding ``VarianceThreshold``
    step. A fixed k larger than the surviving feature count should select all
    survivors rather than fail the complete experiment.
    """

    def __init__(self, k: int | str = 100) -> None:
        self.k = k

    def fit(self, X, y):
        requested = parse_gene_budget(self.k)
        actual: int | str
        if requested == "all":
            actual = "all"
            self.actual_k_ = int(X.shape[1])
        else:
            actual = min(requested, int(X.shape[1]))
            self.actual_k_ = actual
        self.selector_ = SelectKBest(score_func=f_classif, k=actual).fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self, indices: bool = False):
        return self.selector_.get_support(indices=indices)


def parse_gene_budget(value: str | int) -> int | str:
    if isinstance(value, int):
        if value < 1:
            raise ValueError("gene budget must be >= 1")
        return value

    text = str(value).strip().lower()
    if text == "all":
        return "all"
    try:
        budget = int(text)
    except ValueError as exc:
        raise ValueError("gene budget must be a positive integer or 'all'") from exc
    if budget < 1:
        raise ValueError("gene budget must be >= 1")
    return budget


def read_gene_table(
    path: str | Path,
    *,
    expected_features: int | None = None,
) -> list[GeneRecord]:
    records: list[GeneRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"gene_index", "source_gene_id", "symbol", "entrez_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"gene table must contain columns: {sorted(required)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                gene_index = int(row["gene_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid gene_index at {path}:{line_number}") from exc
            records.append(
                GeneRecord(
                    gene_index=gene_index,
                    source_gene_id=row["source_gene_id"],
                    symbol=row["symbol"],
                    entrez_id=row["entrez_id"],
                )
            )

    observed = [record.gene_index for record in records]
    if observed != list(range(len(records))):
        raise ValueError("gene table indices must be contiguous and ordered from 0")
    if expected_features is not None and len(records) != expected_features:
        raise ValueError(
            f"gene table has {len(records)} rows but matrix has {expected_features} features"
        )
    return records


def _model_estimator(
    model_name: str,
    *,
    negative_policy: str,
    scaler: str,
    seed: int,
):
    if model_name not in FEATURE_BUDGET_MODELS:
        raise ValueError(
            f"feature-budget model must be one of {FEATURE_BUDGET_MODELS}"
        )
    base = build_model_pipeline(
        model_name,
        negative_policy=negative_policy,
        scaler=scaler,
        seed=seed,
    )
    return clone(base.named_steps["model"])


def build_feature_budget_pipeline(
    model_name: str,
    gene_budget: int | str,
    *,
    negative_policy: str = "error",
    scaler: str = "standard",
    seed: int = DEFAULT_SEED,
) -> Pipeline:
    """Build log/impute/variance/select/scale/model as one leakage-safe pipeline."""

    budget = parse_gene_budget(gene_budget)
    preprocessor = build_standardized_preprocessor(
        scaler=scaler,
        negative_policy=negative_policy,
    )

    steps: list[tuple[str, object]] = []
    for name, step in preprocessor.steps:
        if name == "scale":
            steps.append(("select", AdaptiveSelectKBest(k=budget)))
        steps.append((name, step))
    steps.append(
        (
            "model",
            _model_estimator(
                model_name,
                negative_policy=negative_policy,
                scaler=scaler,
                seed=seed,
            ),
        )
    )
    return Pipeline(steps)


def _selected_original_indices(estimator: Pipeline) -> np.ndarray:
    variance_indices = np.asarray(
        estimator.named_steps["variance"].get_support(indices=True),
        dtype=np.int64,
    )
    selected_after_variance = np.asarray(
        estimator.named_steps["select"].get_support(indices=True),
        dtype=np.int64,
    )
    return variance_indices[selected_after_variance]


def _class_coefficients(estimator: Pipeline) -> tuple[list[str], np.ndarray]:
    model = estimator.named_steps["model"]
    if not hasattr(model, "coef_") or not hasattr(model, "classes_"):
        raise TypeError("feature-budget models must expose coef_ and classes_")

    coefficients = np.asarray(model.coef_, dtype=np.float64)
    classes = [str(value) for value in model.classes_]

    # sklearn stores one separating hyperplane for binary linear models.
    # Expand it symmetrically so the output stays class-addressable.
    if coefficients.shape[0] == 1 and len(classes) == 2:
        coefficients = np.vstack([-coefficients[0], coefficients[0]])
    if coefficients.shape[0] != len(classes):
        raise AssertionError("model class/coefficient shape mismatch")
    return classes, coefficients


def _controller_peak_rss_mib() -> float | None:
    if resource is None:
        return None
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _rss_growth(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return max(0.0, after - before)


def evaluate_gene_budget(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    gene_budget: int | str,
    *,
    cv_folds: int = 5,
    n_jobs: int = 0,
    negative_policy: str = "error",
    scaler: str = "standard",
    seed: int = DEFAULT_SEED,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate one gene budget and return summary plus compact fold details."""

    budget = parse_gene_budget(gene_budget)
    jobs = resolve_n_jobs(n_jobs, cv_folds=cv_folds)
    pipeline = build_feature_budget_pipeline(
        model_name,
        budget,
        negative_policy=negative_policy,
        scaler=scaler,
        seed=seed,
    )
    cv = make_development_cv(n_splits=cv_folds, seed=seed)

    rss_before = _controller_peak_rss_mib()
    started = time.perf_counter()
    with threadpool_limits(limits=1), parallel_backend(
        "loky",
        inner_max_num_threads=1,
    ):
        scores = cross_validate_with_progress(
            pipeline,
            X,
            y,
            scoring=SCORING,
            cv_splits=cv.split(X, y),
            n_jobs=jobs,
            return_estimator=True,
            error_score="raise",
            progress_callback=progress_callback,
        )
    wall_seconds = time.perf_counter() - started
    rss_after = _controller_peak_rss_mib()

    selection_counts = np.zeros(X.shape[1], dtype=np.int64)
    absolute_coefficient_sum = np.zeros(X.shape[1], dtype=np.float64)
    coefficient_observations = np.zeros(X.shape[1], dtype=np.int64)
    fold_details: list[dict[str, object]] = []

    for fold_number, estimator in enumerate(scores["estimator"], start=1):
        selected_indices = _selected_original_indices(estimator)
        classes, coefficients = _class_coefficients(estimator)
        if coefficients.shape[1] != len(selected_indices):
            raise AssertionError(
                "model coefficient count does not match selected feature count"
            )

        selection_counts[selected_indices] += 1
        mean_abs = np.mean(np.abs(coefficients), axis=0)
        absolute_coefficient_sum[selected_indices] += mean_abs
        coefficient_observations[selected_indices] += 1
        fold_details.append(
            {
                "fold": fold_number,
                "gene_indices": selected_indices,
                "classes": classes,
                "coefficients": coefficients,
            }
        )

    selected_union = np.flatnonzero(selection_counts)
    stability = [
        {
            "gene_index": int(index),
            "selected_folds": int(selection_counts[index]),
            "selection_frequency": float(selection_counts[index] / cv_folds),
            "mean_abs_coefficient": float(
                absolute_coefficient_sum[index] / coefficient_observations[index]
            ),
        }
        for index in selected_union
    ]

    summary: dict[str, object] = {
        "model": model_name,
        "gene_budget": budget,
        "actual_selected_per_fold": [
            int(len(detail["gene_indices"])) for detail in fold_details
        ],
        "metrics": {
            name: _metric_summary(scores[f"test_{name}"])
            for name in SCORING
        },
        "fit_time_seconds": _metric_summary(scores["fit_time"]),
        "score_time_seconds": _metric_summary(scores["score_time"]),
        "wall_seconds": float(wall_seconds),
        "n_jobs": jobs,
        "matrix_bytes": int(X.nbytes),
        "controller_peak_rss_mib_before": rss_before,
        "controller_peak_rss_mib_after": rss_after,
        "controller_peak_rss_growth_mib": _rss_growth(rss_before, rss_after),
        "memory_note": (
            "Controller-process peak RSS only; parallel worker memory is benchmarked "
            "separately in the compute milestone."
        ),
        "stability": stability,
    }
    return summary, fold_details


def _write_stability_rows(
    writer: csv.writer,
    *,
    model_name: str,
    gene_budget: int | str,
    stability: list[dict[str, object]],
    genes: list[GeneRecord],
) -> None:
    for row in stability:
        gene = genes[int(row["gene_index"])]
        writer.writerow(
            [
                model_name,
                gene_budget,
                gene.gene_index,
                gene.source_gene_id,
                gene.symbol,
                gene.entrez_id,
                row["selected_folds"],
                f"{float(row['selection_frequency']):.6f}",
                f"{float(row['mean_abs_coefficient']):.12g}",
            ]
        )


def _write_coefficient_rows(
    writer: csv.writer,
    *,
    model_name: str,
    gene_budget: int | str,
    fold_details: list[dict[str, object]],
    genes: list[GeneRecord],
) -> None:
    for detail in fold_details:
        indices = np.asarray(detail["gene_indices"], dtype=np.int64)
        coefficients = np.asarray(detail["coefficients"], dtype=np.float64)
        classes = list(detail["classes"])
        for class_index, class_name in enumerate(classes):
            for feature_position, gene_index in enumerate(indices):
                gene = genes[int(gene_index)]
                writer.writerow(
                    [
                        model_name,
                        gene_budget,
                        detail["fold"],
                        class_name,
                        gene.gene_index,
                        gene.source_gene_id,
                        gene.symbol,
                        gene.entrez_id,
                        f"{coefficients[class_index, feature_position]:.12g}",
                    ]
                )


def run_feature_budget(
    matrix_path: str | Path,
    split_manifest: str | Path,
    gene_table: str | Path,
    outdir: str | Path,
    *,
    model_name: str = "elastic_net",
    gene_budgets: Iterable[int | str] = DEFAULT_GENE_BUDGETS,
    cv_folds: int = 5,
    n_jobs: int = 0,
    negative_policy: str = "error",
    scaler: str = "standard",
    seed: int = DEFAULT_SEED,
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
    progress_heartbeat_seconds: float = 60.0,
) -> dict[str, object]:
    """Run development-only gene-budget CV and stream interpretation outputs."""

    X, y, participants = load_development_data(matrix_path, split_manifest)
    genes = read_gene_table(gene_table, expected_features=X.shape[1])
    budgets = [parse_gene_budget(value) for value in gene_budgets]
    if len({str(value) for value in budgets}) != len(budgets):
        raise ValueError("gene budget list contains duplicates")

    progress = (
        ProgressReporter(
            prefix="feature-budget",
            task_name="budget",
            unit_name="folds",
            total_tasks=len(budgets),
            total_units=cv_folds,
            stream=progress_stream,
            heartbeat_seconds=progress_heartbeat_seconds,
        )
        if show_progress
        else None
    )

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stability_path = output_dir / "feature_stability.tsv"
    coefficients_path = output_dir / "coefficients.tsv"

    summaries: list[dict[str, object]] = []
    with stability_path.open(
        "w", encoding="utf-8", newline=""
    ) as stability_handle, coefficients_path.open(
        "w", encoding="utf-8", newline=""
    ) as coefficient_handle:
        stability_writer = csv.writer(
            stability_handle,
            delimiter="\t",
            lineterminator="\n",
        )
        stability_writer.writerow(
            [
                "model",
                "gene_budget",
                "gene_index",
                "source_gene_id",
                "symbol",
                "entrez_id",
                "selected_folds",
                "selection_frequency",
                "mean_abs_coefficient",
            ]
        )
        coefficient_writer = csv.writer(
            coefficient_handle,
            delimiter="\t",
            lineterminator="\n",
        )
        coefficient_writer.writerow(
            [
                "model",
                "gene_budget",
                "fold",
                "class",
                "gene_index",
                "source_gene_id",
                "symbol",
                "entrez_id",
                "coefficient",
            ]
        )

        for budget_index, budget in enumerate(budgets, start=1):
            jobs = resolve_n_jobs(n_jobs, cv_folds=cv_folds)
            if progress is not None:
                progress.start_task(
                    task_index=budget_index,
                    task_label=f"{budget} genes",
                    state=f"starting with {jobs} worker(s)",
                )
            try:
                summary, fold_details = evaluate_gene_budget(
                    X,
                    y,
                    model_name,
                    budget,
                    cv_folds=cv_folds,
                    n_jobs=n_jobs,
                    negative_policy=negative_policy,
                    scaler=scaler,
                    seed=seed,
                    progress_callback=(
                        progress.units_completed if progress is not None else None
                    ),
                )
            except BaseException:
                if progress is not None:
                    progress.fail_task()
                raise
            else:
                if progress is not None:
                    progress.finish_task()
            _write_stability_rows(
                stability_writer,
                model_name=model_name,
                gene_budget=budget,
                stability=summary["stability"],
                genes=genes,
            )
            _write_coefficient_rows(
                coefficient_writer,
                model_name=model_name,
                gene_budget=budget,
                fold_details=fold_details,
                genes=genes,
            )
            summary = dict(summary)
            summary.pop("stability")
            summaries.append(summary)

    payload: dict[str, object] = {
        "evaluation_scope": "development_cross_validation_only",
        "holdout_used": False,
        "model": model_name,
        "matrix": str(matrix_path),
        "split_manifest": str(split_manifest),
        "gene_table": str(gene_table),
        "n_development_samples": len(participants),
        "n_input_features": int(X.shape[1]),
        "negative_policy": negative_policy,
        "scaler": scaler,
        "seed": seed,
        "budgets": summaries,
        "outputs": {
            "feature_stability": stability_path.name,
            "coefficients": coefficients_path.name,
        },
    }
    (output_dir / "feature_budget.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
