from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import platform
import time
from typing import Callable, Iterable, TextIO

from joblib import parallel_backend
import numpy as np
from threadpoolctl import threadpool_limits

from .models import MODEL_NAMES, build_model_pipeline
from .progress import ProgressReporter, cross_validate_with_progress
from .splitting import DEFAULT_SEED, make_development_cv


SCORING = {
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}


def resolve_n_jobs(requested: int, *, cv_folds: int) -> int:
    if requested < 0:
        raise ValueError("n_jobs must be >= 0; use 0 for automatic allocation")
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")
    if requested > 0:
        return min(requested, cv_folds)
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    available = int(slurm) if slurm and slurm.isdigit() else (os.cpu_count() or 1)
    return max(1, min(available, cv_folds))


def read_split_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cache_index", "participant_barcode", "cancer_type", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"split manifest must contain columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("split manifest is empty")
    allowed = {"development", "holdout"}
    unknown = sorted({row["split"] for row in rows} - allowed)
    if unknown:
        raise ValueError(f"unknown split labels: {unknown}")
    return rows


def load_development_data(
    matrix_path: str | Path,
    split_manifest: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    matrix = np.load(Path(matrix_path), mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError("expression cache must be a 2-D matrix")
    rows = read_split_manifest(split_manifest)
    development = [row for row in rows if row["split"] == "development"]
    if not development:
        raise ValueError("split manifest has no development samples")
    indices = np.asarray([int(row["cache_index"]) for row in development], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= matrix.shape[0]):
        raise ValueError("split manifest contains cache indices outside the expression matrix")
    if len(set(int(index) for index in indices)) != len(indices):
        raise ValueError("development split contains duplicate cache indices")

    X = np.asarray(matrix[indices], dtype=np.float32)
    y = np.asarray([row["cancer_type"] for row in development], dtype=object)
    participants = [row["participant_barcode"] for row in development]
    return X, y, participants


def _metric_summary(values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=float)
    return {
        "fold_values": [float(value) for value in arr],
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def benchmark_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    *,
    cv_folds: int = 5,
    n_jobs: int = 0,
    negative_policy: str = "error",
    scaler: str = "standard",
    seed: int = DEFAULT_SEED,
    pca_components: int = 100,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, object]:
    jobs = resolve_n_jobs(n_jobs, cv_folds=cv_folds)
    if model_name == "pca_logistic":
        smallest_train_fold = len(y) - int(np.ceil(len(y) / cv_folds))
        components = min(pca_components, X.shape[1], max(1, smallest_train_fold - 1))
    else:
        components = pca_components
    pipeline = build_model_pipeline(
        model_name,
        negative_policy=negative_policy,
        scaler=scaler,
        seed=seed,
        pca_components=components,
    )
    cv = make_development_cv(n_splits=cv_folds, seed=seed)

    started = time.perf_counter()
    with threadpool_limits(limits=1), parallel_backend("loky", inner_max_num_threads=1):
        scores = cross_validate_with_progress(
            pipeline,
            X,
            y,
            scoring=SCORING,
            cv_splits=cv.split(X, y),
            n_jobs=jobs,
            return_estimator=False,
            error_score="raise",
            progress_callback=progress_callback,
        )
    wall_seconds = time.perf_counter() - started

    metrics = {name: _metric_summary(scores[f"test_{name}"]) for name in SCORING}
    return {
        "model": model_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": sorted(str(value) for value in np.unique(y)),
        "cv_folds": cv_folds,
        "seed": seed,
        "n_jobs": jobs,
        "negative_policy": negative_policy,
        "scaler": scaler,
        "pca_components": components if model_name == "pca_logistic" else None,
        "metrics": metrics,
        "fit_time_seconds": _metric_summary(scores["fit_time"]),
        "score_time_seconds": _metric_summary(scores["score_time"]),
        "wall_seconds": float(wall_seconds),
    }


def run_benchmark(
    matrix_path: str | Path,
    split_manifest: str | Path,
    outdir: str | Path,
    *,
    models: Iterable[str] = MODEL_NAMES,
    cv_folds: int = 5,
    n_jobs: int = 0,
    negative_policy: str = "error",
    scaler: str = "standard",
    seed: int = DEFAULT_SEED,
    pca_components: int = 100,
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
    progress_heartbeat_seconds: float = 60.0,
) -> dict[str, object]:
    selected_models = list(models)
    unknown = sorted(set(selected_models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    if len(set(selected_models)) != len(selected_models):
        raise ValueError("model list contains duplicates")

    X, y, participants = load_development_data(matrix_path, split_manifest)
    jobs = resolve_n_jobs(n_jobs, cv_folds=cv_folds)
    progress = (
        ProgressReporter(
            prefix="classical",
            task_name="model",
            unit_name="folds",
            total_tasks=len(selected_models),
            total_units=cv_folds,
            stream=progress_stream,
            heartbeat_seconds=progress_heartbeat_seconds,
        )
        if show_progress
        else None
    )
    results: list[dict[str, object]] = []
    for model_index, name in enumerate(selected_models, start=1):
        if progress is not None:
            progress.start_task(
                task_index=model_index,
                task_label=name,
                state=f"starting with {jobs} worker(s)",
            )
        try:
            result = benchmark_model(
                X,
                y,
                name,
                cv_folds=cv_folds,
                n_jobs=n_jobs,
                negative_policy=negative_policy,
                scaler=scaler,
                seed=seed,
                pca_components=pca_components,
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
        results.append(result)

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_scope": "development_cross_validation_only",
        "holdout_used": False,
        "matrix": str(matrix_path),
        "split_manifest": str(split_manifest),
        "n_development_samples": len(participants),
        "models": results,
        "compute": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "requested_n_jobs": n_jobs,
        },
    }
    (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
