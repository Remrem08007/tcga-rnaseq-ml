from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import time
from typing import Callable, TextIO
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import parallel_backend
from threadpoolctl import threadpool_limits

from .benchmark import SCORING, _metric_summary, load_development_data
from .feature_budget import AdaptiveSelectKBest, _selected_original_indices, parse_gene_budget, read_gene_table
from .normalization import PanCancerLog2p1
from .progress import ProgressReporter, cross_validate_with_progress
from .splitting import DEFAULT_SEED, make_development_cv

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None


@dataclass(frozen=True)
class DeviceProbe:
    requested: str
    available: bool
    resolved: str | None
    reason: str
    xgboost_version: str
    use_cuda_build: bool | None
    cuda_build_version: list[int] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - exercised by install policy, not CI
        raise ImportError(
            "XGBoost support is optional. Install this project with the 'xgboost' extra: "
            "python -m pip install -e '.[xgboost]'"
        ) from exc
    return xgb


def _booster_device(model) -> str:
    config = json.loads(model.get_booster().save_config())
    return str(config["learner"]["generic_param"]["device"])


@lru_cache(maxsize=1)
def probe_cuda() -> DeviceProbe:
    """Prove that XGBoost can actually train on a visible CUDA device.

    A CUDA-enabled wheel is not enough: XGBoost can emit a warning and silently
    fall back to CPU when no GPU is visible. We fit a two-tree probe and inspect
    the resulting booster configuration so a requested GPU benchmark is never
    mislabeled as CUDA when it actually ran on CPU.
    """

    xgb = _require_xgboost()
    build = xgb.build_info()
    use_cuda = build.get("USE_CUDA")
    cuda_version = build.get("CUDA_VERSION")
    if not use_cuda:
        return DeviceProbe(
            requested="cuda",
            available=False,
            resolved=None,
            reason="installed XGBoost build reports USE_CUDA=False",
            xgboost_version=xgb.__version__,
            use_cuda_build=False,
            cuda_build_version=None,
        )

    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int32)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = xgb.XGBClassifier(
                n_estimators=2,
                max_depth=2,
                tree_method="hist",
                device="cuda",
                n_jobs=1,
                random_state=0,
                verbosity=1,
            )
            model.fit(X, y)
        resolved = _booster_device(model)
    except Exception as exc:  # pragma: no cover - exact CUDA failure varies by host
        return DeviceProbe(
            requested="cuda",
            available=False,
            resolved=None,
            reason=f"CUDA training probe failed: {type(exc).__name__}: {exc}",
            xgboost_version=xgb.__version__,
            use_cuda_build=bool(use_cuda),
            cuda_build_version=list(cuda_version) if cuda_version else None,
        )

    fallback_warnings = [
        str(item.message)
        for item in captured
        if "gpu" in str(item.message).lower() or "cuda" in str(item.message).lower()
    ]
    if not resolved.startswith("cuda"):
        reason = f"XGBoost resolved the CUDA probe to {resolved!r}"
        if fallback_warnings:
            reason += "; " + " | ".join(fallback_warnings)
        return DeviceProbe(
            requested="cuda",
            available=False,
            resolved=resolved,
            reason=reason,
            xgboost_version=xgb.__version__,
            use_cuda_build=bool(use_cuda),
            cuda_build_version=list(cuda_version) if cuda_version else None,
        )

    return DeviceProbe(
        requested="cuda",
        available=True,
        resolved=resolved,
        reason="CUDA training probe succeeded and booster remained on CUDA",
        xgboost_version=xgb.__version__,
        use_cuda_build=bool(use_cuda),
        cuda_build_version=list(cuda_version) if cuda_version else None,
    )


def resolve_device(requested: str) -> tuple[str, DeviceProbe | None]:
    value = requested.strip().lower()
    if value not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    if value == "cpu":
        return "cpu", None

    probe = probe_cuda()
    if value == "cuda":
        if not probe.available:
            raise RuntimeError(
                "CUDA was explicitly requested, but the XGBoost CUDA probe failed: "
                + probe.reason
            )
        return "cuda", probe
    return ("cuda" if probe.available else "cpu"), probe


def available_cpus() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if value.isdigit() and int(value) > 0:
        return int(value)
    return max(1, os.cpu_count() or 1)


def resolve_threads(requested: int) -> int:
    if requested < 0:
        raise ValueError("threads must be >= 0; use 0 for automatic allocation")
    return available_cpus() if requested == 0 else requested


class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBClassifier adapter with inverse-frequency multiclass sample weights.

    XGBoost expects contiguous integer labels. This adapter accepts the string
    TCGA labels used everywhere else in the project, encodes them internally,
    balances each training fold with sample weights, and maps predictions back
    to the original labels for sklearn scoring.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.5,
        min_child_weight: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        max_bin: int = 256,
        n_jobs: int = 1,
        random_state: int = DEFAULT_SEED,
    ) -> None:
        self.device = device
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_bin = max_bin
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        xgb = _require_xgboost()
        labels = np.asarray(y, dtype=object)
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        counts = np.bincount(encoded, minlength=len(self.classes_)).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError("training labels contain an empty encoded class")
        class_weights = len(encoded) / (len(self.classes_) * counts)
        sample_weight = class_weights[encoded]

        self.model_ = xgb.XGBClassifier(
            tree_method="hist",
            device=self.device,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            max_bin=self.max_bin,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self.model_.fit(X, encoded.astype(np.int32), sample_weight=sample_weight)
        resolved = _booster_device(self.model_)
        if self.device == "cuda" and not resolved.startswith("cuda"):
            raise RuntimeError(
                f"XGBoost was asked to train on CUDA but the fitted booster resolved to {resolved!r}"
            )
        self.resolved_device_ = resolved
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X):
        encoded = np.asarray(self.model_.predict(X), dtype=np.int64)
        return self.classes_[encoded]

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


def build_xgboost_pipeline(
    *,
    device: str,
    threads: int,
    gene_budget: int | str = 1_000,
    negative_policy: str = "error",
    seed: int = DEFAULT_SEED,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.5,
    min_child_weight: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    max_bin: int = 256,
) -> Pipeline:
    budget = parse_gene_budget(gene_budget)
    if threads < 1:
        raise ValueError("threads must be >= 1")
    return Pipeline(
        steps=[
            ("log2p1", PanCancerLog2p1(negative_policy=negative_policy)),
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("select", AdaptiveSelectKBest(k=budget)),
            (
                "model",
                BalancedXGBClassifier(
                    device=device,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    gamma=gamma,
                    max_bin=max_bin,
                    n_jobs=threads,
                    random_state=seed,
                ),
            ),
        ]
    )


def _peak_rss_mib() -> float | None:
    if resource is None:
        return None
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def evaluate_xgboost_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    requested_device: str = "cpu",
    threads: int = 0,
    fold_jobs: int = 1,
    gene_budget: int | str = 1_000,
    negative_policy: str = "error",
    cv_folds: int = 5,
    seed: int = DEFAULT_SEED,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.5,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    device, probe = resolve_device(requested_device)
    resolved_threads = resolve_threads(threads)
    if fold_jobs < 1:
        raise ValueError("fold_jobs must be >= 1")
    if device == "cuda" and fold_jobs != 1:
        raise ValueError("CUDA CV requires fold_jobs=1 so folds do not compete for one GPU")
    if device == "cpu" and fold_jobs * resolved_threads > available_cpus():
        raise ValueError(
            "fold_jobs * threads exceeds the available CPU allocation; "
            "reduce one level of parallelism to avoid oversubscription"
        )

    pipeline = build_xgboost_pipeline(
        device=device,
        threads=resolved_threads,
        gene_budget=gene_budget,
        negative_policy=negative_policy,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )
    cv = make_development_cv(n_splits=cv_folds, seed=seed)

    rss_before = _peak_rss_mib()
    started = time.perf_counter()
    with threadpool_limits(limits=1), parallel_backend(
        "loky", inner_max_num_threads=1
    ):
        scores = cross_validate_with_progress(
            pipeline,
            X,
            y,
            scoring=SCORING,
            cv_splits=cv.split(X, y),
            n_jobs=fold_jobs,
            return_estimator=True,
            error_score="raise",
            progress_callback=progress_callback,
        )
    wall_seconds = time.perf_counter() - started
    rss_after = _peak_rss_mib()

    fold_details: list[dict[str, object]] = []
    for fold_number, estimator in enumerate(scores["estimator"], start=1):
        indices = _selected_original_indices(estimator)
        model = estimator.named_steps["model"]
        importance = np.asarray(model.feature_importances_, dtype=np.float64)
        if len(indices) != len(importance):
            raise AssertionError("XGBoost importance count does not match selected feature count")
        fold_details.append(
            {
                "fold": fold_number,
                "gene_indices": indices,
                "importance": importance,
                "resolved_device": model.resolved_device_,
            }
        )

    summary: dict[str, object] = {
        "model": "xgboost",
        "requested_device": requested_device,
        "resolved_device": device,
        "device_probe": probe.to_dict() if probe is not None else None,
        "threads": resolved_threads,
        "fold_jobs": fold_jobs,
        "gene_budget": parse_gene_budget(gene_budget),
        "cv_folds": cv_folds,
        "seed": seed,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "metrics": {
            name: _metric_summary(scores[f"test_{name}"]) for name in SCORING
        },
        "fit_time_seconds": _metric_summary(scores["fit_time"]),
        "score_time_seconds": _metric_summary(scores["score_time"]),
        "wall_seconds": float(wall_seconds),
        "controller_peak_rss_mib_before": rss_before,
        "controller_peak_rss_mib_after": rss_after,
        "host_memory_note": (
            "Peak host RSS is controller-process accounting. GPU memory is not claimed "
            "without a device-level peak-memory sampler."
        ),
    }
    return summary, fold_details


def _aggregate_importance(
    fold_details: list[dict[str, object]],
    *,
    n_features: int,
) -> list[dict[str, object]]:
    selected = np.zeros(n_features, dtype=np.int64)
    importance_sum = np.zeros(n_features, dtype=np.float64)
    for detail in fold_details:
        indices = np.asarray(detail["gene_indices"], dtype=np.int64)
        importance = np.asarray(detail["importance"], dtype=np.float64)
        selected[indices] += 1
        importance_sum[indices] += importance
    rows = []
    for index in np.flatnonzero(selected):
        rows.append(
            {
                "gene_index": int(index),
                "selected_folds": int(selected[index]),
                "selection_frequency": float(selected[index] / len(fold_details)),
                "mean_gain_importance": float(importance_sum[index] / selected[index]),
            }
        )
    rows.sort(key=lambda row: (-row["mean_gain_importance"], row["gene_index"]))
    return rows


def run_xgboost_benchmark(
    matrix_path: str | Path,
    split_manifest: str | Path,
    gene_table: str | Path,
    outdir: str | Path,
    *,
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
    progress_heartbeat_seconds: float = 60.0,
    **kwargs,
) -> dict[str, object]:
    X, y, participants = load_development_data(matrix_path, split_manifest)
    genes = read_gene_table(gene_table, expected_features=X.shape[1])
    cv_folds = int(kwargs.get("cv_folds", 5))
    fold_jobs = int(kwargs.get("fold_jobs", 1))
    requested_device = str(kwargs.get("requested_device", "cpu"))
    budget = parse_gene_budget(kwargs.get("gene_budget", 1_000))
    progress = (
        ProgressReporter(
            prefix="xgboost",
            task_name="run",
            unit_name="folds",
            total_tasks=1,
            total_units=cv_folds,
            stream=progress_stream,
            heartbeat_seconds=progress_heartbeat_seconds,
        )
        if show_progress
        else None
    )
    if progress is not None:
        progress.start_task(
            task_index=1,
            task_label=f"{requested_device}, {budget} genes",
            state=f"starting with {fold_jobs} fold worker(s)",
        )
    try:
        summary, folds = evaluate_xgboost_cv(
            X,
            y,
            progress_callback=(
                progress.units_completed if progress is not None else None
            ),
            **kwargs,
        )
    except BaseException:
        if progress is not None:
            progress.fail_task()
        raise
    else:
        if progress is not None:
            progress.finish_task()
    importance = _aggregate_importance(folds, n_features=X.shape[1])

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    importance_path = output_dir / "xgboost_feature_importance.tsv"
    with importance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "gene_index",
                "source_gene_id",
                "symbol",
                "entrez_id",
                "selected_folds",
                "selection_frequency",
                "mean_gain_importance",
            ]
        )
        for row in importance:
            gene = genes[row["gene_index"]]
            writer.writerow(
                [
                    gene.gene_index,
                    gene.source_gene_id,
                    gene.symbol,
                    gene.entrez_id,
                    row["selected_folds"],
                    f"{row['selection_frequency']:.6f}",
                    f"{row['mean_gain_importance']:.12g}",
                ]
            )

    payload = {
        "evaluation_scope": "development_cross_validation_only",
        "holdout_used": False,
        "matrix": str(matrix_path),
        "split_manifest": str(split_manifest),
        "gene_table": str(gene_table),
        "n_development_samples": len(participants),
        "n_input_features": int(X.shape[1]),
        "benchmark": summary,
        "outputs": {"feature_importance": importance_path.name},
    }
    (output_dir / "xgboost_benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def run_compute_scaling(
    matrix_path: str | Path,
    split_manifest: str | Path,
    outdir: str | Path,
    *,
    cpu_threads: list[int],
    include_gpu: bool = False,
    require_gpu: bool = False,
    gene_budget: int | str = 1_000,
    negative_policy: str = "error",
    cv_folds: int = 3,
    seed: int = DEFAULT_SEED,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
    progress_heartbeat_seconds: float = 60.0,
) -> dict[str, object]:
    if not cpu_threads:
        raise ValueError("at least one CPU thread count is required")
    if any(value < 1 for value in cpu_threads):
        raise ValueError("CPU thread counts must all be >= 1")
    if len(set(cpu_threads)) != len(cpu_threads):
        raise ValueError("CPU thread counts contain duplicates")
    allocation = available_cpus()
    oversized = [value for value in cpu_threads if value > allocation]
    if oversized:
        raise ValueError(
            f"CPU thread counts exceed available allocation ({allocation}): {oversized}"
        )

    gpu_probe: DeviceProbe | None = None
    if include_gpu or require_gpu:
        gpu_probe = probe_cuda()
        if require_gpu and not gpu_probe.available:
            raise RuntimeError(
                "GPU benchmark was required but unavailable: " + gpu_probe.reason
            )

    X, y, _ = load_development_data(matrix_path, split_manifest)
    total_runs = len(cpu_threads) + int(bool(gpu_probe and gpu_probe.available))
    progress = (
        ProgressReporter(
            prefix="xgboost-scale",
            task_name="configuration",
            unit_name="folds",
            total_tasks=total_runs,
            total_units=cv_folds,
            stream=progress_stream,
            heartbeat_seconds=progress_heartbeat_seconds,
        )
        if show_progress
        else None
    )
    runs: list[dict[str, object]] = []
    for run_index, thread_count in enumerate(cpu_threads, start=1):
        if progress is not None:
            progress.start_task(
                task_index=run_index,
                task_label=f"cpu, {thread_count} thread(s)",
                state="starting",
            )
        try:
            summary, _ = evaluate_xgboost_cv(
                X,
                y,
                requested_device="cpu",
                threads=thread_count,
                fold_jobs=1,
                gene_budget=gene_budget,
                negative_policy=negative_policy,
                cv_folds=cv_folds,
                seed=seed,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
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
        runs.append(summary)

    gpu_status: dict[str, object] | None = None
    if include_gpu or require_gpu:
        if gpu_probe is not None and gpu_probe.available:
            if progress is not None:
                progress.start_task(
                    task_index=len(cpu_threads) + 1,
                    task_label="cuda",
                    state="starting",
                )
            try:
                summary, _ = evaluate_xgboost_cv(
                    X,
                    y,
                    requested_device="cuda",
                    threads=max(1, min(4, available_cpus())),
                    fold_jobs=1,
                    gene_budget=gene_budget,
                    negative_policy=negative_policy,
                    cv_folds=cv_folds,
                    seed=seed,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
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
            runs.append(summary)
            gpu_status = {"status": "completed", "probe": gpu_probe.to_dict()}
        else:
            if gpu_probe is None:
                raise AssertionError("GPU probe was not run")
            gpu_status = {"status": "unavailable", "probe": gpu_probe.to_dict()}

    baseline = next(run for run in runs if run["resolved_device"] == "cpu")
    baseline_seconds = float(baseline["wall_seconds"])
    compact: list[dict[str, object]] = []
    for run in runs:
        wall = float(run["wall_seconds"])
        compact.append(
            {
                "device": run["resolved_device"],
                "threads": run["threads"],
                "wall_seconds": wall,
                "speedup_vs_first_cpu": baseline_seconds / wall if wall > 0 else None,
                "macro_f1_mean": run["metrics"]["macro_f1"]["mean"],
                "balanced_accuracy_mean": run["metrics"]["balanced_accuracy"]["mean"],
                "controller_peak_rss_mib_after": run["controller_peak_rss_mib_after"],
            }
        )

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "compute_scaling.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(compact[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(compact)

    payload = {
        "evaluation_scope": "development_cross_validation_only",
        "holdout_used": False,
        "cpu_thread_counts": cpu_threads,
        "gpu_status": gpu_status,
        "runs": compact,
        "output": tsv_path.name,
        "note": (
            "CPU/GPU speed is reported separately from model selection. Host RSS is measured; "
            "GPU memory is not labeled as peak usage without a device-level sampler."
        ),
    }
    (output_dir / "compute_scaling.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
