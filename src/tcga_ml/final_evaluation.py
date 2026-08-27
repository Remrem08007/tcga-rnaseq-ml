from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import csv
from html import escape
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .benchmark import read_split_manifest
from .feature_budget import (
    FEATURE_BUDGET_MODELS,
    build_feature_budget_pipeline,
    parse_gene_budget,
)
from .models import build_model_pipeline
from .splitting import DEFAULT_SEED


LOCK_SCHEMA_VERSION = 1
DEVELOPMENT_SCOPE = "development_cross_validation_only"
PRIMARY_METRIC = "macro_f1"
PIPELINE_FAMILIES: tuple[str, ...] = (
    "linear_gene_budget",
    "pca_logistic",
    "xgboost",
)


class FinalEvaluationLockError(ValueError):
    """Raised when the final-evaluation lock is invalid or has been changed."""


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be >= 1")
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _lock_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("lock_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalEvaluationLockError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, *, field: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise FinalEvaluationLockError(f"{field} must be an integer >= {minimum}")
    return value


def _number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalEvaluationLockError(f"{field} must be numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise FinalEvaluationLockError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise FinalEvaluationLockError(f"{field} must be <= {maximum}")
    return result


def _reject_unknown_fields(
    payload: Mapping[str, object],
    *,
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise FinalEvaluationLockError(f"unknown {context} fields: {unknown}")


def _normalize_common_pipeline(
    raw: Mapping[str, object],
    *,
    allowed: set[str],
) -> dict[str, object]:
    _reject_unknown_fields(raw, allowed=allowed, context="pipeline")
    negative_policy = raw.get("negative_policy", "error")
    if negative_policy not in {"error", "clip"}:
        raise FinalEvaluationLockError("negative_policy must be 'error' or 'clip'")
    return {
        "negative_policy": negative_policy,
        "seed": _integer(raw.get("seed", DEFAULT_SEED), field="seed", minimum=0),
    }


def normalize_pipeline_config(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise FinalEvaluationLockError("pipeline must be a JSON object")
    family = raw.get("family")
    if family not in PIPELINE_FAMILIES:
        raise FinalEvaluationLockError(
            f"pipeline family must be one of {PIPELINE_FAMILIES}"
        )

    if family == "linear_gene_budget":
        allowed = {
            "family",
            "model",
            "gene_budget",
            "negative_policy",
            "scaler",
            "seed",
        }
        normalized = _normalize_common_pipeline(raw, allowed=allowed)
        model = raw.get("model")
        if model not in FEATURE_BUDGET_MODELS:
            raise FinalEvaluationLockError(
                f"linear model must be one of {FEATURE_BUDGET_MODELS}"
            )
        scaler = raw.get("scaler", "standard")
        if scaler not in {"standard", "robust"}:
            raise FinalEvaluationLockError("scaler must be 'standard' or 'robust'")
        try:
            budget = parse_gene_budget(raw.get("gene_budget", 1_000))
        except ValueError as exc:
            raise FinalEvaluationLockError(str(exc)) from exc
        return {
            "family": family,
            "model": model,
            "gene_budget": budget,
            "negative_policy": normalized["negative_policy"],
            "scaler": scaler,
            "seed": normalized["seed"],
        }

    if family == "pca_logistic":
        allowed = {
            "family",
            "pca_components",
            "negative_policy",
            "scaler",
            "seed",
        }
        normalized = _normalize_common_pipeline(raw, allowed=allowed)
        scaler = raw.get("scaler", "standard")
        if scaler not in {"standard", "robust"}:
            raise FinalEvaluationLockError("scaler must be 'standard' or 'robust'")
        return {
            "family": family,
            "pca_components": _integer(
                raw.get("pca_components", 100),
                field="pca_components",
            ),
            "negative_policy": normalized["negative_policy"],
            "scaler": scaler,
            "seed": normalized["seed"],
        }

    allowed = {
        "family",
        "device",
        "threads",
        "gene_budget",
        "negative_policy",
        "seed",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "max_bin",
    }
    normalized = _normalize_common_pipeline(raw, allowed=allowed)
    device = raw.get("device", "cpu")
    if device not in {"cpu", "cuda"}:
        raise FinalEvaluationLockError("XGBoost device must be 'cpu' or 'cuda'")
    try:
        budget = parse_gene_budget(raw.get("gene_budget", 1_000))
    except ValueError as exc:
        raise FinalEvaluationLockError(str(exc)) from exc
    return {
        "family": family,
        "device": device,
        "threads": _integer(raw.get("threads", 1), field="threads"),
        "gene_budget": budget,
        "negative_policy": normalized["negative_policy"],
        "seed": normalized["seed"],
        "n_estimators": _integer(
            raw.get("n_estimators", 300),
            field="n_estimators",
        ),
        "max_depth": _integer(raw.get("max_depth", 6), field="max_depth"),
        "learning_rate": _number(
            raw.get("learning_rate", 0.05),
            field="learning_rate",
            minimum=0.0,
        ),
        "subsample": _number(
            raw.get("subsample", 0.8),
            field="subsample",
            minimum=0.0,
            maximum=1.0,
        ),
        "colsample_bytree": _number(
            raw.get("colsample_bytree", 0.5),
            field="colsample_bytree",
            minimum=0.0,
            maximum=1.0,
        ),
        "min_child_weight": _number(
            raw.get("min_child_weight", 1.0),
            field="min_child_weight",
            minimum=0.0,
        ),
        "reg_alpha": _number(
            raw.get("reg_alpha", 0.0),
            field="reg_alpha",
            minimum=0.0,
        ),
        "reg_lambda": _number(
            raw.get("reg_lambda", 1.0),
            field="reg_lambda",
            minimum=0.0,
        ),
        "gamma": _number(raw.get("gamma", 0.0), field="gamma", minimum=0.0),
        "max_bin": _integer(raw.get("max_bin", 256), field="max_bin", minimum=2),
    }


def load_selection_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinalEvaluationLockError(
            f"selection config is not valid JSON: {config_path}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise FinalEvaluationLockError("selection config must be a JSON object")
    _reject_unknown_fields(
        raw,
        allowed={
            "candidate_id",
            "primary_metric",
            "selection_rationale",
            "pipeline",
        },
        context="selection-config",
    )
    primary_metric = raw.get("primary_metric", PRIMARY_METRIC)
    if primary_metric != PRIMARY_METRIC:
        raise FinalEvaluationLockError(
            f"primary_metric must remain locked to {PRIMARY_METRIC!r}"
        )
    return {
        "candidate_id": _nonempty_string(
            raw.get("candidate_id"),
            field="candidate_id",
        ),
        "primary_metric": PRIMARY_METRIC,
        "selection_rationale": _nonempty_string(
            raw.get("selection_rationale"),
            field="selection_rationale",
        ),
        "pipeline": normalize_pipeline_config(raw.get("pipeline")),
    }


def _artifact_record(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FinalEvaluationLockError(f"required artifact is not a file: {source}")
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _evidence_record(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinalEvaluationLockError(
            f"selection evidence is not valid JSON: {source}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FinalEvaluationLockError(
            f"selection evidence must be a JSON object: {source}"
        )
    if payload.get("evaluation_scope") != DEVELOPMENT_SCOPE:
        raise FinalEvaluationLockError(
            f"selection evidence is not development-only: {source}"
        )
    if payload.get("holdout_used") is not False:
        raise FinalEvaluationLockError(
            f"selection evidence must declare holdout_used=false: {source}"
        )
    record = _artifact_record(source)
    record.update(
        {
            "evaluation_scope": DEVELOPMENT_SCOPE,
            "holdout_used": False,
        }
    )
    return record


def create_final_evaluation_lock(
    selection_config: str | Path,
    matrix_path: str | Path,
    split_manifest: str | Path,
    gene_table: str | Path,
    evidence_paths: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, object]:
    evidence = [_evidence_record(path) for path in evidence_paths]
    if not evidence:
        raise FinalEvaluationLockError(
            "at least one development-only selection evidence file is required"
        )
    evidence_hashes = [str(record["sha256"]) for record in evidence]
    if len(set(evidence_hashes)) != len(evidence_hashes):
        raise FinalEvaluationLockError("selection evidence contains duplicate files")

    config = load_selection_config(selection_config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "locked_final_holdout_once",
        "holdout_status": "sealed",
        "selection": config,
        "selection_config": _artifact_record(selection_config),
        "selection_evidence": evidence,
        "artifacts": {
            "matrix": _artifact_record(matrix_path),
            "split_manifest": _artifact_record(split_manifest),
            "gene_table": _artifact_record(gene_table),
        },
    }
    payload["lock_sha256"] = _lock_digest(payload)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise FinalEvaluationLockError(
            f"refusing to overwrite existing final-evaluation lock: {output}"
        ) from exc
    return payload


def load_final_evaluation_lock(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinalEvaluationLockError(
            f"final-evaluation lock is not valid JSON: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise FinalEvaluationLockError("final-evaluation lock must be a JSON object")
    if payload.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise FinalEvaluationLockError(
            f"unsupported final-evaluation lock schema: {payload.get('schema_version')!r}"
        )
    expected = payload.get("lock_sha256")
    if not isinstance(expected, str) or expected != _lock_digest(payload):
        raise FinalEvaluationLockError(
            "final-evaluation lock digest mismatch; the lock was changed"
        )
    if payload.get("holdout_status") != "sealed":
        raise FinalEvaluationLockError("final-evaluation lock is not sealed")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise FinalEvaluationLockError("final-evaluation lock has no selection object")
    normalize_pipeline_config(selection.get("pipeline"))
    if selection.get("primary_metric") != PRIMARY_METRIC:
        raise FinalEvaluationLockError("final-evaluation primary metric was changed")
    return payload


def _verify_artifact(
    path: str | Path,
    expected: object,
    *,
    label: str,
) -> None:
    if not isinstance(expected, Mapping):
        raise FinalEvaluationLockError(f"lock is missing {label} artifact metadata")
    observed = _artifact_record(path)
    if observed["size_bytes"] != expected.get("size_bytes"):
        raise FinalEvaluationLockError(f"{label} size does not match the lock")
    if observed["sha256"] != expected.get("sha256"):
        raise FinalEvaluationLockError(f"{label} SHA-256 does not match the lock")


def verify_final_evaluation_lock(
    lock_path: str | Path,
    matrix_path: str | Path,
    split_manifest: str | Path,
    gene_table: str | Path,
    *,
    evidence_paths: Iterable[str | Path] | None = None,
) -> dict[str, object]:
    payload = load_final_evaluation_lock(lock_path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FinalEvaluationLockError("lock is missing artifact metadata")
    _verify_artifact(matrix_path, artifacts.get("matrix"), label="matrix")
    _verify_artifact(
        split_manifest,
        artifacts.get("split_manifest"),
        label="split manifest",
    )
    _verify_artifact(gene_table, artifacts.get("gene_table"), label="gene table")

    if evidence_paths is not None:
        observed = [_evidence_record(path) for path in evidence_paths]
        expected = payload.get("selection_evidence")
        if not isinstance(expected, list):
            raise FinalEvaluationLockError("lock is missing selection evidence")
        observed_hashes = sorted(str(record["sha256"]) for record in observed)
        expected_hashes = sorted(
            str(record.get("sha256"))
            for record in expected
            if isinstance(record, Mapping)
        )
        if observed_hashes != expected_hashes:
            raise FinalEvaluationLockError(
                "selection evidence SHA-256 set does not match the lock"
            )
    return payload


def build_locked_pipeline(lock_payload: Mapping[str, object]):
    selection = lock_payload.get("selection")
    if not isinstance(selection, Mapping):
        raise FinalEvaluationLockError("lock is missing the selected pipeline")
    pipeline_raw = selection.get("pipeline")
    pipeline = normalize_pipeline_config(pipeline_raw)
    family = pipeline["family"]

    if family == "linear_gene_budget":
        return build_feature_budget_pipeline(
            str(pipeline["model"]),
            pipeline["gene_budget"],
            negative_policy=str(pipeline["negative_policy"]),
            scaler=str(pipeline["scaler"]),
            seed=int(pipeline["seed"]),
        )

    if family == "pca_logistic":
        return build_model_pipeline(
            "pca_logistic",
            negative_policy=str(pipeline["negative_policy"]),
            scaler=str(pipeline["scaler"]),
            seed=int(pipeline["seed"]),
            pca_components=int(pipeline["pca_components"]),
        )

    from .xgboost_benchmark import build_xgboost_pipeline

    return build_xgboost_pipeline(
        device=str(pipeline["device"]),
        threads=int(pipeline["threads"]),
        gene_budget=pipeline["gene_budget"],
        negative_policy=str(pipeline["negative_policy"]),
        seed=int(pipeline["seed"]),
        n_estimators=int(pipeline["n_estimators"]),
        max_depth=int(pipeline["max_depth"]),
        learning_rate=float(pipeline["learning_rate"]),
        subsample=float(pipeline["subsample"]),
        colsample_bytree=float(pipeline["colsample_bytree"]),
        min_child_weight=float(pipeline["min_child_weight"]),
        reg_alpha=float(pipeline["reg_alpha"]),
        reg_lambda=float(pipeline["reg_lambda"]),
        gamma=float(pipeline["gamma"]),
        max_bin=int(pipeline["max_bin"]),
    )


def load_locked_split_data(
    matrix_path: str | Path,
    split_manifest: str | Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    list[str],
]:
    matrix = np.load(Path(matrix_path), mmap_mode="r")
    if matrix.ndim != 2:
        raise FinalEvaluationLockError("expression cache must be a 2-D matrix")
    rows = read_split_manifest(split_manifest)

    try:
        indices = [int(row["cache_index"]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise FinalEvaluationLockError(
            "split manifest contains a non-integer cache index"
        ) from exc
    participants = [row["participant_barcode"] for row in rows]
    if len(set(indices)) != len(indices):
        raise FinalEvaluationLockError(
            "split manifest contains duplicate cache indices"
        )
    if len(set(participants)) != len(participants):
        raise FinalEvaluationLockError(
            "split manifest contains duplicate participants"
        )
    if any(index < 0 or index >= matrix.shape[0] for index in indices):
        raise FinalEvaluationLockError(
            "split manifest contains cache indices outside the expression matrix"
        )

    development = [row for row in rows if row["split"] == "development"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    if not development or not holdout:
        raise FinalEvaluationLockError(
            "split manifest must contain development and holdout samples"
        )

    development_indices = np.asarray(
        [int(row["cache_index"]) for row in development],
        dtype=np.int64,
    )
    holdout_indices = np.asarray(
        [int(row["cache_index"]) for row in holdout],
        dtype=np.int64,
    )
    y_development = np.asarray(
        [row["cancer_type"] for row in development],
        dtype=object,
    )
    y_holdout = np.asarray(
        [row["cancer_type"] for row in holdout],
        dtype=object,
    )
    development_classes = set(str(value) for value in np.unique(y_development))
    holdout_classes = set(str(value) for value in np.unique(y_holdout))
    if development_classes != holdout_classes:
        raise FinalEvaluationLockError(
            "development and holdout must contain the same cancer classes"
        )

    return (
        np.asarray(matrix[development_indices], dtype=np.float32),
        y_development,
        [row["participant_barcode"] for row in development],
        np.asarray(matrix[holdout_indices], dtype=np.float32),
        y_holdout,
        [row["participant_barcode"] for row in holdout],
    )


def _reserve_receipt(
    receipt_path: Path,
    *,
    lock_sha256: str,
    outdir: Path,
) -> dict[str, object]:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "holdout_access_started",
        "holdout_used": True,
        "lock_sha256": lock_sha256,
        "output_directory": str(outdir.resolve()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise FinalEvaluationLockError(
            f"final holdout already has a receipt; refusing a second evaluation: "
            f"{receipt_path}"
        ) from exc
    return payload


def _replace_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _probability_metrics(
    estimator,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray | None, dict[str, object]]:
    if not hasattr(estimator, "predict_proba"):
        return None, {
            "available": False,
            "multiclass_log_loss": None,
            "ovr_roc_auc_macro": None,
            "note": "The locked estimator does not expose predict_proba.",
        }

    probabilities = np.asarray(estimator.predict_proba(X_holdout), dtype=np.float64)
    fitted_classes = [str(value) for value in estimator.classes_]
    if probabilities.shape != (len(y_holdout), len(fitted_classes)):
        raise AssertionError("probability matrix shape does not match fitted classes")
    if fitted_classes != classes:
        order = [fitted_classes.index(label) for label in classes]
        probabilities = probabilities[:, order]

    metrics: dict[str, object] = {
        "available": True,
        "multiclass_log_loss": float(
            log_loss(y_holdout, probabilities, labels=classes)
        ),
        "ovr_roc_auc_macro": None,
        "note": None,
    }
    try:
        if len(classes) == 2:
            binary_truth = np.asarray(y_holdout == classes[1], dtype=np.int64)
            metrics["ovr_roc_auc_macro"] = float(
                roc_auc_score(binary_truth, probabilities[:, 1])
            )
        else:
            metrics["ovr_roc_auc_macro"] = float(
                roc_auc_score(
                    y_holdout,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError as exc:
        metrics["note"] = f"ROC-AUC unavailable: {exc}"
    return probabilities, metrics


def _metric_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: list[str],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
    )
    counts = confusion_matrix(y_true, y_pred, labels=classes)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals != 0,
    )
    return (
        {
            "primary_metric": PRIMARY_METRIC,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "per_class": [
                {
                    "class": label,
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(class_f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(classes)
            ],
        },
        counts,
        normalized,
    )


def _write_predictions(
    path: Path,
    participants: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: list[str],
    probabilities: np.ndarray | None,
) -> None:
    probability_headers = [f"probability_{label}" for label in classes]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "participant_barcode",
                "true_label",
                "predicted_label",
                "correct",
                *probability_headers,
            ]
        )
        for index, participant in enumerate(participants):
            probability_values: list[str] = []
            if probabilities is not None:
                probability_values = [
                    f"{float(value):.12g}" for value in probabilities[index]
                ]
            else:
                probability_values = [""] * len(classes)
            writer.writerow(
                [
                    participant,
                    str(y_true[index]),
                    str(y_pred[index]),
                    str(bool(y_true[index] == y_pred[index])).lower(),
                    *probability_values,
                ]
            )


def _write_confusion_table(
    path: Path,
    classes: list[str],
    counts: np.ndarray,
    normalized: np.ndarray,
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["true_label", "predicted_label", "count", "row_fraction"]
        )
        for true_index, true_label in enumerate(classes):
            for predicted_index, predicted_label in enumerate(classes):
                writer.writerow(
                    [
                        true_label,
                        predicted_label,
                        int(counts[true_index, predicted_index]),
                        f"{float(normalized[true_index, predicted_index]):.12g}",
                    ]
                )


def _write_confusion_svg(
    path: Path,
    classes: list[str],
    counts: np.ndarray,
    normalized: np.ndarray,
) -> None:
    cell = 58
    left = 128
    top = 132
    width = left + cell * len(classes) + 36
    height = top + cell * len(classes) + 74
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18" '
        'font-weight="bold">Final holdout confusion matrix</text>',
        f'<text x="{left + cell * len(classes) / 2:.1f}" y="54" '
        'text-anchor="middle" font-family="sans-serif" font-size="13">'
        'Predicted class</text>',
        f'<text x="20" y="{top + cell * len(classes) / 2:.1f}" '
        'text-anchor="middle" font-family="sans-serif" font-size="13" '
        f'transform="rotate(-90 20 {top + cell * len(classes) / 2:.1f})">'
        'True class</text>',
    ]
    for index, label in enumerate(classes):
        x = left + index * cell + cell / 2
        y = top - 10
        elements.append(
            f'<text x="{x:.1f}" y="{y}" text-anchor="end" '
            'font-family="sans-serif" font-size="11" '
            f'transform="rotate(-45 {x:.1f} {y})">{escape(label)}</text>'
        )
        row_y = top + index * cell + cell / 2 + 4
        elements.append(
            f'<text x="{left - 10}" y="{row_y:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{escape(label)}</text>'
        )
    for row in range(len(classes)):
        for column in range(len(classes)):
            x = left + column * cell
            y = top + row * cell
            fraction = float(normalized[row, column])
            opacity = 0.08 + 0.82 * fraction
            text_color = "white" if fraction >= 0.55 else "#111827"
            elements.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="#2563eb" fill-opacity="{opacity:.3f}" '
                    'stroke="#d1d5db"/>',
                    f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 - 2:.1f}" '
                    f'text-anchor="middle" font-family="sans-serif" '
                    f'font-size="13" font-weight="bold" fill="{text_color}">'
                    f'{int(counts[row, column])}</text>',
                    f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 15:.1f}" '
                    f'text-anchor="middle" font-family="sans-serif" '
                    f'font-size="10" fill="{text_color}">{fraction:.1%}</text>',
                ]
            )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_per_class_f1_svg(
    path: Path,
    per_class: list[dict[str, object]],
) -> None:
    width = 760
    left = 116
    right = 32
    bar_height = 24
    gap = 15
    top = 62
    height = top + len(per_class) * (bar_height + gap) + 44
    plot_width = width - left - right
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18" '
        'font-weight="bold">Final holdout per-class F1</text>',
    ]
    for index, row in enumerate(per_class):
        value = float(row["f1"])
        y = top + index * (bar_height + gap)
        elements.extend(
            [
                f'<text x="{left - 10}" y="{y + 17}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">'
                f'{escape(str(row["class"]))}</text>',
                f'<rect x="{left}" y="{y}" width="{plot_width}" '
                f'height="{bar_height}" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y}" width="{plot_width * value:.2f}" '
                f'height="{bar_height}" fill="#2563eb"/>',
                f'<text x="{left + plot_width * value + 6:.2f}" '
                f'y="{y + 17}" font-family="sans-serif" font-size="11">'
                f'{value:.3f}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def run_final_evaluation(
    lock_path: str | Path,
    matrix_path: str | Path,
    split_manifest: str | Path,
    gene_table: str | Path,
    outdir: str | Path,
    receipt_path: str | Path,
    *,
    evidence_paths: Iterable[str | Path] | None = None,
) -> dict[str, object]:
    lock_payload = verify_final_evaluation_lock(
        lock_path,
        matrix_path,
        split_manifest,
        gene_table,
        evidence_paths=evidence_paths,
    )
    estimator = build_locked_pipeline(lock_payload)

    output_dir = Path(outdir)
    receipt = Path(receipt_path)
    resolved_output = output_dir.resolve()
    resolved_receipt = receipt.resolve()
    if resolved_receipt == resolved_output or resolved_output in resolved_receipt.parents:
        raise FinalEvaluationLockError(
            "receipt must be outside the final output directory"
        )
    if output_dir.exists():
        raise FinalEvaluationLockError(
            f"refusing to reuse an existing final output directory: {output_dir}"
        )

    receipt_payload = _reserve_receipt(
        receipt,
        lock_sha256=str(lock_payload["lock_sha256"]),
        outdir=output_dir,
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        (
            X_development,
            y_development,
            development_participants,
            X_holdout,
            y_holdout,
            holdout_participants,
        ) = load_locked_split_data(matrix_path, split_manifest)

        if lock_payload["selection"]["pipeline"]["family"] == "pca_logistic":
            requested = int(
                lock_payload["selection"]["pipeline"]["pca_components"]
            )
            maximum = min(X_development.shape[1], len(X_development) - 1)
            if requested > maximum:
                raise FinalEvaluationLockError(
                    f"locked PCA components ({requested}) exceed the final-fit "
                    f"maximum ({maximum})"
                )

        fit_started = time.perf_counter()
        estimator.fit(X_development, y_development)
        fit_seconds = time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        predictions = np.asarray(estimator.predict(X_holdout), dtype=object)
        predict_seconds = time.perf_counter() - predict_started
        if len(predictions) != len(y_holdout):
            raise AssertionError("final prediction count does not match holdout size")

        classes = sorted(str(value) for value in np.unique(y_development))
        metrics, counts, normalized = _metric_payload(
            y_holdout,
            predictions,
            classes=classes,
        )
        probabilities, probability_metrics = _probability_metrics(
            estimator,
            X_holdout,
            y_holdout,
            classes,
        )
        metrics["probability_metrics"] = probability_metrics

        predictions_path = output_dir / "final_predictions.tsv"
        confusion_path = output_dir / "final_confusion.tsv"
        metrics_path = output_dir / "final_metrics.json"
        model_path = output_dir / "final_pipeline.joblib"
        confusion_svg = output_dir / "final_confusion.svg"
        per_class_svg = output_dir / "final_per_class_f1.svg"

        _write_predictions(
            predictions_path,
            holdout_participants,
            y_holdout,
            predictions,
            classes=classes,
            probabilities=probabilities,
        )
        _write_confusion_table(confusion_path, classes, counts, normalized)
        _write_confusion_svg(confusion_svg, classes, counts, normalized)
        _write_per_class_f1_svg(per_class_svg, metrics["per_class"])
        joblib.dump(estimator, model_path, compress=3)

        payload: dict[str, object] = {
            "evaluation_scope": "locked_final_holdout_once",
            "holdout_used": True,
            "lock_sha256": lock_payload["lock_sha256"],
            "candidate_id": lock_payload["selection"]["candidate_id"],
            "pipeline": lock_payload["selection"]["pipeline"],
            "primary_metric": PRIMARY_METRIC,
            "n_development_samples": len(development_participants),
            "n_holdout_samples": len(holdout_participants),
            "classes": classes,
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "metrics": metrics,
            "confusion_counts": counts.astype(int).tolist(),
            "confusion_normalized_true": normalized.tolist(),
            "outputs": {
                "predictions": predictions_path.name,
                "confusion": confusion_path.name,
                "model": model_path.name,
                "confusion_svg": confusion_svg.name,
                "per_class_f1_svg": per_class_svg.name,
            },
            "interpretation_note": (
                "This is one locked retrospective TCGA holdout evaluation. "
                "It is not prospective clinical validation."
            ),
        }
        metrics_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload["outputs"]["metrics"] = metrics_path.name

        output_files = [
            predictions_path,
            confusion_path,
            metrics_path,
            model_path,
            confusion_svg,
            per_class_svg,
        ]
        receipt_payload.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_development_samples": len(development_participants),
                "n_holdout_samples": len(holdout_participants),
                "outputs": {
                    path.name: {
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in output_files
                },
            }
        )
        _replace_json(receipt, receipt_payload)
        return payload
    except Exception as exc:
        receipt_payload.update(
            {
                "status": "failed_after_holdout_reservation",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _replace_json(receipt, receipt_payload)
        raise
