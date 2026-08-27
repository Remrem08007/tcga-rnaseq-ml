from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .feature_budget import FEATURE_BUDGET_MODELS, parse_gene_budget
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
