from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


DEVELOPMENT_SCOPE = "development_cross_validation_only"


class CandidateComparisonError(ValueError):
    """Raised when candidate evidence is unsafe or has an unknown schema."""


def _metric(candidate: Mapping[str, Any], name: str) -> tuple[float, float | None]:
    metrics = candidate.get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get(name), Mapping):
        raise CandidateComparisonError(f"candidate is missing metrics.{name}")
    summary = metrics[name]
    mean = summary.get("mean")
    std = summary.get("std")
    if isinstance(mean, bool) or not isinstance(mean, (int, float)):
        raise CandidateComparisonError(f"metrics.{name}.mean must be numeric")
    if std is not None and (isinstance(std, bool) or not isinstance(std, (int, float))):
        raise CandidateComparisonError(f"metrics.{name}.std must be numeric or null")
    return float(mean), None if std is None else float(std)


def _base_row(path: Path, candidate: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    macro_mean, macro_std = _metric(candidate, "macro_f1")
    balanced_mean, _ = _metric(candidate, "balanced_accuracy")
    return {
        "candidate_id": candidate_id,
        "model": str(candidate.get("model", "")),
        "gene_budget": candidate.get("gene_budget"),
        "macro_f1_mean": macro_mean,
        "macro_f1_std": macro_std,
        "balanced_accuracy_mean": balanced_mean,
        "wall_seconds": candidate.get("wall_seconds"),
        "evidence_file": str(path),
    }


def _extract(path: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("evaluation_scope") != DEVELOPMENT_SCOPE:
        raise CandidateComparisonError(f"evidence is not development-only: {path}")
    if payload.get("holdout_used") is not False:
        raise CandidateComparisonError(f"evidence must declare holdout_used=false: {path}")

    if isinstance(payload.get("models"), list):
        rows = []
        for candidate in payload["models"]:
            if not isinstance(candidate, Mapping):
                raise CandidateComparisonError(f"invalid model entry in {path}")
            model = str(candidate.get("model", "unknown"))
            rows.append(_base_row(path, candidate, f"classical:{model}"))
        return rows

    if isinstance(payload.get("budgets"), list) and isinstance(payload.get("model"), str):
        rows = []
        for candidate in payload["budgets"]:
            if not isinstance(candidate, Mapping):
                raise CandidateComparisonError(f"invalid budget entry in {path}")
            budget = candidate.get("gene_budget")
            rows.append(
                _base_row(path, candidate, f"feature-budget:{payload['model']}:{budget}")
            )
        return rows

    if isinstance(payload.get("benchmark"), Mapping):
        candidate = payload["benchmark"]
        budget = candidate.get("gene_budget")
        device = candidate.get("resolved_device")
        return [_base_row(path, candidate, f"xgboost:{device}:{budget}")]

    raise CandidateComparisonError(f"unrecognized candidate-evidence schema: {path}")


def compare_candidates(evidence_paths: list[str | Path], outdir: str | Path) -> dict[str, Any]:
    if not evidence_paths:
        raise CandidateComparisonError("at least one evidence file is required")
    candidates: list[dict[str, Any]] = []
    for raw_path in evidence_paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise CandidateComparisonError(f"evidence must be a JSON object: {path}")
        candidates.extend(_extract(path, payload))
    if not candidates:
        raise CandidateComparisonError("no candidates were found")
    candidates.sort(key=lambda row: (-row["macro_f1_mean"], row["candidate_id"]))
    for rank, row in enumerate(candidates, start=1):
        row["rank_by_macro_f1"] = rank

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=False)
    tsv_path = output_dir / "candidate_comparison.tsv"
    fields = [
        "rank_by_macro_f1", "candidate_id", "model", "gene_budget",
        "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean",
        "wall_seconds", "evidence_file",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidates)

    payload = {
        "evaluation_scope": DEVELOPMENT_SCOPE,
        "holdout_used": False,
        "selection_made": False,
        "primary_metric": "macro_f1",
        "ranking_note": (
            "Candidates are ranked descriptively by development-CV mean macro F1. "
            "This artifact does not select a winner; uncertainty, stability, complexity, "
            "and compute cost must be reviewed before writing the locked selection config."
        ),
        "candidates": candidates,
        "output": tsv_path.name,
    }
    (output_dir / "candidate_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
