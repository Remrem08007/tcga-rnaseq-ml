import json

import pytest

from tcga_ml.candidate_comparison import CandidateComparisonError, compare_candidates


def _metric(mean):
    return {"mean": mean, "std": 0.02, "values": [mean - 0.02, mean + 0.02]}


def test_comparison_ranks_without_selecting_or_using_holdout(tmp_path):
    classical = tmp_path / "benchmark.json"
    classical.write_text(json.dumps({
        "evaluation_scope": "development_cross_validation_only",
        "holdout_used": False,
        "models": [
            {"model": "logistic_l2", "metrics": {"macro_f1": _metric(.91), "balanced_accuracy": _metric(.90)}, "wall_seconds": 3.0},
            {"model": "dummy", "metrics": {"macro_f1": _metric(.10), "balanced_accuracy": _metric(.10)}, "wall_seconds": .1},
        ],
    }))
    feature = tmp_path / "feature.json"
    feature.write_text(json.dumps({
        "evaluation_scope": "development_cross_validation_only",
        "holdout_used": False,
        "model": "elastic_net",
        "budgets": [{"model": "elastic_net", "gene_budget": 100, "metrics": {"macro_f1": _metric(.93), "balanced_accuracy": _metric(.92)}, "wall_seconds": 8.0}],
    }))
    payload = compare_candidates([classical, feature], tmp_path / "out")
    assert payload["holdout_used"] is False
    assert payload["selection_made"] is False
    assert payload["candidates"][0]["candidate_id"] == "feature-budget:elastic_net:100"
    assert (tmp_path / "out" / "candidate_comparison.tsv").is_file()


def test_comparison_refuses_holdout_touched_evidence(tmp_path):
    evidence = tmp_path / "bad.json"
    evidence.write_text(json.dumps({"evaluation_scope": "development_cross_validation_only", "holdout_used": True, "models": []}))
    with pytest.raises(CandidateComparisonError, match="holdout_used=false"):
        compare_candidates([evidence], tmp_path / "out")
