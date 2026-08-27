from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

from joblib import Parallel, delayed, parallel_backend
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from threadpoolctl import threadpool_limits

from .benchmark import _metric_summary, load_development_data, resolve_n_jobs
from .feature_budget import (
    FEATURE_BUDGET_MODELS,
    GeneRecord,
    _class_coefficients,
    _selected_original_indices,
    build_feature_budget_pipeline,
    parse_gene_budget,
    read_gene_table,
)
from .splitting import DEFAULT_SEED, make_development_cv


FOCUSED_PAIRS: dict[str, tuple[str, str]] = {
    "luad_lusc": ("LUAD", "LUSC"),
    "kirc_kirp": ("KIRC", "KIRP"),
}


def parse_pair_name(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key not in FOCUSED_PAIRS:
        raise ValueError(
            f"unknown focused pair {value!r}; choose from {tuple(FOCUSED_PAIRS)}"
        )
    return key


def _signed_margin_toward(
    estimator,
    X: np.ndarray,
    *,
    positive_class: str,
) -> np.ndarray:
    if not hasattr(estimator, "decision_function"):
        return np.full(len(X), np.nan, dtype=np.float64)

    raw = np.asarray(estimator.decision_function(X), dtype=np.float64)
    classes = [str(value) for value in estimator.named_steps["model"].classes_]
    if len(classes) != 2 or positive_class not in classes:
        raise AssertionError("focused-pair estimator must expose exactly two expected classes")

    if raw.ndim == 1:
        # sklearn's binary decision_function is positive toward classes_[1].
        return raw if classes[1] == positive_class else -raw
    if raw.ndim == 2 and raw.shape[1] == 2:
        positive_index = classes.index(positive_class)
        negative_index = 1 - positive_index
        return raw[:, positive_index] - raw[:, negative_index]
    raise AssertionError(f"unexpected binary decision-function shape: {raw.shape}")


def _fit_pair_fold(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    fold_number: int,
    positive_class: str,
) -> dict[str, object]:
    estimator = clone(pipeline)
    estimator.fit(X[train_indices], y[train_indices])
    prediction = np.asarray(estimator.predict(X[test_indices]), dtype=object)
    margin = _signed_margin_toward(
        estimator,
        X[test_indices],
        positive_class=positive_class,
    )
    truth = y[test_indices]
    return {
        "fold": fold_number,
        "test_indices": np.asarray(test_indices, dtype=np.int64),
        "prediction": prediction,
        "margin": margin,
        "estimator": estimator,
        "metrics": {
            "macro_f1": float(f1_score(truth, prediction, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "accuracy": float(accuracy_score(truth, prediction)),
        },
    }


def _class_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
) -> list[dict[str, object]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
    )
    return [
        {
            "class": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]


def _gene_interpretation(
    fold_results: Sequence[dict[str, object]],
    *,
    genes: Sequence[GeneRecord],
    class_a: str,
    class_b: str,
) -> list[dict[str, object]]:
    n_features = len(genes)
    selected_counts = np.zeros(n_features, dtype=np.int64)
    coefficient_sum_a = np.zeros(n_features, dtype=np.float64)
    coefficient_sum_b = np.zeros(n_features, dtype=np.float64)
    coefficient_observations = np.zeros(n_features, dtype=np.int64)
    positive_signs_b = np.zeros(n_features, dtype=np.int64)
    negative_signs_b = np.zeros(n_features, dtype=np.int64)

    for result in fold_results:
        estimator = result["estimator"]
        selected = _selected_original_indices(estimator)
        classes, coefficients = _class_coefficients(estimator)
        if coefficients.shape[1] != len(selected):
            raise AssertionError("focused-pair coefficient count does not match selected genes")
        try:
            index_a = classes.index(class_a)
            index_b = classes.index(class_b)
        except ValueError as exc:
            raise AssertionError("focused-pair fold lost an expected class") from exc

        selected_counts[selected] += 1
        coefficient_sum_a[selected] += coefficients[index_a]
        coefficient_sum_b[selected] += coefficients[index_b]
        coefficient_observations[selected] += 1
        positive_signs_b[selected] += coefficients[index_b] > 0
        negative_signs_b[selected] += coefficients[index_b] < 0

    rows: list[dict[str, object]] = []
    n_folds = len(fold_results)
    for gene_index in np.flatnonzero(selected_counts):
        observations = int(coefficient_observations[gene_index])
        mean_a = float(coefficient_sum_a[gene_index] / observations)
        mean_b = float(coefficient_sum_b[gene_index] / observations)
        nonzero_signs = int(positive_signs_b[gene_index] + negative_signs_b[gene_index])
        sign_consistency = (
            float(
                max(positive_signs_b[gene_index], negative_signs_b[gene_index])
                / nonzero_signs
            )
            if nonzero_signs
            else 0.0
        )
        direction = class_b if mean_b > 0 else class_a if mean_b < 0 else "neutral"
        gene = genes[int(gene_index)]
        rows.append(
            {
                "gene_index": gene.gene_index,
                "source_gene_id": gene.source_gene_id,
                "symbol": gene.symbol,
                "entrez_id": gene.entrez_id,
                "selected_folds": int(selected_counts[gene_index]),
                "selection_frequency": float(selected_counts[gene_index] / n_folds),
                f"mean_coefficient_{class_a}": mean_a,
                f"mean_coefficient_{class_b}": mean_b,
                "mean_abs_coefficient": float((abs(mean_a) + abs(mean_b)) / 2.0),
                "direction": direction,
                "sign_consistency": sign_consistency,
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["selection_frequency"]),
            -float(row["mean_abs_coefficient"]),
            int(row["gene_index"]),
        )
    )
    return rows


def evaluate_focused_pair(
    X: np.ndarray,
    y: np.ndarray,
    participants: Sequence[str],
    genes: Sequence[GeneRecord],
    pair_name: str,
    *,
    model_name: str = "elastic_net",
    gene_budget: int | str = 1_000,
    cv_folds: int = 5,
    n_jobs: int = 0,
    negative_policy: str = "error",
    scaler: str = "standard",
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    pair_key = parse_pair_name(pair_name)
    if model_name not in FEATURE_BUDGET_MODELS:
        raise ValueError(f"focused-pair model must be one of {FEATURE_BUDGET_MODELS}")
    if len(X) != len(y) or len(y) != len(participants):
        raise ValueError("X, y, and participants must have matching sample counts")
    if X.ndim != 2:
        raise ValueError("focused-pair expression matrix must be 2-D")
    if X.shape[1] != len(genes):
        raise ValueError("gene metadata count does not match expression features")

    class_a, class_b = FOCUSED_PAIRS[pair_key]
    mask = np.isin(y, [class_a, class_b])
    pair_indices = np.flatnonzero(mask)
    if not len(pair_indices):
        raise ValueError(f"development data contain no samples for {class_a}/{class_b}")

    X_pair = np.asarray(X[pair_indices], dtype=np.float32)
    y_pair = np.asarray(y[pair_indices], dtype=object)
    participant_pair = [participants[int(index)] for index in pair_indices]
    counts = {label: int(np.count_nonzero(y_pair == label)) for label in (class_a, class_b)}
    too_small = {label: count for label, count in counts.items() if count < cv_folds}
    if too_small:
        raise ValueError(
            f"each focused-pair class needs at least {cv_folds} development samples: {too_small}"
        )

    budget = parse_gene_budget(gene_budget)
    pipeline = build_feature_budget_pipeline(
        model_name,
        budget,
        negative_policy=negative_policy,
        scaler=scaler,
        seed=seed,
    )
    cv = make_development_cv(n_splits=cv_folds, seed=seed)
    splits = list(cv.split(X_pair, y_pair))
    jobs = resolve_n_jobs(n_jobs, cv_folds=cv_folds)

    started = time.perf_counter()
    with threadpool_limits(limits=1), parallel_backend("loky", inner_max_num_threads=1):
        fold_results = Parallel(n_jobs=jobs)(
            delayed(_fit_pair_fold)(
                pipeline,
                X_pair,
                y_pair,
                np.asarray(train_indices, dtype=np.int64),
                np.asarray(test_indices, dtype=np.int64),
                fold_number=fold_number,
                positive_class=class_b,
            )
            for fold_number, (train_indices, test_indices) in enumerate(splits, start=1)
        )
    wall_seconds = time.perf_counter() - started

    predictions = np.empty(len(y_pair), dtype=object)
    fold_numbers = np.zeros(len(y_pair), dtype=np.int64)
    margins = np.full(len(y_pair), np.nan, dtype=np.float64)
    seen = np.zeros(len(y_pair), dtype=np.int64)
    for result in fold_results:
        test_indices = np.asarray(result["test_indices"], dtype=np.int64)
        predictions[test_indices] = np.asarray(result["prediction"], dtype=object)
        fold_numbers[test_indices] = int(result["fold"])
        margins[test_indices] = np.asarray(result["margin"], dtype=np.float64)
        seen[test_indices] += 1
    if not np.all(seen == 1):
        raise AssertionError("focused-pair out-of-fold predictions must cover each sample exactly once")

    labels = [class_a, class_b]
    counts_matrix = confusion_matrix(y_pair, predictions, labels=labels)
    row_totals = counts_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts_matrix,
        row_totals,
        out=np.zeros_like(counts_matrix, dtype=np.float64),
        where=row_totals != 0,
    )

    fold_metric_names = ("macro_f1", "balanced_accuracy", "accuracy")
    fold_metrics = {
        name: _metric_summary(
            np.asarray([result["metrics"][name] for result in fold_results], dtype=np.float64)
        )
        for name in fold_metric_names
    }
    global_metrics = {
        "macro_f1": float(f1_score(y_pair, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_pair, predictions)),
        "accuracy": float(accuracy_score(y_pair, predictions)),
    }

    prediction_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for index, participant in enumerate(participant_pair):
        correct = bool(predictions[index] == y_pair[index])
        margin = float(margins[index]) if np.isfinite(margins[index]) else None
        row = {
            "participant_barcode": participant,
            "true_label": str(y_pair[index]),
            "predicted_label": str(predictions[index]),
            "fold": int(fold_numbers[index]),
            f"signed_margin_toward_{class_b}": margin,
            "absolute_margin": abs(margin) if margin is not None else None,
            "correct": correct,
        }
        prediction_rows.append(row)
        if not correct:
            error_rows.append(dict(row))
    error_rows.sort(
        key=lambda row: (
            -(float(row["absolute_margin"]) if row["absolute_margin"] is not None else -1.0),
            str(row["participant_barcode"]),
        )
    )

    gene_rows = _gene_interpretation(
        fold_results,
        genes=genes,
        class_a=class_a,
        class_b=class_b,
    )

    return {
        "pair": pair_key,
        "class_a": class_a,
        "class_b": class_b,
        "model": model_name,
        "gene_budget": budget,
        "n_samples": int(len(y_pair)),
        "class_counts": counts,
        "cv_folds": cv_folds,
        "n_jobs": jobs,
        "seed": seed,
        "negative_policy": negative_policy,
        "scaler": scaler,
        "wall_seconds": float(wall_seconds),
        "metrics": global_metrics,
        "fold_metrics": fold_metrics,
        "per_class": _class_metric_rows(y_pair, predictions, labels),
        "confusion_counts": counts_matrix.astype(int).tolist(),
        "confusion_normalized_true": normalized.tolist(),
        "predictions": prediction_rows,
        "errors": error_rows,
        "genes": gene_rows,
    }


def _write_outputs(
    outdir: Path,
    studies: Sequence[dict[str, object]],
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "focused_pair_metrics.tsv"
    confusion_path = outdir / "focused_pair_confusion.tsv"
    predictions_path = outdir / "focused_pair_predictions.tsv"
    errors_path = outdir / "focused_pair_errors.tsv"
    genes_path = outdir / "focused_pair_genes.tsv"

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "pair",
                "class_a",
                "class_b",
                "model",
                "gene_budget",
                "n_samples",
                "n_errors",
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
                "wall_seconds",
            ]
        )
        for study in studies:
            writer.writerow(
                [
                    study["pair"],
                    study["class_a"],
                    study["class_b"],
                    study["model"],
                    study["gene_budget"],
                    study["n_samples"],
                    len(study["errors"]),
                    f"{float(study['metrics']['macro_f1']):.12g}",
                    f"{float(study['metrics']['balanced_accuracy']):.12g}",
                    f"{float(study['metrics']['accuracy']):.12g}",
                    f"{float(study['wall_seconds']):.12g}",
                ]
            )

    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["pair", "model", "true_label", "predicted_label", "count", "row_fraction"])
        for study in studies:
            labels = [study["class_a"], study["class_b"]]
            counts = study["confusion_counts"]
            normalized = study["confusion_normalized_true"]
            for true_index, true_label in enumerate(labels):
                for pred_index, predicted_label in enumerate(labels):
                    writer.writerow(
                        [
                            study["pair"],
                            study["model"],
                            true_label,
                            predicted_label,
                            counts[true_index][pred_index],
                            f"{float(normalized[true_index][pred_index]):.12g}",
                        ]
                    )

    prediction_header = [
        "pair",
        "model",
        "participant_barcode",
        "true_label",
        "predicted_label",
        "fold",
        "signed_margin_toward_class_b",
        "absolute_margin",
        "correct",
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle, errors_path.open(
        "w", encoding="utf-8", newline=""
    ) as error_handle:
        prediction_writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        error_writer = csv.writer(error_handle, delimiter="\t", lineterminator="\n")
        prediction_writer.writerow(prediction_header)
        error_writer.writerow(prediction_header)
        for study in studies:
            margin_key = f"signed_margin_toward_{study['class_b']}"
            for collection, writer in (
                (study["predictions"], prediction_writer),
                (study["errors"], error_writer),
            ):
                for row in collection:
                    writer.writerow(
                        [
                            study["pair"],
                            study["model"],
                            row["participant_barcode"],
                            row["true_label"],
                            row["predicted_label"],
                            row["fold"],
                            "" if row[margin_key] is None else f"{float(row[margin_key]):.12g}",
                            "" if row["absolute_margin"] is None else f"{float(row['absolute_margin']):.12g}",
                            str(bool(row["correct"])).lower(),
                        ]
                    )

    with genes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "pair",
                "model",
                "gene_index",
                "source_gene_id",
                "symbol",
                "entrez_id",
                "selected_folds",
                "selection_frequency",
                "mean_coefficient_class_a",
                "mean_coefficient_class_b",
                "mean_abs_coefficient",
                "direction",
                "sign_consistency",
            ]
        )
        for study in studies:
            coefficient_a = f"mean_coefficient_{study['class_a']}"
            coefficient_b = f"mean_coefficient_{study['class_b']}"
            for row in study["genes"]:
                writer.writerow(
                    [
                        study["pair"],
                        study["model"],
                        row["gene_index"],
                        row["source_gene_id"],
                        row["symbol"],
                        row["entrez_id"],
                        row["selected_folds"],
                        f"{float(row['selection_frequency']):.12g}",
                        f"{float(row[coefficient_a]):.12g}",
                        f"{float(row[coefficient_b]):.12g}",
                        f"{float(row['mean_abs_coefficient']):.12g}",
                        row["direction"],
                        f"{float(row['sign_consistency']):.12g}",
                    ]
                )

    return {
        "metrics": metrics_path.name,
        "confusion": confusion_path.name,
        "predictions": predictions_path.name,
        "errors": errors_path.name,
        "genes": genes_path.name,
    }


def run_focused_pair_studies(
    matrix_path: str | Path,
    split_manifest: str | Path,
    gene_table: str | Path,
    outdir: str | Path,
    *,
    pairs: Iterable[str] = FOCUSED_PAIRS,
    models: Iterable[str] = ("elastic_net",),
    gene_budget: int | str = 1_000,
    cv_folds: int = 5,
    n_jobs: int = 0,
    negative_policy: str = "error",
    scaler: str = "standard",
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    pair_keys = [parse_pair_name(value) for value in pairs]
    if not pair_keys:
        raise ValueError("at least one focused pair is required")
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("focused pair list contains duplicates")

    model_names = list(models)
    if not model_names:
        raise ValueError("at least one focused-pair model is required")
    unknown_models = sorted(set(model_names) - set(FEATURE_BUDGET_MODELS))
    if unknown_models:
        raise ValueError(f"unknown focused-pair models: {unknown_models}")
    if len(set(model_names)) != len(model_names):
        raise ValueError("focused-pair model list contains duplicates")

    X, y, participants = load_development_data(matrix_path, split_manifest)
    genes = read_gene_table(gene_table, expected_features=X.shape[1])
    studies = [
        evaluate_focused_pair(
            X,
            y,
            participants,
            genes,
            pair_key,
            model_name=model_name,
            gene_budget=gene_budget,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            negative_policy=negative_policy,
            scaler=scaler,
            seed=seed,
        )
        for pair_key in pair_keys
        for model_name in model_names
    ]

    ranking = sorted(
        (
            {
                "pair": study["pair"],
                "model": study["model"],
                "macro_f1": study["metrics"]["macro_f1"],
                "balanced_accuracy": study["metrics"]["balanced_accuracy"],
                "n_errors": len(study["errors"]),
            }
            for study in studies
        ),
        key=lambda row: (float(row["macro_f1"]), str(row["pair"]), str(row["model"])),
    )

    output_dir = Path(outdir)
    outputs = _write_outputs(output_dir, studies)
    payload: dict[str, object] = {
        "evaluation_scope": "development_cross_validation_only",
        "holdout_used": False,
        "matrix": str(matrix_path),
        "split_manifest": str(split_manifest),
        "gene_table": str(gene_table),
        "pairs": pair_keys,
        "models": model_names,
        "gene_budget": parse_gene_budget(gene_budget),
        "cv_folds": cv_folds,
        "seed": seed,
        "ranking_hardest_first": ranking,
        "studies": studies,
        "outputs": outputs,
        "interpretation_note": (
            "Gene coefficients and selection stability are predictive associations within "
            "development-set CV; they are not causal or clinically validated biomarkers."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "focused_pairs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
