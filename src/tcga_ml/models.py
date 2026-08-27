from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .normalization import build_standardized_preprocessor
from .splitting import DEFAULT_SEED


MODEL_NAMES: tuple[str, ...] = (
    "dummy",
    "logistic_l2",
    "elastic_net",
    "linear_svm",
    "pca_logistic",
)


def _append_estimator(preprocessor: Pipeline, name: str, estimator) -> Pipeline:
    return Pipeline(steps=[*preprocessor.steps, (name, estimator)])


def build_model_pipeline(
    name: str,
    *,
    negative_policy: str = "error",
    scaler: str = "standard",
    seed: int = DEFAULT_SEED,
    pca_components: int = 100,
) -> Pipeline:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")

    pre = build_standardized_preprocessor(scaler=scaler, negative_policy=negative_policy)

    if name == "dummy":
        return _append_estimator(pre, "model", DummyClassifier(strategy="prior", random_state=seed))

    if name == "logistic_l2":
        model = LogisticRegression(
            C=1.0,
            l1_ratio=0.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        )
        return _append_estimator(pre, "model", model)

    if name == "elastic_net":
        model = LogisticRegression(
            C=1.0,
            l1_ratio=0.5,
            solver="saga",
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
        )
        return _append_estimator(pre, "model", model)

    if name == "linear_svm":
        model = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=10_000,
            random_state=seed,
        )
        return _append_estimator(pre, "model", model)

    if pca_components < 1:
        raise ValueError("pca_components must be >= 1")
    model = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
    )
    return Pipeline(
        steps=[
            *pre.steps,
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    svd_solver="randomized",
                    random_state=seed,
                ),
            ),
            ("model", model),
        ]
    )
