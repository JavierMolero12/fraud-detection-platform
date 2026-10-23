"""scikit-learn pipelines. Preprocessing lives INSIDE the pipeline, so it is fitted on the
training rows only (leakage risk L6: a scaler or imputer fitted on all data leaks the
validation/test distribution)."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from fraudplatform.features.catalog import BOOLEAN, CATEGORICAL, NUMERIC


def signed_log1p(x):
    """Compress heavy tails (counts, amounts, seconds) for a linear model; keeps the sign."""
    return np.sign(x) * np.log1p(np.abs(x))


# Types that skops may load besides scikit-learn's own (models are saved with skops, not
# pickle, so that loading a model file cannot run arbitrary code).
SKOPS_TRUSTED_TYPES = [
    "fraudplatform.models.pipelines.signed_log1p",
    "numpy.dtype",
    "sklearn.tree._tree.Tree",  # random forest
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",  # HistGB
    "sklearn.utils.validation.check_array",  # wrapped in a partial by HistGB
    "functools.partial",
]


# Linear model: log_amount instead of amount_eur, and the hour as a category (the effect is
# U-shaped, a single linear coefficient cannot represent it).
LINEAR_NUMERIC = tuple(c for c in NUMERIC if c not in ("amount_eur", "local_hour"))
LINEAR_CATEGORICAL = (*CATEGORICAL, "local_hour")
# Trees are scale-invariant: the raw amount is enough, its log would be a duplicate.
TREE_NUMERIC = tuple(c for c in NUMERIC if c != "log_amount")


def model_columns(kind: str) -> list[str]:
    if kind == "linear":
        return [*LINEAR_NUMERIC, *BOOLEAN, *LINEAR_CATEGORICAL]
    return [*TREE_NUMERIC, *BOOLEAN, *CATEGORICAL]


def build_logistic_regression(params: dict, class_weight=None, random_state: int = 0) -> Pipeline:
    numeric = make_pipeline(
        # NULL means "no history" or "no device": keep that information as an indicator.
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        FunctionTransformer(signed_log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    boolean = SimpleImputer(
        strategy="constant", fill_value=0, add_indicator=True, keep_empty_features=True
    )
    categorical = OneHotEncoder(handle_unknown="ignore", min_frequency=50)
    pre = ColumnTransformer(
        [
            ("numeric", numeric, list(LINEAR_NUMERIC)),
            ("boolean", boolean, list(BOOLEAN)),
            ("categorical", categorical, list(LINEAR_CATEGORICAL)),
        ]
    )
    model = LogisticRegression(
        C=params.get("C", 1.0),
        max_iter=params.get("max_iter", 2000),
        class_weight=class_weight,
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("model", model)])


def _tree_preprocessor() -> ColumnTransformer:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
    )
    pre = ColumnTransformer(
        [
            ("categorical", encoder, list(CATEGORICAL)),
            ("rest", "passthrough", [*TREE_NUMERIC, *BOOLEAN]),  # NaN handled natively
        ],
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")


def build_random_forest(params: dict, class_weight=None, random_state: int = 0) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        min_samples_leaf=params.get("min_samples_leaf", 50),
        max_features=params.get("max_features", "sqrt"),
        max_samples=params.get("max_samples", 0.3),
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("pre", _tree_preprocessor()), ("model", model)])


def build_hist_gradient_boosting(
    params: dict, class_weight=None, random_state: int = 0
) -> Pipeline:
    model = HistGradientBoostingClassifier(
        learning_rate=params.get("learning_rate", 0.05),
        max_iter=params.get("max_iter", 400),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        l2_regularization=params.get("l2_regularization", 1.0),
        categorical_features=list(CATEGORICAL),
        # No early stopping: its internal split is random, i.e. not temporal. Phase 7 tunes
        # the number of iterations with time-ordered folds instead.
        early_stopping=False,
        class_weight=class_weight,
        random_state=random_state,
    )
    return Pipeline([("pre", _tree_preprocessor()), ("model", model)])
