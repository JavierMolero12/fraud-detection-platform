"""Model explanations: permutation importance (global) and SHAP values (local and global).

Why not shap.TreeExplainer? It is the fast, exact method for tree ensembles, but on
HistGradientBoosting with NATIVE categorical splits it silently returns wrong values: the
attributions do not add up to the model output (errors up to ~4.6 log-odds on this model),
because categorical splits are read as numeric thresholds. Nothing warns about it; checking
additivity does. So the explanations here use shap.PermutationExplainer, which only calls the
model and is exact for any model, at a higher compute cost. Every explanation is checked:
base value + sum of attributions must equal the model's log-odds output.

What SHAP values are: for one prediction, how much each feature moved the model output (here
in log-odds) away from the average output on a background sample, splitting interaction
effects fairly between the features involved.
What they are NOT: causal effects ("changing X would change the risk by ..."), a property of
the fraud itself, or independent of the background data. Correlated features share credit
arbitrarily between them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline


def split_pipeline(model: Pipeline):
    """(preprocessor, estimator): explanations work on the preprocessed columns, which map
    one-to-one to the original features for the tree pipeline (ordinal encoding)."""
    return model.named_steps["pre"], model.named_steps["model"]


def shap_values(
    model: Pipeline,
    rows: pd.DataFrame,
    background: pd.DataFrame,
    max_evals: int = 400,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Exact-by-construction SHAP values in log-odds for `rows`.

    Returns (attributions [rows x features], base values, model log-odds) so that
    base + attributions.sum(axis=1) == log-odds can be checked by the caller.
    """
    pre, estimator = split_pipeline(model)
    x_rows = pre.transform(rows)
    x_background = pre.transform(background)
    explainer = shap.PermutationExplainer(
        estimator.decision_function,
        shap.maskers.Independent(x_background, max_samples=len(x_background)),
        seed=seed,
    )
    explanation = explainer(x_rows, max_evals=max(max_evals, 2 * x_rows.shape[1] + 1), silent=True)
    attributions = pd.DataFrame(explanation.values, columns=x_rows.columns, index=rows.index)
    return attributions, np.asarray(explanation.base_values), estimator.decision_function(x_rows)


def additivity_error(attributions: pd.DataFrame, base, log_odds) -> float:
    return float(np.max(np.abs(attributions.sum(axis=1).to_numpy() + base - log_odds)))


def single_permutation_importance(
    model: Pipeline, X: pd.DataFrame, y, n_repeats: int, seed: int
) -> pd.DataFrame:
    """Drop in PR-AUC when one column is shuffled. Can underestimate features with a
    correlated twin (the model falls back on the twin), which is why the grouped version is
    reported too. On this gradient boosting the effect turned out small: with an exact
    duplicate of a feature, single-feature importance was within ~0.01 of the grouped one."""
    result = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    return pd.DataFrame(
        {"importance": result.importances_mean, "std": result.importances_std}, index=X.columns
    ).sort_values("importance", ascending=False)


def grouped_permutation_importance(
    model: Pipeline, X: pd.DataFrame, y, groups: dict[str, list[str]], n_repeats: int, seed: int
) -> pd.DataFrame:
    """Shuffle every column of a group with the SAME row permutation, so that correlated
    features inside the group cannot stand in for each other."""
    y = np.asarray(y)
    baseline = average_precision_score(y, model.predict_proba(X)[:, 1])
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            permuted = X.copy()
            order = rng.permutation(len(X))
            permuted[cols] = X[cols].iloc[order].to_numpy()
            drops.append(baseline - average_precision_score(y, model.predict_proba(permuted)[:, 1]))
        rows.append(
            {
                "group": name,
                "features": len(cols),
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
