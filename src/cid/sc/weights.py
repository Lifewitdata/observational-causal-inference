"""Synthetic control for a single treated unit.

With one treated unit and no comparable control, the counterfactual has to be *constructed*: a
weighted average of untreated ("donor") states chosen to match the treated unit before
treatment. The weights solve a convex program --

    minimise   || x_treated - X_donor' w ||^2_V
    subject    w >= 0,   sum(w) = 1

where X_donor is J donors x k features and w is a J-vector. Non-negativity stops the
counterfactual from being an extrapolation, and the unit-sum constraint stops it from quietly
absorbing a level difference. Both are enforced with SLSQP rather than unconstrained least
squares, so the synthetic unit is always a *convex combination* of real donors -- which is also
what makes the weight table readable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TREATMENT_YEAR = 1989


def simplex_weights(
    x1: np.ndarray,
    X0: np.ndarray,
    V: np.ndarray | None = None,
    max_iter: int = 800,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve the synthetic-control weight program. `X0` is donors x features; returns J weights."""
    x1 = np.asarray(x1, float).ravel()
    X0 = np.atleast_2d(np.asarray(X0, float))
    J, k = X0.shape
    if x1.size != k:
        raise ValueError(f"target has {x1.size} features, donor matrix has {k}")
    V = np.eye(k) if V is None else np.asarray(V, float)
    if not np.isfinite(X0).all() or not np.isfinite(x1).all():
        raise ValueError("feature matrix contains NaN; drop incomplete predictors first")

    def objective(w):
        r = x1 - X0.T @ w
        return float(r @ V @ r)

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(J)},)
    res = minimize(
        objective, np.full(J, 1.0 / J), method="SLSQP",
        bounds=((0.0, 1.0),) * J, constraints=cons,
        options={"maxiter": max_iter, "ftol": tol},
    )
    if not res.success:
        raise RuntimeError(f"weight solver failed: {res.message}")
    w = np.clip(res.x, 0, None)
    if w.sum() <= 0:
        raise RuntimeError("solver returned degenerate weights")
    return w / w.sum()


def build_features(
    panel: pd.DataFrame,
    *,
    outcome: str = "cigsale",
    unit_col: str = "state",
    time_col: str = "year",
    pre_end: int = 1988,
    pre_start: int = 1970,
    extra: list[str] | None = None,
) -> pd.DataFrame:
    """One row per state: predictor summaries over the pre-treatment window.

    Includes the pre-period *level* and *slope* of the outcome itself, not just covariates --
    matching the trajectory is what makes a synthetic counterfactual credible, since it absorbs
    common time-varying shocks that no covariate in the data measures.

    Covariates with gaps (lnincome, beer, age15to24) are dropped when the mean is not available
    for every unit, so the weight program never sees a NaN.
    """
    extra = list(extra or [])
    pre = panel[(panel[time_col] <= pre_end) & (panel[time_col] >= pre_start)]
    rows = {}
    for unit, blk in pre.groupby(unit_col):
        blk = blk.sort_values(time_col)
        feats = {c: float(blk[c].mean(skipna=True)) for c in extra}
        feats[f"{outcome}_mean"] = float(blk[outcome].mean())
        if blk[time_col].nunique() > 1:
            slope = np.polyfit(blk[time_col].to_numpy(float), blk[outcome].to_numpy(float), 1)[0]
            feats[f"{outcome}_slope"] = float(slope)
        rows[unit] = feats

    F = pd.DataFrame.from_dict(rows, orient="index")
    complete = F.columns[F.notna().all(axis=0)]
    F.attrs["dropped_incomplete"] = [c for c in F.columns if c not in set(complete)]
    F.attrs["missing_share"] = {c: float(F[c].isna().mean()) for c in F.columns}
    return F[complete]


def fit(
    panel: pd.DataFrame,
    *,
    treated: str = "California",
    outcome: str = "cigsale",
    unit_col: str = "state",
    time_col: str = "year",
    treatment_year: int = TREATMENT_YEAR,
    predictors: list[str] | None = None,
    extra: list[str] | None = None,
    feature_start: int = 1970,
    feature_frame: pd.DataFrame | None = None,
    standardize: bool = True,
) -> dict:
    """Estimate weights from pre-treatment features, then build the post-period counterfactual."""
    predictors = list(predictors or []) + list(extra or [])
    donors = sorted(u for u in panel[unit_col].unique() if u != treated)

    if feature_frame is None:
        feature_frame = build_features(
            panel, outcome=outcome, unit_col=unit_col, time_col=time_col,
            pre_end=treatment_year - 1, pre_start=feature_start, extra=predictors,
        )
    missing_units = [treated, *donors]
    absent = [u for u in missing_units if u not in feature_frame.index]
    if absent:
        raise ValueError(f"feature frame missing units {absent}")

    cols = list(feature_frame.columns)
    F = feature_frame.loc[[treated] + donors, cols].astype(float)
    scale = F.loc[donors].std(ddof=1).replace(0, 1.0) if standardize else pd.Series(1.0, index=cols)
    Fs = F.div(scale, axis=1)

    w = simplex_weights(Fs.loc[treated].to_numpy(float), Fs.loc[donors].to_numpy(float))

    wide = panel.pivot_table(index=time_col, columns=unit_col, values=outcome, aggfunc="mean")
    actual = wide[treated].astype(float)
    counterfactual = pd.Series(wide[donors].to_numpy(float) @ w, index=wide.index)

    return _summarise(actual, counterfactual, pd.Series(w, index=donors), treatment_year)


def trajectory_fit(
    panel: pd.DataFrame,
    *,
    treated: str = "California",
    outcome: str = "cigsale",
    unit_col: str = "state",
    time_col: str = "year",
    treatment_year: int = TREATMENT_YEAR,
) -> dict:
    """Match the whole pre-period outcome path instead of summary statistics.

    A stricter test: the weights have to reproduce the shape of California's consumption history,
    with no covariates available to hide a bad fit.
    """
    wide = panel.pivot_table(index=time_col, columns=unit_col, values=outcome, aggfunc="mean")
    donors = [c for c in wide.columns if c != treated]
    pre = wide.loc[wide.index < treatment_year]

    M = pre[donors].T.to_numpy(float)
    x1 = pre[treated].to_numpy(float)
    scale = M.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    w = simplex_weights(x1 / scale, M / scale)

    actual = wide[treated].astype(float)
    counterfactual = pd.Series(wide[donors].to_numpy(float) @ w, index=wide.index)
    return _summarise(actual, counterfactual, pd.Series(w, index=donors), treatment_year)


def _summarise(actual: pd.Series, counterfactual: pd.Series, weights: pd.Series, treatment_year: int) -> dict:
    gap = actual - counterfactual
    pre = actual.index < treatment_year
    post = ~pre
    pre_rmspe = float(np.sqrt(np.mean(gap[pre] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap[post] ** 2)))
    return {
        "weights": weights,
        "active_donors": weights[weights > 1e-4].sort_values(ascending=False),
        "actual": actual,
        "counterfactual": counterfactual,
        "gap": gap,
        "cumulative_effect": float(gap[post].sum()),
        "mean_annual_effect": float(gap[post].mean()),
        "final_year_effect": float(gap.loc[max(gap.index[post])]),
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "rmspe_ratio": post_rmspe / pre_rmspe if pre_rmspe else np.inf,
        "n_active_donors": int((weights > 1e-4).sum()),
        "treatment_year": treatment_year,
        "donors": list(weights.index),
    }
