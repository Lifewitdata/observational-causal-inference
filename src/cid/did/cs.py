"""Callaway & Sant'Anna (2021) group-time average treatment effects.

The estimand is ATT(g, t): the average effect on units first treated in period g, measured in
period t >= g. Everything is built from the *change* in a unit's outcome between its own baseline
period (g - 1) and t, which differences out time-invariant unit heterogeneity. The comparison
group is restricted to units **not yet treated as of t**, because a previously treated unit's
outcome path already contains the effect being estimated -- using it as a control contaminates
the counterfactual.

Four estimators target the same estimand and differ only in how the comparison group is
reweighted:

  simple : unweighted difference in mean outcome changes
  reg    : outcome-regression adjustment (g-computation using the control-group model)
  ipw    : inverse-probability reweighting from a logistic propensity model
  dr     : doubly robust; consistent if either nuisance model is correct

Standard errors come from a unit-level bootstrap of the whole cell, rather than three separately
derived analytic influence functions -- same guarantee, far less room for an algebra error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261013)


def logit(X: np.ndarray, y: np.ndarray, iters: int = 50, ridge: float = 1e-6) -> np.ndarray:
    """Fitted probabilities by IRLS. Local so the repo needs no statsmodels."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    Z = np.column_stack([np.ones(len(y)), X])
    y = np.asarray(y, float)
    b = np.zeros(Z.shape[1])
    for _ in range(iters):
        eta = np.clip(Z @ b, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-8, None)
        grad = Z.T @ (y - mu) - ridge * b
        hess = Z.T @ (Z * w[:, None]) + ridge * np.eye(Z.shape[1])
        step = np.linalg.solve(hess, grad)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(Z @ b, -30, 30))), 1e-6, 1 - 1e-6)


def _ols_coef(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return b


def _ols_predict(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones(len(X)), X]) if X.size else np.ones((len(X), 1))
    return Z @ b


def delta_frame(
    df: pd.DataFrame,
    *,
    g: float,
    t: int,
    outcome: str = "lemp",
    id_col: str = "countyreal",
    time_col: str = "year",
    g_col: str = "g",
    covars: list[str] | None = None,
    control: str = "not-yet-treated",
) -> pd.DataFrame:
    """One row per unit: its outcome change from its baseline to t, baseline covariates, group."""
    covars = list(covars or [])
    base = int(g - 1)
    wide = df.pivot_table(index=id_col, columns=time_col, values=outcome, aggfunc="mean")
    for col in (base, t):
        if col not in wide.columns:
            raise ValueError(f"panel lacks period {col} needed for ATT({g},{t})")
    out = pd.DataFrame({"delta": wide[t] - wide[base]})
    out = out.join(df[df[time_col] == base].drop_duplicates(id_col).set_index(id_col)[covars], how="left")
    out = out.join(df.drop_duplicates(id_col).set_index(id_col)[g_col], how="left")
    out = out.rename(columns={g_col: "_g"})
    out["treated"] = (out["_g"] == g).astype(int)
    if control == "not-yet-treated":
        eligible = (out["_g"] > t) | ~np.isfinite(out["_g"])
    elif control == "never-treated":
        eligible = ~np.isfinite(out["_g"])
    else:
        raise ValueError(f"unknown control rule {control!r}")
    out = out[eligible | (out.treated == 1)]
    out = out.dropna(subset=["delta", *covars])
    if out.treated.sum() == 0 or (out.treated == 0).sum() == 0:
        raise ValueError(f"no usable comparison group for ATT({g},{t})")
    return out


def _att_from_frame(f: pd.DataFrame, covars: list[str], estimator: str) -> float:
    y = f["delta"].to_numpy(float)
    D = f["treated"].to_numpy(float)
    X = f[covars].to_numpy(float) if covars else np.empty((len(f), 0))
    t_mask, c_mask = D == 1, D == 0

    if estimator == "simple" or not covars:
        return float(y[t_mask].mean() - y[c_mask].mean())

    if estimator == "reg":
        b0 = _ols_coef(y[c_mask], X[c_mask])
        return float(y[t_mask].mean() - _ols_predict(b0, X[t_mask]).mean())

    e = logit(X, t_mask.astype(float))
    e = np.clip(e, 0.05, 0.95)

    if estimator == "ipw":
        w_t = 1.0 / e[t_mask]
        w_c = e[c_mask] / (1 - e[c_mask])
        return float((y[t_mask] * w_t).sum() / w_t.sum() - (y[c_mask] * w_c).sum() / w_c.sum())

    if estimator == "dr":
        b0 = _ols_coef(y[c_mask], X[c_mask])
        resid = y - _ols_predict(b0, X)
        n_t = t_mask.sum()
        w_c = np.where(c_mask, e / (1 - e), 0.0)
        return float(((D * resid) - w_c * resid).sum() / n_t)

    raise ValueError(f"unknown estimator {estimator!r}")


def group_time_att(
    df: pd.DataFrame,
    *,
    g: float,
    t: int,
    outcome="lemp",
    id_col="countyreal",
    time_col="year",
    g_col="g",
    covars=None,
    estimator="dr",
    control="not-yet-treated",
    n_boot: int = 300,
    rng: np.random.Generator | None = None,
) -> dict:
    """ATT(g, t) plus a unit-level bootstrap standard error and percentile interval."""
    covars = list(covars or [])
    f = delta_frame(
        df, g=g, t=t, outcome=outcome, id_col=id_col, time_col=time_col, g_col=g_col,
        covars=covars, control=control,
    )
    units_t = np.where(f.treated.to_numpy() == 1)[0]
    units_c = np.where(f.treated.to_numpy() == 0)[0]
    att = _att_from_frame(f, covars, estimator)
    rng = rng or RNG

    unadjusted = estimator == "simple" or not covars
    if n_boot <= 0:
        if not unadjusted:
            raise ValueError("n_boot must be positive for covariate-adjusted estimators")
        y = f["delta"].to_numpy(float)
        mask = f.treated.to_numpy() == 1
        v1 = y[mask].var(ddof=1) if mask.sum() > 1 else 0.0
        v0 = y[~mask].var(ddof=1) if (~mask).sum() > 1 else 0.0
        se = float(np.sqrt(v1 / mask.sum() + v0 / (~mask).sum()))
        ci_low, ci_high = att - 1.96 * se, att + 1.96 * se
    else:
        vals = f[["delta", *(covars or [])]].to_numpy(float)
        draws = []
        for _ in range(n_boot):
            idx = np.concatenate(
                [
                    (rng or RNG).choice(units_t, len(units_t), replace=True),
                    (rng or RNG).choice(units_c, len(units_c), replace=True),
                ]
            )
            boot = pd.DataFrame(vals[idx], columns=["delta", *(covars or [])])
            boot["treated"] = np.r_[np.ones(len(units_t)), np.zeros(len(units_c))]
            try:
                draws.append(_att_from_frame(boot, covars, estimator))
            except Exception:
                continue
        draws = np.asarray(draws, float)
        if draws.size < 2:
            raise ValueError("bootstrap produced no usable replicates")
        se = float(draws.std(ddof=1))
        ci_low, ci_high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    return {
        "g": float(g),
        "t": int(t),
        "rel_time": int(t - g),
        "n_treated": int(len(units_t)),
        "n_control": int(len(units_c)),
        "att": att,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "estimator": estimator,
        "control_rule": control,
    }


def att_surface(
    df: pd.DataFrame,
    *,
    outcome="lemp",
    id_col="countyreal",
    time_col="year",
    g_col="g",
    covars=None,
    estimator="dr",
    control="not-yet-treated",
    min_treated: int = 5,
    n_boot: int = 300,
) -> pd.DataFrame:
    """Every ATT(g, t) cell for all treated cohorts and all post-treatment periods."""
    covars = list(covars or [])
    gs = sorted(v for v in np.unique(df[g_col]) if np.isfinite(v))
    years = sorted(df[time_col].unique())
    rng = np.random.default_rng(7)
    rows = []
    for g in gs:
        for t in years:
            if t < g:
                continue
            try:
                cell = group_time_att(
                    df, g=g, t=int(t), outcome=outcome, id_col=id_col, time_col=time_col,
                    g_col=g_col, covars=covars, estimator=estimator, control=control,
                    n_boot=n_boot, rng=rng,
                )
            except ValueError:
                continue
            if cell["n_treated"] < min_treated:
                continue
            rows.append(cell)
    if not rows:
        raise ValueError("no estimable ATT(g,t) cells")
    return pd.DataFrame(rows)


def aggregate(surface: pd.DataFrame, how: str = "simple") -> pd.DataFrame:
    """Collapse ATT(g, t) cells into a headline number or an event-time path.

    Weights are treated-unit counts, so a 131-county cohort dominates a 20-county cohort.
    Note these SEs treat cells as independent; `bootstrap.py` gives the honest overall CI.
    """
    s = surface.copy()
    w = s["n_treated"].astype(float)

    if how == "simple":
        return pd.DataFrame(
            [
                {
                    "label": "overall ATT",
                    "att": float(np.average(s.att, weights=w)),
                    "se": float(np.sqrt(np.sum((w / w.sum()) ** 2 * s.se**2))),
                    "n_cells": len(s),
                    "n_treated": int(s.n_treated.max()),
                }
            ]
        )

    if how == "event":
        rows = []
        for rel, blk in s.groupby("rel_time"):
            ww = blk.n_treated.astype(float)
            rows.append(
                {
                    "label": f"event time {int(rel):+d}",
                    "rel_time": int(rel),
                    "att": float(np.average(blk.att, weights=ww)),
                    "se": float(np.sqrt(np.sum((ww / ww.sum()) ** 2 * blk.se**2))),
                    "n_cells": len(blk),
                    "n_treated": int(ww.sum()),
                }
            )
        return pd.DataFrame(rows)

    if how == "group":
        rows = []
        for g, blk in s.groupby("g"):
            ww = blk.n_treated.astype(float)
            rows.append(
                {
                    "label": f"cohort {int(g)}",
                    "att": float(np.average(blk.att, weights=ww)),
                    "se": float(np.sqrt(np.sum((ww / ww.sum()) ** 2 * blk.se**2))),
                    "n_cells": len(blk),
                    "n_treated": int(blk.n_treated.max()),
                }
            )
        return pd.DataFrame(rows)

    raise ValueError(f"unknown aggregation {how!r}")
