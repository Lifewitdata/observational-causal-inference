"""Naive two-way fixed effects and the dynamic (event-study) specification.

These are the estimators reported first in the notebooks *because* they are the ones a
practitioner reaches for by default, and under staggered adoption the static version averages
heterogeneous effects with weights that can be negative. Showing that failure is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cid import linreg
from cid.io import relative_time


def _dummies(s: pd.Series) -> np.ndarray:
    """One column per level, first level dropped. Collinearity is resolved downstream."""
    cats, levels = pd.factorize(s, sort=True)
    if len(levels) < 2:
        return np.empty((len(s), 0))
    return np.eye(len(levels))[cats][:, :-1]


def two_way_fe(
    df: pd.DataFrame,
    outcome: str,
    id_col: str,
    time_col: str,
    treat: str = "treated_post",
) -> linreg.Fit:
    """OLS of y on unit FE + time FE + treatment indicator, clustered by unit."""
    d = df.copy()
    d["treated_post"] = (d[treat] == 1).astype(float) if treat != "treated_post" else d[treat].astype(float)
    X = np.column_stack(
        [
            _dummies(d[id_col]),
            _dummies(d[time_col]),
            d["treated_post"].to_numpy(float),
        ]
    )
    names = [f"{id_col}_fe{i}" for i in range(_dummies(d[id_col]).shape[1])]
    names += [f"{time_col}_fe{i}" for i in range(_dummies(d[time_col]).shape[1])]
    names += ["ATT"]
    return linreg.ols(
        X,
        d[outcome].to_numpy(float),
        cluster=d[id_col].to_numpy(),
        add_const=True,
        names=names,
    )


def static_twfe(df: pd.DataFrame, outcome="lemp", id_col="countyreal", time_col="year", g_col="g"):
    """Post-period treatment flag, then two-way FE. Returns the ATT row."""
    d = df.copy()
    d["g"] = d[g_col]
    d["treated_post"] = ((d["g"] != np.inf) & (d[time_col] >= d["g"])).astype(int)
    fit = two_way_fe(d, outcome, id_col, time_col)
    row = fit.frame()
    att = row.loc[row.term == "ATT"].iloc[0]
    return {
        "estimate": float(att.coef),
        "std_error": float(att.std_error),
        "ci_low": float(att.ci_low),
        "ci_high": float(att.ci_high),
        "n_units": d[id_col].nunique(),
        "n_obs": len(d),
        "fit": fit,
    }


def event_study(
    df: pd.DataFrame,
    outcome: str = "lemp",
    id_col: str = "countyreal",
    time_col: str = "year",
    g_col: str = "g",
    lo: int = -4,
    hi: int = 3,
    ref: int = -1,
) -> pd.DataFrame:
    """Dynamic treatment effects with endpoints binned, never-treated units kept as controls.

    Two choices here are load-bearing rather than cosmetic.

    *Relative time is exactly collinear with unit and time effects if the never-treated units are
    dropped*: rel = year - cohort, year is spanned by the time dummies and cohort by the unit
    dummies, so the design loses rank and a least-squares solve returns an arbitrary member of the
    solution set. On this panel the pre-treatment coefficients come out at +2.3 log points, which
    is a number the data does not contain. Keeping the never-treated counties (all of whose
    relative-time dummies are zero) restores identification.

    *The reference period is the untreated period immediately before treatment*, so coefficients on
    negative relative times are the parallel-trends test and should hug zero. Endpoints are binned
    because with a finite panel the extreme lags are linear combinations of the others.
    """
    d = df.copy()
    d["rel"] = relative_time(d[time_col], d[g_col])
    d["rel_bin"] = d["rel"].clip(lo, hi)

    periods = sorted(int(p) for p in d["rel_bin"].dropna().unique() if p != ref)
    unit_dummies = _dummies(d[id_col])
    time_dummies = _dummies(d[time_col])
    rel_dummies = np.column_stack([(d["rel_bin"] == p).astype(float) for p in periods])
    X = np.column_stack([unit_dummies, time_dummies, rel_dummies])
    names = (
        [f"{id_col}_fe{i}" for i in range(unit_dummies.shape[1])]
        + [f"{time_col}_fe{i}" for i in range(time_dummies.shape[1])]
        + [f"rel_{p:+d}" for p in periods]
    )

    fit = linreg.ols(
        X,
        d[outcome].to_numpy(float),
        cluster=d[id_col].to_numpy(),
        add_const=True,
        names=names,
    )

    tab = fit.frame()
    out = tab[tab.term.str.startswith("rel_")].copy()
    out["rel_time"] = out.term.str.replace("rel_", "", regex=False).astype(int)
    out["endpoints_binned"] = out.rel_time.isin([lo, hi])
    pre = out[out.rel_time < 0]
    post = out[out.rel_time > 0]
    attribution = out.loc[out.dropped_for_collinearity, "term"].tolist()
    return out[["rel_time", "coef", "std_error", "ci_low", "ci_high", "endpoints_binned"]], {
        "pre_trend_joint": _joint_zero_test(pre, fit),
        "post_joint": _joint_zero_test(post, fit),
        "dropped": attribution,
        "n_units": d[id_col].nunique(),
        "n_obs": len(d),
    }


def _joint_zero_test(rows: pd.DataFrame, fit: linreg.Fit) -> dict:
    """Wald test that all coefficients in `rows` are jointly zero."""
    if rows.empty:
        return {"stat": np.nan, "df": 0, "p_value": np.nan}
    idx = [fit.names.index(n) for n in (f"rel_{int(r):+d}" for r in rows.rel_time) if n in fit.names]
    idx = [i for i in idx if i in set(fit.kept.tolist())]
    if not idx:
        return {"stat": np.nan, "df": 0, "p_value": np.nan}
    b = fit.beta[idx]
    V = fit.vcov[np.ix_(idx, idx)]
    try:
        stat = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        stat = float(b @ np.linalg.pinv(V) @ b)
    df = len(idx)
    from scipy.stats import chi2

    return {"stat": stat, "df": df, "p_value": float(chi2.sf(stat, df))}
