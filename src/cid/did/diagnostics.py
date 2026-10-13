"""Diagnostics: is the identifying assumption testable, and where does naive DiD break?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cid.did import cs


def summary(df: pd.DataFrame, *, id_col="countyreal", time_col="year", g_col="g", outcome="lemp") -> dict:
    units = df.drop_duplicates(id_col)
    treated = units[np.isfinite(units[g_col])]
    return {
        "n_units": int(units[id_col].nunique()),
        "n_periods": int(df[time_col].nunique()),
        "period_range": (int(df[time_col].min()), int(df[time_col].max())),
        "n_never_treated": int((~np.isfinite(units[g_col])).sum()),
        "n_treated": int(len(treated)),
        "cohorts": {int(g): int((treated[g_col] == g).sum()) for g in sorted(treated[g_col].unique())},
        "outcome_sd": float(df[outcome].std()),
    }


def baseline_balance(df, covars, *, base_year=2003, id_col="countyreal", time_col="year", g_col="g") -> pd.DataFrame:
    """Covariate means by cohort at the common baseline. Different lpop means imply the
    cohorts differ in level, which the parallel-trends assumption does not rule out."""
    d = df[df[time_col] == base_year].drop_duplicates(id_col)
    d = d.assign(cohort=np.where(np.isfinite(d[g_col]), d[g_col].astype(int), 0))
    return d.groupby("cohort")[covars].agg(["mean", "std", "count"])


def forbidden_comparisons(
    df: pd.DataFrame,
    *,
    outcome="lemp",
    id_col="countyreal",
    time_col="year",
    g_col="g",
) -> pd.DataFrame:
    """Every 2x2 DiD a static two-way FE estimate secretly averages over.

    Goodman-Bacon's point is that the TWFE coefficient is a weighted average of pairwise 2x2
    DiDs, including *treated-versus-already-treated* pairs. For those pairs the control unit is
    already under treatment during the 'pre' window, so its own change contains part of the
    effect and the 2x2 is biased. Here each pair is computed and flagged explicitly.
    """
    gs = sorted(v for v in np.unique(df[g_col]) if np.isfinite(v))
    years = sorted(df[time_col].unique())
    wide = df.pivot_table(index=id_col, columns=time_col, values=outcome, aggfunc="mean")
    gvec = df.drop_duplicates(id_col).set_index(id_col)[g_col]

    rows = []
    for g1 in gs:
        for g0 in gs + [np.inf]:
            if g0 == g1:
                continue
            pre1 = max(years[0], int(g1 - 1))
            post = int(g1)
            if post not in wide.columns or pre1 not in wide.columns or pre1 >= post:
                continue
            treated_ids = list(gvec.index[gvec == g1])
            if np.isfinite(g0):
                control_ids = list(gvec.index[gvec == g0])
                already_treated = bool(g0 <= pre1)
            else:
                control_ids = list(gvec.index[~np.isfinite(gvec)])
                already_treated = False

            tv = wide.loc[treated_ids, [pre1, post]].dropna()
            cv = wide.loc[control_ids, [pre1, post]].dropna()
            if len(tv) == 0 or len(cv) == 0:
                continue
            rows.append(
                {
                    "treated_cohort": int(g1),
                    "control_cohort": int(g0) if np.isfinite(g0) else "never",
                    "pre_period": pre1,
                    "post_period": post,
                    "n_treated": int(len(tv)),
                    "n_control": int(len(cv)),
                    "did_2x2": float((tv[post] - tv[pre1]).mean() - (cv[post] - cv[pre1]).mean()),
                    "control_already_treated_in_pre": already_treated,
                }
            )
    return pd.DataFrame(rows)


def pretrend_cells(df: pd.DataFrame, *, covars=None, estimator="dr", **kw) -> pd.DataFrame:
    """ATT(g, t) estimated for periods *before* treatment.

    These are placebo estimates: they should be indistinguishable from zero. Any nonzero
    pre-treatment cell is direct evidence against the common-trends assumption that licenses
    the whole design -- which is why this repo reports them before it reports any post period.
    """
    covars = list(covars or [])
    gs = sorted(v for v in np.unique(df[kw.get("g_col", "g")]) if np.isfinite(v))
    years = sorted(df[kw.get("time_col", "year")].unique())
    rng = np.random.default_rng(23)
    rows = []
    for g in gs:
        for t in years:
            if t >= g or t == int(g - 1):
                continue
            try:
                cell = cs.group_time_att(
                    df, g=g, t=int(t), covars=covars, estimator=estimator, n_boot=kw.get("n_boot", 200),
                    rng=rng, **{k: v for k, v in kw.items() if k not in ("n_boot", "g", "t")},
                )
            except ValueError:
                continue
            cell["is_pre_treatment"] = True
            rows.append(cell)
    return pd.DataFrame(rows)
