"""Sun & Abraham (2021) interaction-weighted estimator.

For each cohort g and event time e, the cohort-specific effect is a plain two-period difference
against the *never-treated* group; the event-time estimate is those cohort effects weighted by
each cohort's share of treated units observed at that event time. This gives an interpretable
dynamic path without the negative weights that can afflict static two-way fixed effects when
treatment timing is staggered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cid.did import cs


def cohort_effects(
    df: pd.DataFrame,
    *,
    outcome="lemp",
    id_col="countyreal",
    time_col="year",
    g_col="g",
    lo=None,
    hi=None,
    n_boot: int = 0,
) -> pd.DataFrame:
    """ATT(g, e) for every cohort and relative period, using never-treated units as controls."""
    gs = sorted(v for v in np.unique(df[g_col]) if np.isfinite(v))
    years = sorted(df[time_col].unique())
    rng = np.random.default_rng(11)
    rows = []
    for g in gs:
        for t in years:
            e = int(t - g)
            if e == -1:
                continue  # the baseline period: the contrast is against itself, so it is 0 by construction
            if lo is not None and e < lo:
                continue
            if hi is not None and e > hi:
                continue
            try:
                cell = cs.group_time_att(
                    df, g=g, t=int(t), outcome=outcome, id_col=id_col, time_col=time_col,
                    g_col=g_col, covars=[], estimator="simple", control="never-treated",
                    n_boot=n_boot, rng=rng,
                )
            except ValueError:
                continue
            cell["rel_time"] = e
            rows.append(cell)
    return pd.DataFrame(rows)


def iw_event_time(cohort_table: pd.DataFrame) -> pd.DataFrame:
    """Sun-Abraham event-time path: cohort effects weighted by treated-share."""
    out = []
    for rel, blk in cohort_table.groupby("rel_time"):
        w = blk.n_treated.astype(float)
        if w.sum() == 0:
            continue
        att = float(np.average(blk.att, weights=w))
        se = float(np.sqrt(np.sum((w / w.sum()) ** 2 * blk.se**2))) if blk.se.notna().all() else np.nan
        out.append(
            {
                "rel_time": int(rel),
                "att": att,
                "se": se,
                "ci_low": att - 1.96 * se if np.isfinite(se) else np.nan,
                "ci_high": att + 1.96 * se if np.isfinite(se) else np.nan,
                "n_treated": int(w.sum()),
                "cohorts": list(blk.g.astype(int)),
            }
        )
    return pd.DataFrame(out)


def overall_iw(cohort_table: pd.DataFrame) -> dict:
    """Average post-treatment effect, weighting each (g, e) cell by its cohort size."""
    post = cohort_table[cohort_table.rel_time >= 0]
    w = post.n_treated.astype(float)
    return {
        "estimate": float(np.average(post.att, weights=w)),
        "n_cells": len(post),
        "cohorts": sorted(post.g.astype(int).unique().tolist()),
    }
