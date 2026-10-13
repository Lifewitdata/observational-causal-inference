"""Inference for a single-unit comparison: placebo tests.

A synthetic control has no sampling distribution -- there is one treated unit, so there is no
standard error in the usual sense. The defensible alternative is *speculative* inference: run the
same procedure on every donor state as though it had been treated, and ask how unusual the real
treated unit's result looks inside that reference distribution. If the method "finds" a large
effect for a state that was never treated, the apparent effect for the real one is worth less.

Two flavours:
  in_space  -- each donor state pretends to be California
  in_time   -- California pretends Prop 99 happened earlier, when nothing did
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cid.sc import weights


def _run_fit(panel, treated, *, method, outcome, unit_col, time_col, treatment_year, predictors, extra, feature_start):
    if method == "trajectory":
        return weights.trajectory_fit(
            panel, treated=treated, outcome=outcome, unit_col=unit_col,
            time_col=time_col, treatment_year=treatment_year,
        )
    return weights.fit(
        panel, treated=treated, outcome=outcome, unit_col=unit_col, time_col=time_col,
        treatment_year=treatment_year, predictors=predictors, extra=extra, feature_start=feature_start,
    )


def in_space(
    panel: pd.DataFrame,
    *,
    treated: str = "California",
    outcome: str = "cigsale",
    unit_col: str = "state",
    time_col: str = "year",
    treatment_year: int = weights.TREATMENT_YEAR,
    predictors: list[str] | None = None,
    extra: list[str] | None = None,
    method: str = "features",
    feature_start: int = 1970,
) -> dict:
    """Fit the real case, then refit it once per donor state as a pseudo-treatment.

    California is removed from the donor pool for every placebo run, so a placebo never gets to
    use the genuinely treated unit as part of someone else's counterfactual.
    """
    kwargs = dict(
        method=method, outcome=outcome, unit_col=unit_col, time_col=time_col,
        treatment_year=treatment_year, predictors=list(predictors or []),
        extra=list(extra or []), feature_start=feature_start,
    )

    main = _run_fit(panel, treated, **kwargs)

    pool = [u for u in panel[unit_col].unique() if u != treated]
    sub = panel[panel[unit_col].isin(pool)]

    rows, gap_curves, failed = [], {}, []
    for d in pool:
        try:
            r = _run_fit(sub, d, **kwargs)
        except (RuntimeError, KeyError, ValueError) as exc:
            # recorded rather than dropped: a p-value whose denominator quietly shrank has to be
            # visible, or the permutation test looks more honest than it is
            failed.append({"unit": d, "error": str(exc)})
            continue
        rows.append(
            {
                "unit": d,
                "cumulative_effect": r["cumulative_effect"],
                "mean_annual_effect": r["mean_annual_effect"],
                "pre_rmspe": r["pre_rmspe"],
                "post_rmspe": r["post_rmspe"],
                "rmspe_ratio": r["rmspe_ratio"],
            }
        )
        gap_curves[d] = r["gap"]

    placebo = pd.DataFrame(rows).sort_values("rmspe_ratio", ascending=False)
    ratio = main["post_rmspe"] / main["pre_rmspe"] if main["pre_rmspe"] else np.inf

    all_ge = float((placebo.rmspe_ratio.abs() >= abs(ratio)).mean()) if len(placebo) else np.nan
    restricted = placebo[placebo.pre_rmspe <= main["pre_rmspe"]]
    restr_ge = float((restricted.rmspe_ratio.abs() >= abs(ratio)).mean()) if len(restricted) else np.nan

    return {
        "main": main,
        "placebo": placebo,
        "gap_curves": pd.DataFrame(gap_curves),
        "rmspe_ratio": float(ratio),
        "p_value_all_donors": all_ge,
        "p_value_pre_fit_filtered": restr_ge,
        "n_placebo": len(placebo),
        "failed_placebos": failed,
    }


def in_time(
    panel: pd.DataFrame,
    *,
    treated: str = "California",
    outcome: str = "cigsale",
    unit_col: str = "state",
    time_col: str = "year",
    real_treatment_year: int = weights.TREATMENT_YEAR,
    pseudo_years: list[int] | None = None,
    predictors: list[str] | None = None,
    extra: list[str] | None = None,
    method: str = "features",
    feature_start: int = 1970,
    min_pre: int = 8,
) -> pd.DataFrame:
    """Move the treatment date backwards, and *truncate the sample at the real treatment year*.

    Without truncation a 1984 pseudo-treatment would have a post-window running through 2000,
    which contains the genuine effect of Prop 99 -- the placebo would "detect" the real shock and
    look like a refutation of everything. Restricting to pre-1989 data is what makes this a test.
    Estimates here should be flat and near zero.
    """
    predictors = list(predictors or [])
    extra = list(extra or [])
    pre_only = panel[panel[time_col] < real_treatment_year]
    years = sorted(pre_only[time_col].unique())
    start = int(min(years))
    pseudo_years = list(pseudo_years) if pseudo_years else list(range(start + min_pre, real_treatment_year, 3))

    rows = []
    for py in pseudo_years:
        if py - start < min_pre:
            continue
        try:
            r = _run_fit(
                pre_only, treated, outcome=outcome, unit_col=unit_col, time_col=time_col,
                treatment_year=py, predictors=predictors, extra=extra, method=method,
                feature_start=feature_start,
            )
        except (RuntimeError, KeyError, ValueError):
            continue
        rows.append(
            {
                "pseudo_treatment_year": py,
                "mean_annual_effect": r["mean_annual_effect"],
                "pre_rmspe": r["pre_rmspe"],
                "rmspe_ratio": r["rmspe_ratio"],
                "n_post_periods": int(np.count_nonzero(np.asarray(years) >= py)),
            }
        )
    return pd.DataFrame(rows)


def effective_donors(w: pd.Series) -> float:
    """Herfindahl inverse: how many states actually carry the synthetic counterfactual.

    Three or four dominant donors means the 'synthetic' California is close to one or two
    states chosen by optimisation, which is a much weaker claim than the method's reputation.
    """
    p = np.asarray(w, float)
    p = p[p > 0]
    p = p / p.sum()
    return float(1.0 / np.sum(p**2))
