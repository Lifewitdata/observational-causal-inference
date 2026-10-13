"""Dataset loading and integrity checks.

Both panels are validated on load rather than trusted, because every downstream estimator in
this repo assumes a balanced unit-time grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"

NEVER_TREATED = 0


def assert_balanced(df: pd.DataFrame, id_col: str, time_col: str) -> None:
    counts = df.groupby(id_col)[time_col].nunique()
    n_t = df[time_col].nunique()
    if not (counts == n_t).all():
        bad = counts[counts != n_t]
        raise ValueError(f"panel is not balanced; {len(bad)} units with != {n_t} periods")
    if df.duplicated([id_col, time_col]).any():
        raise ValueError("duplicate unit-time rows")


def _clean(df, id_col, time_col, outcome, covars, require_covars: bool = False) -> pd.DataFrame:
    """Require a complete outcome grid; treat covariates as allowed to be sparse.

    lnincome, beer and age15to24 are genuinely missing for many state-years in the 1970s and
    1990s. Requiring them to be complete would throw away most of the panel -- and the counterfactual
    only needs the outcome path. Predictor gaps are handled where they belong, in the feature
    builder, which drops any summary statistic that is not available for every donor.
    """
    needed = [id_col, time_col, outcome]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; have {list(df.columns)}")
    df = df.dropna(subset=needed).copy()
    assert_balanced(df, id_col, time_col)
    if require_covars:
        absent = [c for c in covars if c not in df.columns]
        if absent:
            raise ValueError(f"missing covariates {absent}")
        df = df.dropna(subset=covars)
        assert_balanced(df, id_col, time_col)
    return df


def load_mpdta() -> pd.DataFrame:
    """500 US counties x 2003-2007. Outcome is log teen employment.

    `first.treat` is the year a county's minimum wage first rose; 0 means never treated.
    """
    df = pd.read_csv(DATA / "mpdta.csv").rename(columns={"first.treat": "first_treat"})
    df["county"] = df["countyreal"].astype("category")
    df["treated"] = (df["first_treat"] != NEVER_TREATED).astype(int)
    df["g"] = np.where(df["first_treat"] == NEVER_TREATED, np.inf, df["first_treat"])
    return _clean(df, "countyreal", "year", "lemp", ["lpop"], require_covars=True)


def load_smoking() -> pd.DataFrame:
    """39 states x 1970-2000. Outcome is per-capita cigarette sales in packs.

    California enacted Proposition 99 (a 25-cent excise tax) effective 1989.
    """
    df = pd.read_csv(DATA / "smoking.csv")
    df["state"] = df["state"].astype(str)
    df["year"] = df["year"].astype(int)
    df["treated"] = (df["state"] == "California").astype(int)
    df["g"] = np.where(df["treated"] == 1, 1989.0, np.inf)
    df["lnprice"] = np.log(df["retprice"])
    return _clean(df, "state", "year", "cigsale", ["lnincome", "beer", "age15to24", "lnprice"])


def cohorts(df: pd.DataFrame, g_col: str = "g") -> list[float]:
    vals = np.sort(df.drop_duplicates(df.index.name or df.columns[0])[g_col].unique())
    return [v for v in vals if np.isfinite(v)]


def relative_time(year: pd.Series, g: pd.Series) -> pd.Series:
    """Periods since treatment; never-treated units get NaN so callers can exclude them."""
    rel = year.astype(float) - g.astype(float)
    return rel.where(np.isfinite(g))
