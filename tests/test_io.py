"""The panels are validated on load, so the loaders themselves need validating.

Every estimator downstream assumes a balanced unit-time grid and a particular coding of the
treatment cohort, so these are the facts worth pinning.
"""

import numpy as np
import pandas as pd
import pytest

from cid import io


@pytest.fixture(scope="module")
def mpdta():
    return io.load_mpdta()


@pytest.fixture(scope="module")
def smoking():
    return io.load_smoking()


def test_mpdta_is_the_expected_balanced_panel(mpdta):
    assert len(mpdta) == 2500
    assert mpdta["countyreal"].nunique() == 500
    assert sorted(mpdta["year"].unique()) == [2003, 2004, 2005, 2006, 2007]
    assert mpdta.groupby("countyreal")["year"].nunique().eq(5).all()
    assert not mpdta.duplicated(["countyreal", "year"]).any()


def test_mpdta_cohort_sizes_and_never_treated_count(mpdta):
    first = mpdta.drop_duplicates("countyreal")
    counts = first[np.isfinite(first["g"])]["g"].astype(int).value_counts().sort_index()
    assert counts.to_dict() == {2004: 20, 2006: 40, 2007: 131}
    assert int((~np.isfinite(first["g"])).sum()) == 309
    assert int(first["treated"].sum()) == 191


def test_mpdta_g_is_inf_for_never_treated(mpdta):
    merged = mpdta.drop_duplicates("countyreal").set_index("countyreal")
    assert np.isinf(merged.loc[merged["first_treat"] == 0, "g"]).all()
    assert (merged.loc[merged["first_treat"] != 0, "g"] == merged.loc[merged["first_treat"] != 0, "first_treat"]).all()


def test_smoking_is_the_expected_balanced_panel(smoking):
    assert len(smoking) == 39 * 31
    assert smoking["state"].nunique() == 39
    assert smoking["year"].min() == 1970 and smoking["year"].max() == 2000
    assert smoking.groupby("state")["year"].nunique().eq(31).all()


def test_smoking_treats_only_california_at_1989(smoking):
    first = smoking.drop_duplicates("state").set_index("state")
    assert sorted(first.index[first["treated"] == 1]) == ["California"]
    assert first.loc["California", "g"] == 1989
    assert np.isinf(first.loc[first["treated"] == 0, "g"]).all()


def test_smoking_lnprice_is_the_log_of_retail_price(smoking):
    assert np.allclose(smoking["lnprice"], np.log(smoking["retprice"]))


def test_assert_balanced_rejects_a_holed_panel(mpdta):
    """Dropping one county-period leaves the rest of the grid intact, which is exactly the
    imbalance the loaders refuse: a synthetic control or DiD design silently reindexing around
    it is worse than failing loudly."""
    one = mpdta[(mpdta["countyreal"] == mpdta["countyreal"].iloc[0]) & (mpdta["year"] == 2005)]
    with pytest.raises(ValueError, match="not balanced"):
        io.assert_balanced(mpdta.drop(one.index), "countyreal", "year")


def test_assert_balanced_rejects_duplicate_rows(mpdta):
    with pytest.raises(ValueError, match="duplicate"):
        io.assert_balanced(pd.concat([mpdta, mpdta.head()]), "countyreal", "year")


def test_relative_time_is_nan_for_never_treated(mpdta):
    rel = io.relative_time(mpdta["year"], mpdta["g"])
    never = mpdta["g"].notna() & np.isinf(mpdta["g"])
    assert rel[never].isna().all()
    treated = ~never
    assert np.allclose(rel[treated], (mpdta["year"] - mpdta["g"])[treated])
