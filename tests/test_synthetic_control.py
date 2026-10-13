"""Synthetic control for California's Proposition 99.

Two things are worth locking down here. The first is the mechanical invariant that makes the
method what it is: the counterfactual is a *convex combination* of real donors, so it can never
lie outside the donor range in any year. The second is the reproduction of Abadie, Diamond &
Hansen's published result -- a 1989-onward gap of roughly 20 packs per capita per year, with a
pre-period fit good enough that the gap is not just an artefact of a bad synthetic state.
"""

import numpy as np
import pandas as pd
import pytest

from cid import io
from cid.sc import placebo, weights


@pytest.fixture(scope="module")
def smoking():
    return io.load_smoking()


@pytest.fixture(scope="module")
def traj(smoking):
    return weights.trajectory_fit(smoking)


@pytest.fixture(scope="module")
def in_space(smoking):
    """39 constrained fits -- the one expensive thing the placebo tests share."""
    return placebo.in_space(smoking, method="trajectory")


# ---------------------------------------------------------------- invariants


def test_weights_are_a_convex_combination(traj):
    w = traj["weights"]
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert len(w) == 38
    assert "California" not in w.index
    assert 1 <= traj["n_active_donors"] <= 38


def test_counterfactual_stays_inside_the_donor_envelope(traj, smoking):
    """A direct consequence of w >= 0, sum w = 1: if the synthetic California is outside the range
    spanned by real states, the optimisation has extrapolated and the estimate is not a control."""
    wide = smoking.pivot_table(index="year", columns="state", values="cigsale")
    lo = wide[traj["donors"]].min(axis=1)
    hi = wide[traj["donors"]].max(axis=1)
    cf = traj["counterfactual"]
    assert (cf >= lo - 1e-8).all()
    assert (cf <= hi + 1e-8).all()


def test_simplex_solver_rejects_missing_features():
    with pytest.raises(ValueError, match="NaN"):
        weights.simplex_weights(np.array([1.0, np.nan]), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_simplex_solver_rejects_a_mis_shaped_target():
    with pytest.raises(ValueError, match="features"):
        weights.simplex_weights(np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_build_features_drops_predictors_that_are_not_complete(smoking):
    damaged = smoking.copy()
    damaged.loc[damaged["state"] == "Utah", "lnincome"] = np.nan
    f = weights.build_features(damaged, extra=["lnincome", "beer"])
    assert "lnincome" in f.attrs["dropped_incomplete"]
    assert "lnincome" not in f.columns
    assert "beer" in f.columns
    assert not f.isna().any().any()


def test_build_features_keeps_the_outcome_summary(smoking):
    f = weights.build_features(smoking, extra=["lnincome"])
    assert {"cigsale_mean", "cigsale_slope"} <= set(f.columns)
    assert f.loc["California", "cigsale_mean"] == pytest.approx(
        smoking[(smoking.state == "California") & (smoking.year <= 1988)]["cigsale"].mean(),
        abs=1e-6,
    )


# ---------------------------------------------------------------- reproduction


def test_trajectory_fit_reproduces_prop_99(traj):
    assert traj["pre_rmspe"] == pytest.approx(1.70, abs=0.3)
    assert traj["mean_annual_effect"] == pytest.approx(-19.75, abs=1.5)
    assert traj["final_year_effect"] == pytest.approx(-26.90, abs=2.0)
    assert traj["cumulative_effect"] == pytest.approx(-237.0, abs=12.0)
    assert traj["gap"].loc[1989:].lt(0).all()
    assert traj["gap"].loc[:1988].abs().mean() < 2.0


def test_the_sign_survives_every_feature_window(smoking):
    """The specification curve that matters: three different predictor windows plus full-trajectory
    matching. They disagree about magnitude and agree about direction."""
    pred = ["lnincome", "beer", "age15to24", "lnprice"]
    specs = {
        "features 1970-88": weights.fit(smoking, predictors=pred, feature_start=1970),
        "features 1980-88": weights.fit(smoking, predictors=pred, feature_start=1980),
        "features 1985-88": weights.fit(smoking, predictors=pred, feature_start=1985),
        "trajectory 1970-88": weights.trajectory_fit(smoking),
    }
    assert all(r["mean_annual_effect"] < -10 for r in specs.values())
    best = min(specs, key=lambda k: specs[k]["pre_rmspe"])
    assert best == "trajectory 1970-88"
    assert specs["trajectory 1970-88"]["pre_rmspe"] < specs["features 1980-88"]["pre_rmspe"]


def test_weight_concentration_is_reported_honestly(traj):
    w = traj["weights"].sort_values(ascending=False)
    assert w.index[0] == "Utah"
    assert placebo.effective_donors(w) == pytest.approx(3.76, abs=0.4)
    assert w.nlargest(3).sum() > 0.8
    assert w.iloc[0] < 0.5  # no single donor is the counterfactual on its own


def test_effective_donors_counts_uniform_weights_exactly():
    assert placebo.effective_donors(pd.Series([0.25, 0.25, 0.25, 0.25])) == pytest.approx(4.0)
    assert placebo.effective_donors(pd.Series([1.0, 0.0, 0.0])) == pytest.approx(1.0)


# ---------------------------------------------------------------- placebo tests


@pytest.mark.slow
def test_in_space_placebos_bound_the_p_value(in_space):
    pl = in_space
    # every donor must either produce a refit or be reported as a failure -- the p-value denominator
    # is only honest if the two add back up to the pool
    assert pl["n_placebo"] + len(pl["failed_placebos"]) == 38
    assert pl["n_placebo"] >= 36
    assert pl["rmspe_ratio"] == pytest.approx(pl["main"]["post_rmspe"] / pl["main"]["pre_rmspe"])
    floor = 1 / pl["n_placebo"]
    assert floor <= pl["p_value_all_donors"] <= 3 * floor
    # restricting to donors that fit at least as well as California can only weaken the result
    assert pl["p_value_pre_fit_filtered"] >= pl["p_value_all_donors"]
    assert pl["p_value_pre_fit_filtered"] > 0.05


@pytest.mark.slow
def test_in_space_placebo_curves_cover_every_donor(in_space, smoking):
    pl = in_space
    assert set(pl["gap_curves"].columns) == set(pl["placebo"]["unit"])
    assert len(pl["gap_curves"]) == smoking["year"].nunique()
    assert pl["placebo"]["pre_rmspe"].min() > 0


@pytest.mark.slow
def test_in_time_placebos_are_a_fraction_of_the_real_effect(smoking):
    """Pseudo-treatments before 1989 manufacture a few packs per year of divergence out of nothing.
    That is genuine evidence of pre-existing drift, and it is roughly a fifth of the headline --
    which is why the memo reports the direction and order of magnitude rather than the decimal."""
    it = placebo.in_time(smoking, method="trajectory")
    assert list(it["pseudo_treatment_year"]) == [1978, 1981, 1984, 1987]
    assert it["mean_annual_effect"].mean() == pytest.approx(-4.3, abs=1.5)
    assert abs(it["mean_annual_effect"].mean()) < 0.5 * abs(
        weights.trajectory_fit(smoking)["mean_annual_effect"]
    )
    assert (it["mean_annual_effect"] < 0).all()
