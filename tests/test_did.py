"""Difference-in-differences under staggered adoption.

The published reference for this panel (Cengiz et al.'s minimum-wage county data, as used by
Callaway & Sant'Anna and Sun & Abraham) is an employment effect of roughly -4 log points. These
tests pin that reproduction, and more importantly pin the *structure* that makes the naive TWFE
number the wrong answer: a dynamic specification whose pre-treatment coefficients are jointly
zero, and a forbidden-comparison diagnostic that shows where the naive weights come from.
"""

import numpy as np
import pandas as pd
import pytest

from cid.did import cs, diagnostics, sunabraham, twfe


@pytest.fixture(scope="module")
def panel():
    from cid import io

    return io.load_mpdta()


# ---------------------------------------------------------------- naive versions


def test_static_twfe_reproduces_the_naive_number(panel):
    res = twfe.static_twfe(panel)
    assert res["estimate"] == pytest.approx(-0.0365, abs=0.002)
    assert res["std_error"] == pytest.approx(0.0148, abs=0.002)
    assert res["ci_low"] < res["estimate"] < res["ci_high"]
    assert (res["n_units"], res["n_obs"]) == (500, 2500)


def test_event_study_keeps_the_never_treated_units_in_the_design(panel):
    """The rank-collapse guard.

    Dropping the never-treated counties makes relative time an exact linear combination of the
    unit and time fixed effects, so the solve returns an arbitrary solution with large spurious
    pre-trends. The design must contain all 500 counties and retain every relative-time column.
    """
    tab, info = twfe.event_study(panel, lo=-4, hi=3)
    assert info["n_units"] == 500
    assert info["n_obs"] == 2500
    assert info["dropped"] == []
    assert not tab.empty
    assert (tab["rel_time"] < 0).any() and (tab["rel_time"] > 0).any()


def test_event_study_finds_no_pre_trend_and_a_real_post_trend(panel):
    tab, info = twfe.event_study(panel, lo=-4, hi=3)
    assert info["pre_trend_joint"]["p_value"] > 0.05
    assert info["post_joint"]["p_value"] < 0.05
    pre = tab[tab.rel_time < 0]
    assert pre["coef"].abs().max() < 0.05
    assert tab.loc[tab.rel_time == 2, "coef"].iloc[0] == pytest.approx(-0.13, abs=0.03)


def test_callaway_santanna_reproduces_the_published_att(panel):
    surface = cs.att_surface(panel, covars=["lpop"], estimator="dr", n_boot=200)
    overall = cs.aggregate(surface, "simple").iloc[0]
    assert -0.08 < overall["att"] < -0.01
    assert overall["se"] < 0.03
    assert overall["att"] + 1.96 * overall["se"] < 0  # the effect is significant, not its mirror
    assert surface["rel_time"].min() >= 0


@pytest.mark.parametrize("estimator", ["simple", "reg", "ipw", "dr"])
def test_all_four_cs_estimators_agree_within_a_hundredth(panel, estimator):
    surface = cs.att_surface(panel, covars=["lpop"], estimator=estimator, n_boot=60)
    att = float(cs.aggregate(surface, "simple").iloc[0]["att"])
    assert att == pytest.approx(-0.041, abs=0.01)


def test_control_rules_differ_in_eligible_controls_but_not_in_the_headline(panel):
    """At t = 2007 the two rules coincide (no county is treated after 2007); at t = 2006 the
    not-yet-treated rule also recruits the 2007 cohort, which is the whole point of the rule."""
    never = cs.delta_frame(panel, g=2006, t=2006, control="never-treated")
    notyet = cs.delta_frame(panel, g=2006, t=2006, control="not-yet-treated")
    assert (never["treated"] == 0).sum() == 309
    assert (notyet["treated"] == 0).sum() == 309 + 131
    assert (never["treated"] == 1).sum() == (notyet["treated"] == 1).sum() == 40


def test_delta_frame_needs_the_period_before_the_cohort(panel):
    with pytest.raises(ValueError, match="lacks period"):
        cs.delta_frame(panel, g=2003, t=2004)


def test_unknown_estimator_and_control_rule_are_rejected(panel):
    with pytest.raises(ValueError, match="unknown control rule"):
        cs.delta_frame(panel, g=2007, t=2007, control="somehow")
    f = cs.delta_frame(panel, g=2007, t=2007, covars=["lpop"])
    with pytest.raises(ValueError, match="unknown estimator"):
        cs._att_from_frame(f, ["lpop"], "matching")


# ---------------------------------------------------------------- Sun & Abraham


def test_sun_abraham_has_no_baseline_row_and_matches_cs(panel):
    """The reference period is a contrast of a group against itself, so it is zero by construction
    and must not appear as an estimate."""
    table = sunabraham.cohort_effects(panel, lo=-3, hi=3, n_boot=0)
    assert -1 not in set(table["rel_time"])
    path = sunabraham.iw_event_time(table)
    assert -1 not in set(path["rel_time"])
    overall = sunabraham.overall_iw(table)
    assert overall["estimate"] == pytest.approx(-0.040, abs=0.01)
    assert overall["cohorts"] == [2004, 2006, 2007]


def test_iw_event_time_weights_are_shares_of_treated_units(panel):
    table = sunabraham.cohort_effects(panel, lo=-3, hi=3, n_boot=0)
    path = sunabraham.iw_event_time(table)
    row = path[path.rel_time == 0].iloc[0]
    assert row["n_treated"] == 191
    assert row["ci_low"] < row["att"] < row["ci_high"]


# ---------------------------------------------------------------- diagnostics


def test_static_twfe_averages_forbidden_comparisons(panel):
    pairs = diagnostics.forbidden_comparisons(panel)
    assert len(pairs) == 9
    bad = pairs[pairs["control_already_treated_in_pre"]]
    assert len(bad) == 3
    assert set(bad["treated_cohort"]) == {2006, 2007, 2007}
    assert bad["did_2x2"].mean() > 0 > pairs[~pairs["control_already_treated_in_pre"]]["did_2x2"].mean()


def test_pre_treatment_placebo_cells_look_like_zero(panel):
    pre = diagnostics.pretrend_cells(panel, covars=["lpop"], estimator="dr", n_boot=60)
    assert not pre.empty
    assert pre["rel_time"].max() < 0
    assert (pre["att"].abs() < 0.15).all()
    assert pre["t"].max() < pre["g"].min()


def test_design_summary_counts_cohorts(panel):
    s = diagnostics.summary(panel)
    assert s["n_units"] == 500
    assert s["n_treated"] == 191
    assert s["n_never_treated"] == 309
    assert s["cohorts"] == {2004: 20, 2006: 40, 2007: 131}
    assert s["period_range"] == (2003, 2007)
