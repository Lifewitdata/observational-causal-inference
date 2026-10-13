"""Regression tests for the hand-rolled least-squares layer.

Two of these guard past bugs rather than hypothetical ones. The fixed-effect design matrices in
`cid.did.twfe` are built from `_dummies`, and an early version emitted one column per *row*
instead of per *level*; the resulting rank deficiency silently distorted the event study. The
second guards the coefficient/name alignment that every reported table depends on.
"""

import numpy as np
import pandas as pd
import pytest

from cid import linreg
from cid.did.twfe import _dummies


def test_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 2.0 + 1.5 * x + rng.normal(scale=0.2, size=400)
    fit = linreg.ols(np.column_stack([x]), y, add_const=True, names=["x"])
    assert fit.beta[0] == pytest.approx(2.0, abs=0.05)
    assert fit.beta[1] == pytest.approx(1.5, abs=0.05)
    assert "const" in fit.names


def test_dummies_have_one_column_per_level_not_per_row():
    s = pd.Series(["a", "b"] * 100)
    d = _dummies(s)
    assert d.shape == (200, 1)
    assert list(d.sum(axis=0)) == [100]
    assert _dummies(pd.Series(["only"] * 10)).shape == (10, 0)


def test_collinear_column_is_dropped_and_named_not_silently_absorbed():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 3.0 * x + rng.normal(scale=0.1, size=200)
    X = np.column_stack([x, x, x + x])
    fit = linreg.ols(X, y, add_const=True, names=["x1", "x2", "x3"])
    assert len(fit.dropped) == 2
    assert len(fit.beta) == len(fit.names) == 4
    kept_x = [n for n in ("x1", "x2", "x3") if n not in fit.dropped]
    assert len(kept_x) == 1
    # whichever duplicate survived carries the whole slope, so the fit itself is still 3x
    assert all(fit.beta[fit.names.index(n)] == 0.0 for n in fit.dropped)
    beta_x = fit.beta[[fit.names.index(n) for n in ("x1", "x2", "x3")]]
    fitted = X @ beta_x
    assert np.polyfit(x, fitted, 1)[0] == pytest.approx(3.0, abs=0.05)
    assert abs(fit.resid).max() < 0.5


def test_default_names_stay_aligned_when_a_constant_is_added():
    """`ols` used to size its automatic names against the design *after* prepending the intercept,
    which raised `names length 3 != design width 2` for any unlabelled single-regressor fit."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=100)
    fit = linreg.ols(np.column_stack([x]), 2.0 + 0.7 * x + rng.normal(scale=0.1, size=100), add_const=True)
    assert fit.names == ["const", "x0"]
    assert fit.beta[1] == pytest.approx(0.7, abs=0.05)


def test_frame_rows_line_up_with_names_after_dropping():
    rng = np.random.default_rng(2)
    x = rng.normal(size=150)
    X = np.column_stack([x, 2 * x])
    fit = linreg.ols(X, x * 1.0 + rng.normal(scale=0.1, size=150), names=["a", "b"])
    frame = fit.frame()
    assert list(frame["term"]) == fit.names
    assert list(frame["coef"]) == list(fit.beta)
    assert list(frame["ci_low"]) == list(fit.beta - 1.96 * fit.se)
    assert frame["dropped_for_collinearity"].tolist() == [n in fit.dropped for n in fit.names]


def test_cluster_robust_se_differs_from_homoskedastic_se():
    """Cluster-robust variance matters when the regressor is nearly constant within a cluster and
    the errors share a cluster-level shock -- the usual state/county-panel situation. Ignoring the
    clustering makes the effective sample look like the number of rows rather than the clusters."""
    rng = np.random.default_rng(3)
    groups, x, y = [], [], []
    for g in range(40):
        x_level = rng.normal()
        n = 15
        shock = rng.normal()
        xs = x_level + 0.1 * rng.normal(size=n)
        groups += [g] * n
        x += list(xs)
        y += list(1.0 * xs + 0.6 * shock + 0.1 * rng.normal(size=n))
    X = np.column_stack([x])
    plain = linreg.ols(X, np.array(y), add_const=True)
    clustered = linreg.ols(X, np.array(y), add_const=True, cluster=np.array(groups))
    slope_i, slope_c = plain.names.index("x0"), clustered.names.index("x0")
    assert plain.beta[slope_i] == pytest.approx(clustered.beta[slope_c], abs=1e-9)
    assert clustered.se[slope_c] > 3 * plain.se[slope_i]


def test_residuals_are_orthogonal_to_retained_columns():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(scale=0.3, size=120)
    fit = linreg.ols(X, y)
    assert abs(fit.resid @ X[:, fit.kept]).max() < 1e-8
    assert fit.nobs == 120
