"""Least squares with cluster-robust standard errors and rank-deficiency handling.

Written directly in NumPy so the estimators in this repo are inspectable rather than a
call into a modelling package. The rank check matters: fixed-effect designs with staggered
adoption are routinely collinear, and a solver that silently returns *a* solution rather than
*failing* is how an event study ends up with a plausible-looking but meaningless table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import qr


def _rank_preserving_indices(X: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Column indices of a maximal linearly independent subset, in original order."""
    if X.shape[1] == 0:
        return np.arange(0)
    _, R, piv = qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return np.arange(X.shape[1])
    cutoff = tol * diag.max()
    r = int(np.searchsorted(-diag, -cutoff, side="right")) if diag[0] >= cutoff else 0
    r = max(1, min(r, len(piv)))
    return np.sort(piv[:r])


@dataclass
class Fit:
    beta: np.ndarray
    se: np.ndarray
    names: list[str]
    kept: np.ndarray
    resid: np.ndarray
    nobs: int
    vcov: np.ndarray
    dropped: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.tstat = np.divide(self.beta, self.se, out=np.zeros_like(self.beta), where=self.se > 0)
        self.ci_low = self.beta - 1.96 * self.se
        self.ci_high = self.beta + 1.96 * self.se

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "term": self.names,
                "coef": self.beta,
                "std_error": self.se,
                "t": self.tstat,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "dropped_for_collinearity": [n in self.dropped for n in self.names],
            }
        )


def ols(
    X,
    y,
    *,
    cluster=None,
    add_const: bool = False,
    names=None,
) -> Fit:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    n_features = X.shape[1]
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])

    all_names = list(names) if names is not None else [f"x{i}" for i in range(n_features)]
    if add_const:
        all_names = ["const"] + all_names
    if len(all_names) != X.shape[1]:
        raise ValueError(f"names length {len(all_names)} != design width {X.shape[1]}")

    kept = _rank_preserving_indices(X)
    dropped = [all_names[i] for i in range(X.shape[1]) if i not in set(kept.tolist())]
    Xk = X[:, kept]

    beta_k, *_ = np.linalg.lstsq(Xk, y, rcond=None)
    beta = np.zeros(X.shape[1])
    beta[kept] = beta_k

    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(Xk.T @ Xk)
    n, k = Xk.shape

    if cluster is None:
        s2 = float(resid @ resid) / (n - k)
        Vk = s2 * XtX_inv
    else:
        cluster = np.asarray(cluster)
        groups, idx = np.unique(cluster, return_inverse=True)
        u = Xk * resid[:, None]
        scores = np.zeros((len(groups), k))
        np.add.at(scores, idx, u)
        meat = scores.T @ scores
        G = len(groups)
        small_sample = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
        Vk = small_sample * (XtX_inv @ meat @ XtX_inv)

    V = np.zeros((X.shape[1], X.shape[1]))
    V[np.ix_(kept, kept)] = Vk

    return Fit(
        beta=beta,
        se=np.sqrt(np.clip(np.diag(V), 0, None)),
        names=all_names,
        kept=kept,
        resid=resid,
        nobs=n,
        vcov=V,
        dropped=dropped,
    )
