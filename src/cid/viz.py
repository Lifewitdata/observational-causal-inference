"""Figures. Every plot here is a diagnostic, not decoration.

The event-study plot exists to be *failed*: if the pre-treatment coefficients are not hugging
zero, the design is not credible and the post-treatment coefficients should not be read as
causal. Likewise the placebo overlay is there to show how often the method finds a fake effect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_FIGS = Path(__file__).resolve().parents[2] / "report" / "figures"

STYLE = {
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
}


def event_study(tab: pd.DataFrame, title: str, path: str | Path | None = None, ref=-1):
    """Dynamic treatment effects with 95% intervals and the reference period pinned at zero."""
    plt.rcParams.update({k: v for k, v in STYLE.items() if not k.startswith("axes.spines")})
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = tab.rel_time.to_numpy(int)
    coef = tab.coef.to_numpy(float) if "coef" in tab else tab.att.to_numpy(float)
    se = tab.std_error.to_numpy(float) if "std_error" in tab else tab.se.to_numpy(float)
    order = np.argsort(x)
    x, coef, se = x[order], coef[order], se[order]

    ax.errorbar(x, coef, yerr=1.96 * se, fmt="o", color="#1f4e79", capsize=3, ms=5, lw=1.4)
    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(ref - 0.5, color="#c0392b", ls="--", lw=1.1, label=f"treatment (ref = {ref})")
    ax.set_xlabel("periods since treatment")
    ax.set_ylabel("effect on log outcome")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)

    pre = coef[x < 0]
    if len(pre) == 0:
        verdict, colour = "no pre-treatment periods in this path", "dimgray"
    else:
        flat = bool(np.all(np.abs(pre) <= 1.96 * se[x < 0]))
        verdict = "pre-trends indistinguishable from 0" if flat else "PRE-TREND VIOLATION"
        colour = "#187f4a" if flat else "#c0392b"
    ax.text(
        0.02, 0.02, verdict, transform=ax.transAxes, fontsize=8,
        color=colour, va="bottom",
    )
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def sc_trajectories(res: dict, treated: str, title: str, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    res["actual"].plot(ax=ax, color="#1f4e79", lw=2, label=f"{treated} (observed)")
    res["counterfactual"].plot(ax=ax, color="#c0392b", lw=2, ls="--", label="synthetic {0}".format(treated))
    ax.axvline(res["treatment_year"], color="grey", ls=":", lw=1.2)
    ax.set_ylabel("per-capita cigarette sales (packs)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def sc_gap(res: dict, treated: str, title: str, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.fill_between(res["gap"].index, res["gap"].to_numpy(float), 0,
                    where=res["gap"].to_numpy(float) >= 0, color="#187f4a", alpha=0.25)
    ax.fill_between(res["gap"].index, res["gap"].to_numpy(float), 0,
                    where=res["gap"].to_numpy(float) < 0, color="#c0392b", alpha=0.25)
    res["gap"].plot(ax=ax, color="#1f4e79", lw=1.6)
    ax.axvline(res["treatment_year"], color="grey", ls=":", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("actual - synthetic")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def placebo_overlay(placebo_gaps: pd.DataFrame, treated_gap: pd.Series, rmspe_cutoff: float,
                    treated_label: str, title: str, path: str | Path | None = None):
    """Donor-state placebos, keeping only those that fit at least as well as the real case."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for col in placebo_gaps.columns:
        pre = treated_gap.index[treated_gap.index < 1989]
        good = placebo_gaps[col].loc[pre]
        if np.sqrt(np.nanmean(good**2)) <= rmspe_cutoff:
            ax.plot(placebo_gaps.index, placebo_gaps[col], color="grey", alpha=0.35, lw=0.8)
    ax.plot(treated_gap.index, treated_gap.to_numpy(float), color="#c0392b", lw=2.0,
            label=f"{treated_label} (real)")
    ax.axvline(1989, color="black", ls=":", lw=1.1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("gap (actual - synthetic)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def cohort_gantt(df: pd.DataFrame, title: str, path: str | Path | None = None, g_col="g",
                 id_col="countyreal", time_col="year"):
    """Who got treated when. Makes staggered adoption concrete instead of abstract."""
    u = df.drop_duplicates(id_col).set_index(id_col)[g_col]
    treated = u[np.isfinite(u)].sort_values()
    if treated.empty:
        raise ValueError("no treated units")
    fig, ax = plt.subplots(figsize=(7.2, 1.0 + 0.02 * len(treated)))
    for i, (unit, g) in enumerate(treated.items()):
        ax.hlines(i, g, df[time_col].max(), color="#1f4e79", lw=2.4)
        ax.plot(g, i, "o", color="#c0392b", ms=4)
    ax.set_yticks([])
    ax.set_xlabel("year")
    ax.set_title(f"{title}  ({len(treated)} treated units, {int((~np.isfinite(u)).sum())} never-treated)")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def pairwise_dids(tab: pd.DataFrame, title: str, path: str | Path | None = None):
    """2x2 DiDs a static TWFE averages over, with contaminated pairs in red."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bad = tab[tab.control_already_treated_in_pre]
    ok = tab[~tab.control_already_treated_in_pre]
    jitter = np.arange(len(ok))
    ax.bar(jitter, ok.did_2x2, color="#1f4e79", alpha=0.8, label="clean comparison")
    ax.bar(np.arange(len(ok), len(ok) + len(bad)), bad.did_2x2, color="#c0392b", alpha=0.85,
           label="control already treated (biased)")
    ax.axhline(0, color="black", lw=0.9)
    labels = [f"{r.treated_cohort} vs {r.control_cohort}" for r in ok.itertuples()] + [
        f"{r.treated_cohort} vs {r.control_cohort}" for r in bad.itertuples()
    ]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("2x2 DiD estimate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig
