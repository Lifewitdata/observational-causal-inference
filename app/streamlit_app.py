"""Interactive explorer for the two designs in this repository.

It is deliberately thin: every number on screen comes from the same `cid` functions the notebooks
call, so the app cannot show something the analysis does not already support. Nothing here is
imported by the notebooks or the tests.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from cid import io, viz
from cid.did import cs, diagnostics, sunabraham, twfe
from cid.sc import placebo, weights

PRED = ["lnincome", "beer", "age15to24", "lnprice"]
SC_SPECS = {
    "trajectory 1970-88 (selected on pre-fit)": "trajectory",
    "features 1970-88": 1970,
    "features 1980-88": 1980,
    "features 1985-88": 1985,
}


@st.cache_data(show_spinner=False)
def load_mpdta():
    return io.load_mpdta()


@st.cache_data(show_spinner=False)
def load_smoking():
    return io.load_smoking()


@st.cache_data(show_spinner=False)
def cs_surface(estimator: str, control: str, use_lpop: bool, n_boot: int):
    return cs.att_surface(
        load_mpdta(), covars=["lpop"] if use_lpop else [], estimator=estimator,
        control=control, n_boot=n_boot,
    )


@st.cache_data(show_spinner=False)
def sc_result(spec: str):
    df = load_smoking()
    if spec == "trajectory":
        return weights.trajectory_fit(df)
    return weights.fit(df, predictors=PRED, feature_start=int(spec))


@st.cache_data(show_spinner=False)
def in_space_result(spec: str):
    df = load_smoking()
    if spec == "trajectory":
        return placebo.in_space(df, method="trajectory")
    return placebo.in_space(df, method="features", predictors=PRED, feature_start=int(spec))


@st.cache_data(show_spinner=False)
def in_time_result(spec: str):
    df = load_smoking()
    if spec == "trajectory":
        return placebo.in_time(df, method="trajectory")
    return placebo.in_time(df, method="features", predictors=PRED, feature_start=int(spec))


def metric_row(items: list[tuple[str, str, str]]):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


# --------------------------------------------------------------------- minimum wage


def did_panel():
    st.caption(
        "500 US counties, 2003-2007. Outcome: log teen employment. Three treatment cohorts "
        "(2004: 20 counties, 2006: 40, 2007: 131) and 309 never-treated."
    )
    method = st.radio(
        "Estimator",
        ["Naive static two-way FE", "Event study (dynamic)", "Callaway-Sant'Anna ATT(g,t)",
         "Sun-Abraham interaction-weighted"],
        horizontal=True,
    )

    if method == "Naive static two-way FE":
        res = twfe.static_twfe(load_mpdta())
        metric_row([
            ("ATT", f"{res['estimate']:+.4f}", "one post-treatment flag, county and year fixed effects"),
            ("Cluster SE", f"{res['std_error']:.4f}", "clustered by county"),
            ("95% CI", f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]", ""),
        ])
        st.warning(
            "This is the number a default regression gives, and it is attenuated: with staggered "
            "timing the same regression averages in treated-versus-already-treated comparisons. "
            "The pairwise decomposition is below.",
            icon="\u26a0\ufe0f",
        )
        pairs = diagnostics.forbidden_comparisons(load_mpdta())
        shown = pairs.assign(control_cohort=pairs.control_cohort.astype(str)).round(4)
        st.dataframe(shown, hide_index=True, width='stretch')
        st.pyplot(viz.pairwise_dids(pairs, "2x2 DiDs pooled by static TWFE"))

    elif method == "Event study (dynamic)":
        lo = st.slider("Earliest period binned", -5, -2, -3)
        tab, meta = twfe.event_study(load_mpdta(), lo=lo, hi=3)
        pre, post = meta["pre_trend_joint"], meta["post_joint"]
        st.markdown(
            f"**Parallel-trends test** (joint Wald, pre-treatment coefficients): "
            f"\u03c7\u00b2 = {pre['stat']:.2f} on {pre['df']} df, **p = {pre['p_value']:.3f}** \u2014 "
            f"{'no detectable pre-trend' if pre['p_value'] > 0.05 else 'PRE-TREND VIOLATION'}.\n\n"
            f"**Joint test, post-treatment coefficients**: p = {post['p_value']:.4f}."
        )
        st.dataframe(tab.round(4), hide_index=True, width='stretch')
        st.pyplot(viz.event_study(tab, "Dynamic treatment effects on log teen employment"))

    elif method == "Callaway-Sant'Anna ATT(g,t)":
        estimator = st.selectbox("Reweighting", ["dr", "simple", "reg", "ipw"])
        control = st.selectbox("Comparison group", ["not-yet-treated", "never-treated"])
        use_lpop = st.toggle("Adjust for log county population", True)
        n_boot = st.slider("Bootstrap draws", 50, 400, 100, step=50)
        if not use_lpop and estimator != "simple":
            st.info(
                "With no covariates there is nothing to reweight on, so reg / IPW / DR all reduce "
                "to the unweighted difference in mean changes.",
                icon="\u2139\ufe0f",
            )
        surface = cs_surface(estimator, control, use_lpop, n_boot)
        overall = cs.aggregate(surface, "simple").iloc[0]
        event = cs.aggregate(surface, "event")
        metric_row([
            ("Overall ATT", f"{overall['att']:+.4f}", "treated-count-weighted mean of all cells"),
            ("Bootstrap SE", f"{overall['se']:.4f}", "unit-level bootstrap of each cell"),
            ("Cells", f"{overall['n_cells']}", "estimable (g, t) pairs"),
        ])
        st.caption(
            f"Read as log points: about a {100 * abs(overall['att']):.1f}% change in teen "
            "employment, averaged over all post-treatment periods."
        )
        st.dataframe(event[["label", "att", "se", "n_treated"]].round(4),
                     hide_index=True, width='stretch')
        st.pyplot(viz.event_study(
            event.rename(columns={"att": "coef", "se": "std_error"}),
            f"Callaway-Sant'Anna: {estimator}, {control} controls",
        ))
        st.dataframe(surface.round(4), hide_index=True, width='stretch')

    else:
        table = sunabraham.cohort_effects(load_mpdta(), lo=-3, hi=3, n_boot=0)
        path = sunabraham.iw_event_time(table)
        overall = sunabraham.overall_iw(table)
        metric_row([
            ("Overall (post)", f"{overall['estimate']:+.4f}", "cohort-share weighted"),
            ("Cells", f"{overall['n_cells']}", ""),
            ("Cohorts", ", ".join(str(g) for g in overall["cohorts"]), ""),
        ])
        st.dataframe(path.drop(columns=["cohorts"]).round(4), hide_index=True, width='stretch')
        st.pyplot(viz.event_study(
            path.rename(columns={"att": "coef", "se": "std_error"}),
            "Sun-Abraham interaction-weighted event path",
        ))
        st.caption(
            "Cohort-specific cells are differences against the never-treated group only, so no "
            "already-treated unit ever serves as a control."
        )


# ----------------------------------------------------------------- synthetic control


def sc_panel():
    st.caption(
        "39 states, 1970-2000. Outcome: per-capita cigarette sales in packs. California enacted "
        "Proposition 99 (a 25-cent excise) in 1989; every other state is a potential donor."
    )
    label = st.selectbox("Matching specification", list(SC_SPECS), index=0)
    st.caption(
        "Weights solve a constrained programme: non-negative, summing to one. The specification is "
        "chosen on pre-period fit, which is the only criterion the data offers."
    )
    res = sc_result(SC_SPECS[label])

    metric_row([
        ("Gap in 2000", f"{res['final_year_effect']:+.1f}", "packs per capita, actual minus synthetic"),
        ("Mean annual", f"{res['mean_annual_effect']:+.2f}", "packs per capita, 1989-2000"),
        ("Pre-period RMSPE", f"{res['pre_rmspe']:.2f}", "packs, 1970-88 \u2014 lower is a better fit"),
        ("Effective donors", f"{placebo.effective_donors(res['weights']):.1f}", "1 / sum(w\u00b2), of 38 available"),
    ])

    st.pyplot(viz.sc_trajectories(res, "California", f"California vs synthetic California \u2014 {label}"))
    st.pyplot(viz.sc_gap(res, "California", "Estimated effect (actual \u2212 synthetic)"))

    w = res["weights"].sort_values(ascending=False)
    st.dataframe(
        pd.DataFrame({"weight": w}).query("weight > 1e-4").round(4),
        width="stretch",
    )
    st.caption(
        f"{res['n_active_donors']} states carry non-zero weight; the top three hold "
        f"{100 * w.nlargest(3).sum():.0f}% of it. The counterfactual is a blend of a handful of "
        "states, not an average of 38."
    )

    if st.checkbox("Run the placebo tests (39 extra constrained fits, ~10 s)", False):
        pl = in_space_result(SC_SPECS[label])
        floor = 1 / pl["n_placebo"]
        st.markdown(
            f"**In-space placebos.** Refitting the same procedure to each donor state as if it had "
            f"been treated: California's post/pre RMSPE ratio is beaten by "
            f"{round(pl['p_value_all_donors'] * pl['n_placebo'])} of {pl['n_placebo']} states "
            f"(p = {pl['p_value_all_donors']:.3f}; smallest attainable p is {floor:.3f}). "
            f"Restricted to the {int((pl['placebo'].pre_rmspe <= pl['main']['pre_rmspe']).sum())} "
            f"donors that fit at least as well as California, **p = "
            f"{pl['p_value_pre_fit_filtered']:.3f}** \u2014 not significant."
        )
        if pl["failed_placebos"]:
            st.warning(
                "The p-value denominator excludes "
                + ", ".join(f["unit"] for f in pl["failed_placebos"])
                + ": the constrained refit did not converge."
            )
        st.pyplot(viz.placebo_overlay(
            pl["gap_curves"], res["gap"], res["pre_rmspe"], "California",
            "Placebo gaps: donor states (grey) vs California (red)",
        ))
        it = in_time_result(SC_SPECS[label])
        st.markdown(
            f"**In-time placebos.** Pretending Prop 99 happened earlier, with the sample truncated "
            f"at 1988 so the real effect cannot leak in: mean pseudo-effect "
            f"**{it['mean_annual_effect'].mean():+.2f} packs/yr** against the headline "
            f"{res['mean_annual_effect']:+.2f}. Some of the apparent effect is pre-existing drift."
        )
        st.dataframe(it.round(2), hide_index=True, width='stretch')


# ------------------------------------------------------------------------------- shell


st.set_page_config(page_title="Observational causal inference", layout="wide")
st.title("Two causal designs without a randomised trial")
st.caption(
    "Every figure and number here is produced by the same functions the notebooks call. "
    "Estimators are hand-rolled in NumPy/SciPy \u2014 see src/cid/."
)

tab_did, tab_sc = st.tabs(["A \u00b7 Minimum wage (staggered DiD)", "B \u00b7 Prop 99 (synthetic control)"])
with tab_did:
    did_panel()
with tab_sc:
    sc_panel()
