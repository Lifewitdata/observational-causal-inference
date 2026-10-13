# Data provenance

Two observational panels. Neither contains a randomized experiment — that is the point of this
repository. Every estimate here has to be defended by design, not by randomization.

## 1. `mpdta.csv` — county teen employment and minimum wage (staggered adoption)

| | |
|---|---|
| Source | `data/mpdta.csv` in [`d2cml-ai/csdid`](https://github.com/d2cml-ai/csdid), a port of the dataset shipped with the R package `did` by Callaway & Sant'Anna |
| Original | Callaway and Sant'Anna (2021), *Difference-in-Differences with Multiple Time Periods*, Journal of Econometrics |
| Retrieval | `https://raw.githubusercontent.com/d2cml-ai/csdid/main/data/mpdta.csv`, 2026-10-13 |
| Size | 127,380 bytes, sha256 `c726460cbd24d4b78501...` |
| Structure | 2,500 rows = 500 US counties x 5 years (2003-2007), balanced |

Columns: `year`, `countyreal`, `lpop` (log county population), `lemp` (log teen employment —
outcome), `first.treat` (first year of minimum-wage increase; `0` = never treated), `treat`.

Treatment cohorts: 309 never-treated, 20 counties first treated 2004, 40 in 2006, 131 in 2007.
Staggered timing with three cohorts of very different size is exactly the setting where naive
two-way fixed effects misbehaves, so the naive estimator is reported here on purpose.

## 2. `smoking.dta` / `smoking.csv` — California Proposition 99 (single treated unit)

| | |
|---|---|
| Source | `smoking.dta` in [`scunning1975/mixtape`](https://github.com/scunning1975/mixtape), stored in Git LFS |
| Retrieval | `https://media.githubusercontent.com/media/scunning1975/mixtape/main/smoking.dta`, 2026-10-13 |
| Size | 47,045 bytes, sha256 `3f5de21351fef721b745...` |
| Original | Abadie, Diamond and Hannon (2010), *Synthetic Control Methods for Comparative Case Studies*, JASA |
| Treatment | California's Proposition 99, a 25-cent cigarette excise tax effective 1989 |

Columns: `state` (39 states), `year` (1970-2000), `cigsale` (per-capita cigarette sales, packs —
outcome), `lnincome`, `beer` (per-capita beer sales), `age15to24`, `retprice` (retail price).
`smoking.csv` is a UTF-8 dump of the same frame so the analysis has no Stata dependency.

Balanced: 39 states x 31 years = 1,209 rows. California is the only treated unit, which makes
this the canonical synthetic-control case and makes conventional panel estimators useless here —
there is no within-state control group to difference against.

## Why these two together

They are opposite failure modes of naive analysis:

- `mpdta` has *many* treated units and *staggered* timing, where the danger is contamination of
  treated-versus-treated comparisons and negative weights.
- `smoking` has *one* treated unit and *no* comparable control, where the danger is that any
  single donor state you pick by eye is a bad counterfactual.

Both datasets are small enough to keep in git, and both have published reference estimates, so
`tests/` can assert that these implementations reproduce known numbers rather than merely run.
