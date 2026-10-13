# MEMO — did the policy work, and how much should we trust the number?

**To:** anyone deciding on the basis of an observational estimate
**From:** analysis (full work in `notebooks/`, reproduction in `tests/`)
**Date:** 2026-10-13
**Length:** one page, on purpose

## Bottom line

Two real policy questions, both answered without a randomised trial:

1. **Raising the minimum wage reduced teen employment in affected counties by about 4%.**
   I am confident in the direction and reasonably confident in the size. Every defensible way of
   computing it lands within half a percentage point of the others.
2. **California's 1989 cigarette tax reduced per-capita sales by roughly 20 packs a year, ending
   about 27 packs below where the state would otherwise have been by 2000.**
   I am confident in the direction and the order of magnitude. I am **not** confident in the
   statistical significance, and I would not put a p-value in front of a decision-maker on this one.

## What the first answer rests on

Comparing counties that raised the wage against counties that did not, before and after. The
naive version of that comparison — one regression with county and year controls — gives −3.65%.
It is *wrong in a knowable direction*: with three different adoption years, the naive regression
silently compares later-raising counties against counties that had already raised theirs, so
already-affected places serve as "controls" and the estimate is pulled toward zero. Correcting for
that moves the answer to −4.1%, and four independent correction methods agree to within 0.3%.

The assumption I cannot prove is that, absent the wage change, employment in these counties would
have trended the same way. I tested it in the only way available — by looking at the periods
*before* each county acted — and found no divergence. That is a failure to disprove the assumption,
not a proof of it, and the test is not very powerful with 191 treated counties.

## What the second answer rests on

California has no twin: no single state, and no average of the other 38, looks like California. So
the comparison is *constructed* — a weighted blend of donor states chosen to match California's
pre-1989 consumption path. The blend matches to within about 1.5% error, and after 1989 the real
state falls away from it.

Three things stop me from calling this settled:

- **The blend is really three or four states**, not 38. Utah, Montana and Nevada carry 84% of the
  weight. Delete either of the two heaviest and the answer moves by 5–8%; across all 38 single-state
  deletions the 2000 gap ranges from −27.9 to −24.9 packs.
- **Formal significance is borderline at best.** Testing California against the same method applied
  to states that did nothing gives p = 0.05 using all donors, but 0.17 once restricted to donors
  that fit California at least as well as the real blend does. 0.17 is not significant.
- **There was already drift.** Re-running the method with a pretend tax date in the early 1980s —
  a period with no policy change — manufactures a 4-pack-a-year decline out of nothing. That is
  roughly a fifth of the reported effect, and it is real, not noise.

So: the tax cut smoking, and by a lot. But the honest claim is "direction certain, magnitude about
20 packs a year give or take a few," not "−19.75, p < 0.05."

## What would change my mind

| Finding | What it would do |
|---|---|
| Teen employment in treated counties had already been falling relative to controls before each wage increase | Would invalidate answer 1 entirely — this is the one test that matters, and it currently passes |
| A neighbouring-county analysis showing employment shifting across borders | Would mean answer 1 is partly redistribution, not a net employment loss |
| A donor state whose pre-1989 path matches California better than the current blend | Would move answer 2. Deleting either of the two heaviest current donors already moves it by 5–8%, and the four specifications I trust span 4.9 packs |
| Evidence California's tax reflected a broader anti-smoking culture (media campaigns, advertising rules) | Would mean answer 2 attributes to the tax what a culture shift did |

## What I deliberately did not claim

- I did not claim Prop 99 raised tax revenue or improved health. The outcome measured is
  **quantity sold**, which moves with price, cross-border purchasing and mail order — all of which
  would *understate* the tax's effect on legal sales.
- I did not claim the minimum-wage result applies to adults, to full-time employment, or to
  employment in other industries. The data only covers teenagers in county-level payroll records.
- I did not report a single synthetic-control number as if it were the answer. Across four
  defensible specifications the effect ranges from −19.8 to −24.7 packs. The *width of that range*
  is the finding; the point estimate taken from the middle of it is not.

## One thing to take away

In the absence of a randomised trial, the honest measure of what the data supports is the spread
across defensible analyses — not the number in the middle of the table. Both of these estimates
survive that test; only one of them survives conventional significance testing.
