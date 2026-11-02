# Methods audit, 2026-11-02

## Notebook dependency cleanup, 2026-11-03

Removed ten MF test files and the local ARC test file after all 109 tests
passed. Removed unused solvers from `methods_mat_fac.py` and
`methods_legacy.py`, duplicate oracle definitions, unused plotting functions,
and generated caches. `utils_new.run_methods` now uses the existing runner
from `utils.py`. Alternating descent, the older dense cubic entry point, both
cubic backends, all plotting helpers used by notebooks, and the local ARC
experiment remain. The local ARC folder is still excluded from Git.

The three older experiment notebooks now import their dependencies explicitly.
Two unused residual reads were removed from each. One of those reads accessed
`func_at_lssolve` on PALM and would fail with the current alternating notebook's
method order; its plots already use `func`, so removing the read does not
change a plotted value. Other settings, seeds, outputs and user edits are
preserved. All retained solver and plotting function bodies are unchanged;
the oracle counter only loses unused interfaces. No acceptance test, tolerance,
Hessian reuse policy, initialization, regularization or subproblem solve changed.

Before/after checks compare 44 runs at dimensions (6,4,2), seeds 549/498,
noise .2, alpha .01, beta 500 and initial factor standard deviation .05.
They cover gradient and alternating descent, LeAP/GLAd with exact/GN and
fresh/lazy Hessians, dense cubic, GLRT, the legacy cubic wrapper and ARC.
Final iterates, statuses, non-timing histories and MF metrics match exactly.
The 82 local notebook imports resolve. All eight experiment notebooks
(including local ARC) and five plotting notebooks execute from fresh kernels
with small settings; plotting inputs come from those isolated small runs.
These are correctness checks, not a reproduction or timing benchmark.

Validation used Windows 11, Python 3.12.14, NumPy 2.4.6, SciPy 1.18.0,
GALAHAD 5.5.3 and float64, with one BLAS/OpenMP thread. The parallel check
used two seed workers. Tests, original sources, data and validation artifacts
are preserved in a `mf-cleanup-20261103-*` recovery directory in the local
temporary folder. Existing reference outputs and historical diagnostics remain.
Paper alignment is unchanged: arXiv:2602.08069v1, Section 5.2 / Appendix E.2;
see `CUBIC_SUBPROBLEM.md` and notebook settings for figure inputs and parameters.

This audit separates structural cleanup from numerical corrections. Notebook
settings, seeds, saved results and figures were not edited. The initial audit
preserved public imports; subsequent changes below record their locations and removals.

## Follow-up: end mean curves at the target or final recording

Each method in the final arithmetic-mean figure now ends at the first shared
iteration/time where every seed's current recorded norm is <= the display
target. A mean below the target is insufficient, and an earlier individual hit
that later rebounds does not count as currently below target. If this never
happens, the curve ends at that method's last recorded point, rather than being
extended to the longest other method. All seeds remain in the arithmetic mean.

`mean_gradient_target` defaults to the completed batch's recorded `grad_tol`
(currently 1e-20), independently of any later edits to the settings cell.
Changing it only changes the display. The notebook lists missed targets per
axis; these endpoints are not reported as convergence. With `target=None`,
only each method's last recording determines its endpoint. Genuine plateaus
within recorded histories remain. Full numerical means are unchanged; the
returned summaries and optional export add endpoint indices, coordinates and
all-seed target flags. Axis limits follow displayed endpoints unless zoomed.

Only the last two notebook cells and the final-mean plotting function changed;
the first 27 cells/outputs, aggregation formula, solvers and all experiment
settings are preserved. This remains a display convention for the MF gradient
metric (arXiv:2602.08069v1, Section 5.2 / Appendix E.2 / Figure 4), not a new
solver stopping rule or an asserted convention from the paper.

All 100 tests pass, including target cutoffs, above-target seed retention and
rebounds. A fresh small kernel run (seeds 549/498, dimensions (6,4,2), 20 GD /
five Newton updates, otherwise current settings) verifies all 11 mean curves,
both grids and full arithmetic means independently. It checks both missed
strict targets and successful display-only targets; all 13 notebook figures
complete. Saved six-seed histories also render correctly. This is a plotting
check, not full paper reproduction. Snapshots and executed/rendered checks are
in `_diagnostics_archive/mean_plot_cutoff_20261102_195513/` (ignored).

## Follow-up: LeAP in the final mean figure

The final mean section also selects LeAP-SSN and its GN counterpart when
recorded for every seed: black solid and black dashed, respectively. With
the current GN exponents (.2,.5), the figure has 11 curves. Only the last two
notebook cells changed; earlier cells/outputs, solver settings and the mean
plot helper are unchanged. The settings already present at this edit were
alpha 1, beta 500, initial factor std .5, 1,000 Newton updates and plotting
threshold 1e-20; these user changes are retained.

The five mean tests pass. A small fresh-kernel check uses seeds 549/498,
dimensions (6,4,2), 20 GD/five Newton updates and otherwise the current settings.
It completes all 13 figures and verifies both means independently, including
the two LeAP curves. Doubling all recorded timestamps doubles the runtime
grid and leaves the iteration panel unchanged, confirming that runtime uses
measured seconds. Both panels retain the same final seed values, so matching
terminal plateaus are expected. This check is not a full reproduction; sources,
the rendered figure, configuration and executed notebook are in the ignored
`_diagnostics_archive/mean_plot_leap_20261102_191604/` folder.

## Follow-up: GLAd and Gauss-Newton variants in the final mean figure

The last mean section now selects GLAd p=.2,.4,.5,.6,.8 and each corresponding
GN history available for every seed, alongside GD and cubic GLRT. Exact/GN
pairs share a color; GN curves are dashed. The saved batch settings already
compute GN p=.2,.5, so the current view has nine curves. Missing variants are
listed explicitly; adding their exponents to `gn_exponents` and rerunning the
batch makes them appear automatically. No histories are fabricated or seeds
dropped. The export record includes line styles and missing method names.

The first 27 notebook cells, their outputs and all experiment settings remain
unchanged. `mean_gradient_histories` is unchanged; only selection, styles and
legend space changed. This remains the supplementary Figure 4 gradient metric
described below, not a change to Hessian approximations or solver behavior.

All five focused mean tests pass. A fresh-kernel check with two seeds and small
dimensions temporarily enabled all five GN exponents in memory: 28 solver runs,
with the 22 existing baseline runs exactly unchanged apart from timing. Both
the nine-curve current-data view and twelve-curve full-pair view were checked
numerically and rendered. Source snapshots and checks are in the ignored
`_diagnostics_archive/mean_plot_variants_20261102_190120/` folder. This is a
small plotting check, not a full experiment rerun.

## Follow-up: mean with terminal values retained

Two cells are appended after the representative plots. All previous 27 cells,
their outputs and notebook metadata are unchanged. The additional figure shows
the arithmetic mean gradient norm for GLAd p=.5, cubic GLRT and GD, against
runtime and iteration. It reads `all_seeds_histories`, retaining each run's
actual final value after any exit, including failures and iteration limits.
No run is discarded, forced to zero, or stopped at an earlier target crossing.
Both axes extend to the longest selected run by default; optional limits only
zoom the display. The numeric summaries remain available in the notebook.

`mean_plots.py` aligns iteration indices directly. Runtime uses a common grid
of recorded timestamps, starting when every run has a recorded observation;
at each time it uses the latest observation, with the last duplicate winning.
This is a mean of available returned-solution norms, not predicted continuation
of a terminated solver. No time arrays are averaged and no logarithms enter
the arithmetic mean. The linear region below 1e-14 only makes zeros visible.

This is an additional display of the MF gradient metric mapped to Section 5.2,
Appendix E.2 and Figure 4 of arXiv:2602.08069v1 in `CUBIC_SUBPROBLEM.md`.
The local revision's 20-seed averaging experiment does not prescribe this
alignment convention. Current six-seed notebook settings are unchanged, as
are solvers, tolerances, stopping rules, initialization and Hessian handling.
The original averages and medoid figures remain available.

All 97 tests pass, including five checks of unequal lengths, unsuccessful
plateaus, duplicate timestamps, gradient rebounds, zeros and display limits.
A fresh kernel with seeds 549/498, dimensions (6,4,2), 20 GD and five Newton
updates completed 13 figures. All 22 solver histories/statuses match the prior
baseline exactly apart from timing, and both new means match an independent
budget-by-budget calculation. Rendered checks also use saved six-seed data.
This is a small correctness check, not full paper reproduction. Snapshots,
test output, PNG previews and the executed notebook are preserved in the
ignored `_diagnostics_archive/mean_plot_20261102_184448/` folder.

## Follow-up: supplementary representative-run plots

Five cells are appended to the parallel notebook; the existing 22 cells and
their outputs are unchanged. `representative_plots.py` selects one shared seed
and plots its objective residual, gradient and weighted penalty against runtime
and iteration. Curves use complete recorded histories, including zeros and
rebounds. The iteration view defaults to a labelled 150-update zoom; runtime
shows the full range. The objective reference is the best recorded value across
the plotted methods for that seed, not a known optimum.

The selection is a medoid of first-hit iteration ranks at fixed gradient targets
`[1e2, 1, 1e-2, 1e-4, 1e-6, 1e-8]`. GD, LeAP, GLAd p=.5 and cubic GLRT receive
equal weight, avoiding extra weight from multiple GLAd variants. Unreached
targets tie last; the smallest seed breaks numerical ties. The same selected
seed is used for every plotted method and metric. This is a documented
diagnostic convention, not a standard optimization benchmark protocol; central
gradient milestones need not imply central penalty behavior. Runtime and
relative method winners do not influence selection. An all-seed target-time
plot and exit-status table retain failed/unreached cases. Its reporting target
`1e-6` does not change the solver's `grad_tol=1e-20`.

The public mapping remains arXiv:2602.08069v1, Section 5.2 / Appendix E.2,
Figures 3, 4 and 6. The locally available manuscript `Lazy_Hessians_Neurips-20.pdf`
(PDF metadata dated 2026-10-22) additionally specifies 20-seed averages in
Appendix E.2 and Figure 7. Existing averages are therefore retained. Neither
reference prescribes this representative-seed rule, and current notebook
settings differ from the manuscript. The local file is not independently
verified as the final accepted version; reviewer comments are not treated as
adopted changes. No solver, tolerance, stopping rule, initialization or seed
setting changed.

All 92 tests pass, including ten selection/plot checks. A fresh kernel with
seeds 549/498, dimensions (6,4,2), 20 GD and five Newton updates completed all
12 figures. Its 22 solver runs match the prior baseline exactly apart from
timing. New figures were inspected as PNGs; existing outputs were preserved.
The saved six-seed diagnostic histories select seed 483 under this rule.
These are small validation runs and reused diagnostic data, not full paper
reproduction. Snapshots, rendered checks, the selection record and the executed
notebook are in the ignored
`_diagnostics_archive/representative_plots_20261102_181623/` folder.

## Follow-up: show zero and nonmonotone penalties

The parallel notebook's penalty figures use `utils_new.plot_figures` and the
recorded weighted penalty (paper Figure 3 / Section 5.2 / Appendix E.2). The
helper previously stopped a curve at its first value <= `plot_threshold` and
used a log axis that cannot display zero. In saved six-seed data, GD reaches
zero at iteration 13, then becomes positive again; its remaining history was
hidden. An entirely zero series produced no visible curve and log warnings.

Penalty panels now retain all entries allowed by `n_iters`, show exact zeros
with a linear region up to `plot_threshold` (1e-10 here), and use a logarithmic
scale above it. A one-point history gets a marker. Objective/gradient threshold
behavior, solvers, seed averaging and original outputs are unchanged. Only the
notebook's explanatory plotting text changed; rerun imports and plotting cells
to use existing in-memory histories with the corrected helper.

All 82 tests pass, including zero/rebounding penalties and unchanged gradient
trimming. A fresh small parallel run completed all eight figures; all four
penalty panels contain every recorded mean value, and 22 solver histories
match the prior baseline. Before/after figures, source snapshots and the
executed notebook are in the ignored
`_diagnostics_archive/penalty_plot_fix_20261102_172526/` folder. This is a display
correction, not a full experiment rerun or a change to the penalty formula.

## Earlier follow-up: simplify methods.py for valid experiment inputs

`methods.py` is now 600 lines, down from 790. Explicit input type, range and
shape checks were removed; callers supply valid vectors and solver settings.
The old gradient-tolerance validator is local to `methods_legacy.py`, preserving
its older callers. Dense/GLRT subproblem checks remain in their own modules.
Unsupported method names fail through the normal backend lookup.

CG/MINRES are called directly with `rtol`, as verified in installed SciPy 1.18.0;
support for old `tol`-only interfaces was removed. The unused iteration callback
and duplicate solver dispatch are gone. Alternating descent shares one block
backtracking routine. Docstrings are shorter, and SSN progress respects
`warnings=False`. Unexpected programming errors now propagate instead of being
caught as numerical failures; numerical solve failures still retry as before.

Public solver signatures, formulas, regularization, acceptance tests, Hessian
reuse, stopping criteria and recorded statistics are unchanged for valid inputs.
Finite-value checks, failed-solve handling and stagnation checks remain because
they prevent invalid steps and false convergence, even with valid inputs.

All 79 remaining tests pass; only the removed input-validation test was deleted.
All 62 direct runs match exactly in iterates, statuses and non-timing histories:
seeds 549/498, `(6,4,2)`, noise .5, alpha 5, beta 200, initial std 1.5, 20 GD /
10 alternating / five Newton updates, one library thread. Cases include metrics,
tracing on/off, CG/MINRES, exact/Gauss-Newton, fresh/lazy Hessians, dense/GLRT and
legacy cubic. Fresh kernels also ran the parallel notebook (22 exact baseline
matches, eight plots) and comparison notebook (eight methods, including alternating).
Notebook sources and reference outputs are unchanged. These are small checks,
not full paper reproduction; figure/settings mapping remains in the cubic guide.
Inputs, sources and executed checks are preserved separately in the ignored
`_diagnostics_archive/simplify_methods_20261102_165942/` folder.

## Earlier follow-up: remove the custom Krylov backend

Removed `cubic_newton_step_nonconvex_krylov`, `_CubicKrylovModel`, the
`subproblem_solver="krylov"` branch, `krylov_max_iter` / `krylov_seed`, and
its dimension diagnostics. At that stage, selecting it raised `ValueError`;
the removed keywords are no longer accepted. Its 11 tests were removed with it.
The remaining choices are `"glrt"` and the `"dense"` reference, with the
default still dense. No notebook selected the custom backend. GLRT's own
Lanczos/Krylov implementation and documentation remain unchanged.

This only removes an optional comparison baseline from the MF experiments
mapped to Section 5.2 / Appendix E.2 in the [cubic guide](CUBIC_SUBPROBLEM.md).
The retained solvers' acceptance rules, tolerances, Hessian reuse, initialization,
seeds and stopping conditions were unchanged. `methods.py` became 790 lines and
`cubic_subproblems.py` 228 lines: 209 implementation lines removed.

All 80 remaining tests pass. Four dense/GLRT runs (two seeds, `(6,4,2)`, five
updates, normal data/start, alpha 5, beta 200, H_0=10, gradient tolerance 1e-20,
subproblem tolerance 1e-7) match exactly in final iterate, status and non-timing
history. A fresh-kernel parallel notebook check uses seeds 549/498, `(6,4,2)`,
20 GD / five Newton updates and otherwise the notebook settings. All 22 method
histories, statuses and cubic diagnostics match exactly; all eight plots complete.
Notebook source and original outputs are unchanged. The pre-removal baseline,
sources and executed check are in the ignored
`_diagnostics_archive/remove_custom_krylov_20261102_165038/` folder. These are
small float64 checks in the environment below, not a full paper reproduction.

## Earlier follow-up: keep the parallel solvers in methods.py

After the module split, `methods.py` contained the four solvers used by the parallel notebook:
`gradient_method`, `mf_leapssn`, `mf_gladssn` and `cubic_newton_nonconvex_new`,
plus `alternating_gradient_method` at the user's request and their required
helpers. It was 813 lines, down from 1,157 after the audit. `vanilla_newton`,
`super_newton`, `fast_gradient_method`, `cubic_newton` and `cubic_newton_step`
moved unchanged to `methods_legacy.py`. Two older notebooks now import legacy
cubic from that module; alternating notebook imports stay unchanged. Standalone
cubic steps are imported directly from `cubic_subproblems.py` or `cubic_glrt.py`;
they are no longer re-exported here.

All function syntax trees are identical across this move. Solver tolerances,
Hessian reuse, regularization, acceptance rules, seeds and stopping behavior
are unchanged. The parallel notebook source and all saved outputs are retained.
All 91 tests passed: the existing 88 plus three alternating checks covering
sequential block gradients, the full-gradient stopping tolerance, and descent
to a known stationary point with identical traced/untraced results. The U and V
gradients agree with the same penalized MF objective; V uses the updated U.
The existing failed-block regression also passes. No numerical correction was
needed in this follow-up.

Fresh kernels ran the parallel notebook with the two-seed configuration below
(20 GD / five Newton updates), and the comparison notebook with `(6,4,2)`, seed
0 and 20 updates. All 22 parallel histories, statuses and cubic diagnostics
match the pre-move baseline exactly, excluding timing; all eight parallel plots
and all eight comparison methods, including alternating, completed. Eight
alternating baselines also match exactly: seeds 549/498, `(6,4,2)`, standard
normal data/start, alpha 5, beta 200, 20 updates, initial block constants 512,
`tol=1e-20`, with line search and tracing each on/off, one library thread.
All solver import statements in the MF notebooks resolve. These are small
checks, not full reproduction. Pre-move sources, baseline inputs/results and
executed notebooks are preserved separately in the ignored
`_diagnostics_archive/parallel_methods_only_20261102_163601/` folder.

## Initial audit: simplification

- LeAP and GLAd now share one loop. Their distinct initial shifts, factors
  of two/four, GLAd fallback, Hessian reuse and acceptance tests are preserved.
- Dense and custom Krylov cubic subproblems moved to `cubic_subproblems.py`;
  `methods.py` re-exports their existing names. Their executable syntax trees
  are unchanged. GALAHAD remains in `cubic_glrt.py`.
- Both cubic outer APIs use the same checked loop. The long cubic notes were
  replaced by a current [reference guide](CUBIC_SUBPROBLEM.md).

`methods.py` decreased from 1,840 to 1,157 lines. Including the new 414-line
subproblem module, the implementation totals 1,571 lines, about 15% less than
the original file. Added failure safeguards account for some retained length.
Original framework attribution remains in [THIRD_PARTY.txt](../THIRD_PARTY.txt).

## Corrections that can change results

| Routine | Correction and effect |
| --- | --- |
| `vanilla_newton` | Tests the actual gradient against `grad_tol`; previously any supplied tolerance caused immediate success. Handles `trace=False` and failed/nonfinite solves. |
| `super_newton` | Retries failed factorizations with increased regularization, stops on exhausted searches, handles zero gradient and `trace=False`, and copies history lists rather than aliasing them. |
| Gradient / accelerated gradient | Do not apply rejected or nonfinite trials after search exhaustion. `Binv` alone now defines the same metric as supplying both `B` and `Binv`. |
| Accelerated gradient | Records the gradient at the accepted point; the previous squared-gradient history stayed at its initial value. |
| Alternating gradient | Stops with an explicit failed-block status. If U was accepted before V failed, the returned point and history retain that U update. |
| LeAP / GLAd | Final time and oracle counters include unsuccessful search work. Iterates, stopping decisions and acceptance arithmetic are unchanged. |
| Legacy cubic | Replaces the unsafe scalar root search with the dense global reference; checks failure status and model decrease, rejects exhausted searches, supports `trace=False`, and fixes stale gradient histories. |
| Cubic outer loop | Uses a stable norm fallback when a finite nonzero gradient's ordinary norm underflows or overflows; normal-scale arithmetic is unchanged. |
| GLRT diagnostics | Retains native work counters when a nonfinite Hessian-vector product aborts a trial. The failure status and returned step are unchanged. |

The legacy `cubic_newton_step` counterexample is `g=[1]`, `A=[[-2]]`, `H=1`:
the old routine returned step `+1` with positive model change `1/3`; the
correct global step is `-1-sqrt(2)`. This is an algorithm correction, not a
refactor. Old results using `cubic_newton` should be regenerated. Its `lss_*`
keywords remain accepted but no longer select linear solvers. Dense cost is
O(d^3); large experiments should use the explicit GLRT backend already selected
in the two cubic comparison notebooks. Fixed-regularization legacy mode makes
one trial per update and reports an unusable solve without changing H.

No new outer tolerance, seed, initialization, lazy schedule or inner tolerance
was imposed on callers. The LeAP/GLAd `H_min` argument was already unused and
remains so; its docstrings now say this explicitly. Their linear solves remain
inexact. Numerical stagnation remains a failure above a requested tolerance,
not a convergence certificate. This audit does not change plot aggregation.

## Validation

Before editing, all 62 existing tests passed and 38 small MF runs were saved.
The final suite passes **88 tests**, including indefinite cubic hard cases,
rejected/failed searches, trace-disabled execution, stopping boundaries,
metric handling, gradient histories, and final work counters. Run from the
repository root in an environment with the optional GALAHAD package installed:

```text
python -B -m unittest discover -s MF -p "test_*.py"
```

The 38 baselines use seeds 549 and 498, dimensions `(6,4,2)`, noise .5,
alpha 5, beta 200, Gaussian initialization standard deviation 1.5 and one
library thread. GD used 20 updates; accelerated/alternating GD used 10;
cubic methods used five. LeAP/GLAd used 10 updates, H_0=10, CG or MINRES
tolerance 1e-7 / cap 800, fresh/exact, Gauss-Newton and lazy cases (period 3).
New cubic backends used H_0=10, tolerance 1e-7 and GLRT cap 6,000.

All 36 non-legacy-cubic final iterates and exit statuses match exactly.
Non-timing histories also match except the intentionally corrected accelerated
gradients and failed-work counters. All six dense/Krylov/GLRT outer runs are
unchanged. The two corrected legacy cubic runs differ from the old iterates by
at most 1.54e-10 per coordinate on these examples; this does not bound changes
on indefinite or failed-search examples, where the old algorithm was wrong.

Fresh kernels executed the parallel and serial cubic notebooks on two seeds
with dimensions `(6,4,2)`, 20 GD and five Newton updates. All methods and plots
completed. The legacy MF notebook also executed with these reduced dimensions
and budgets; its exports were redirected to the audit folder. No full-size,
six-seed experiment, paper reproduction or timing benchmark was performed.

Environment: Windows, Intel Core i7-11800H, Python 3.12.14, NumPy 2.4.6,
SciPy 1.18.0, Matplotlib 3.11.0, Joblib 1.5.3, GALAHAD Optrove 5.5.3,
float64. Timing fields are excluded from equality checks. The ignored local
`_diagnostics_archive/methods_audit_20261102_123728/` contains source snapshots,
configurations, per-run objective/gradient/penalty statistics, comparisons and
executed notebooks. The [cubic guide](CUBIC_SUBPROBLEM.md#experiment-and-figure-mapping)
maps the notebooks and plotting inputs to Section 5.2 / Appendix E.2 of
[arXiv:2602.08069v1](https://arxiv.org/abs/2602.08069v1).
