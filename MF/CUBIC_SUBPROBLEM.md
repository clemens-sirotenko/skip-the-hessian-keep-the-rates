# Cubic Newton methods: implementation and references

This note describes the MF cubic baselines and the dense reference used to
check them. Import the outer `cubic_newton_nonconvex_new` from `methods.py`,
the dense subproblem solver from `cubic_subproblems.py`, and
`cubic_newton_step_glrt` from `cubic_glrt.py`. The older `cubic_newton` and
`cubic_newton_step` entry points now live in `methods_legacy.py`.

## Model and reference

For gradient `g` of shape `(d,)`, symmetric Hessian `A`, and coefficient
`sigma > 0`, the subproblem is

\[
m(s)=g^Ts+\tfrac12s^TAs+\tfrac{\sigma}{3}\|s\|_B^3,
\qquad \|s\|_B=\sqrt{s^TBs}.
\]

`B` is symmetric positive definite; `B=None` means the Euclidean norm.
In subproblem calls, the argument named `H` is **sigma**, not the Hessian.
The outer methods pass `H=H_k/2`, giving the cubic term `H_k/6 * ||s||**3`.
All implementations use real float64 arithmetic.

The main reference is Cartis, Gould and Toint (2011), *Adaptive cubic
regularisation methods for unconstrained optimization. Part I: motivation,
convergence and numerical results*, Mathematical Programming 127, 245-295:
[DOI](https://doi.org/10.1007/s10107-009-0286-5),
[author-hosted paper](https://www.numerical.rl.ac.uk/media/people/nick-gould/CartGoulToin11_mp.pdf).
The relevant parts are:

- Theorem 3.1: global optimality requires `(A + lambda*B)s = -g`,
  `lambda = sigma*||s||_B`, and `A + lambda*B` positive semidefinite.
- Section 6.1, equations (6.1)-(6.6): spectral reduction, scalar norm equation,
  and the hard case, where an eigenspace component must be added.
- Section 6.2, equations (6.16)-(6.19): approximate solution in a Lanczos space.
- Section 6.3: extension to a positive definite norm matrix.

Nesterov and Polyak (2006), *Cubic regularization of Newton method and its
global performance*, is the foundational cubic-regularization reference:
[DOI](https://doi.org/10.1007/s10107-006-0706-8).
Neither citation makes the present MF implementation a reproduction of its
outer algorithm or a proof that its assumptions hold for this penalized,
semismooth objective.

## Which function to use

| Function / backend | Purpose and limitations |
| --- | --- |
| `cubic_newton_step_nonconvex_new` / `"dense"` | Small-problem reference. Materializes `A`, diagonalizes it, solves the scalar equation, and checks stationarity, shifted curvature and model decrease. Supports `B`. Storage is O(d^2), arithmetic O(d^3). |
| `cubic_newton_step_glrt` / `"glrt"` | Established GALAHAD Lanczos implementation. Suitable for larger problems; Euclidean norm in this adapter. Uses Hessian-vector products without forming the full Hessian. |
| `cubic_newton_nonconvex_new` | Outer method selecting `"dense"` or `"glrt"`. The default remains `"dense"` for compatibility; the current cubic notebooks explicitly select `"glrt"`. |

Dense success is a numerical check of global model optimality to tolerance.
GLRT starts from the gradient and may miss negative curvature
orthogonal to it. Its native success plus a small residual is only an inexact
first-order model solve. At zero gradient, GLRT returns zero without searching
for negative curvature; dense can take a negative-curvature step.

## Outer acceptance and stopping

`cubic_newton_nonconvex_new` evaluates a fresh generalized Hessian at each
accepted outer point. Dense decompositions are reused
across its backtracking trials. GLRT starts a fresh native solve per trial,
reusing only the fixed Hessian callable. There are no lazy Hessian updates.

A trial is accepted only after a successful finite subproblem solve with

```text
model_change <= 0
f(x + step) <= f(x) + model_change
```

Rejected trials double `H_k`. Acceptance halves it with floor `H_min=1e-5`.
`max_backtracks` allows that many retries after the initial trial. Dense solve
failures may retry with a larger coefficient; GLRT failures stop
immediately, preserving the last accepted point and the failure status.

This is a model-majorization backtracking rule. It is **not the CGT ARC
algorithm**, whose equations (2.4)-(2.6) use an actual/predicted reduction ratio
and separate success thresholds. No ARC ratio or Cauchy-step test is added here.

`grad_tol` is an optional absolute outer gradient tolerance. It is independent
of subproblem accuracy and plotting thresholds. A failure or a step lost to
rounding can stop a run above that tolerance; `numerical_stagnation` does not
mean convergence. `iterations_exceeded` means the outer budget was used.
Objective, gradient and penalty histories describe accepted points. Runtime
includes rejected trials; the last timestamp also includes work before failure.

## GALAHAD controls and cost

Install the pinned interface from the repository root:

```text
python -m pip install -r MF/requirements-glrt.txt
```

The distribution is `galahad-optrove==5.5.3`, not the unrelated `galahad`
package. Other backends do not require GALAHAD. See the
[official GLRT documentation](https://ralna.github.io/galahad_docs/html/Python/glrt.html).
Upstream authors, license and framework attribution remain in
[THIRD_PARTY.txt](../THIRD_PARTY.txt).

The adapter uses `power=3`, `weight=H_k/2`, `unitm=True`, `stopping_rule=3`,
`stop_relative=subproblem_tol`, `stop_absolute=0`, `fraction_opt=1`, and
`rminvr_zero=0`. It independently checks the reconstructed step:

```text
||g + A@s + sigma*||s||*s|| <= tol*||g||
                              + 64*u*(||g|| + ||A@s|| + sigma*||s||**2)
sigma = H_k/2; u = float64 machine epsilon
```

The dense tolerances instead use scaled residual and curvature
checks. Equal tolerance numbers do not imply identical subproblem accuracy.
MINRES settings do not control any of these cubic subproblem solvers.

GLRT's cap counts native iterations, not total Hessian-vector products.
`hess_vec_calls` includes reconstruction and the independent residual check;
`glrt_iterations`, `glrt_status` and `subproblem_relative_residual` provide
additional diagnostics. The native state is protected by a process-local lock.

The 5.5.3 Python wrapper can lose partial reconstruction sums during reverse
communication. `extra_vectors=max_iter+1` retains the Lanczos vectors to avoid
that path; see the [upstream wrapper](https://github.com/ralna/GALAHAD/blob/master/src/glrt/Python/glrt_pyiface.c).
This costs O(d*max_iter) storage: about 173 MB for d=3,600 and cap 6,000,
or 14.4 MB for d=300, plus workspace. A native status-zero result at the cap
still fails if its independent residual is too large.

## Legacy entry points: correctness changes

`methods_legacy.cubic_newton_step` delegates to the dense global reference. Its old scalar
root search could select a nonglobal stationary point for an indefinite
Hessian: `g=[1]`, `A=[[-2]]`, `H=1` returned step `[1]` and positive model
change `1/3`. The global step is `[-1-sqrt(2)]`.
The legacy linear-solver keywords remain accepted for caller compatibility
but no longer control this dense solve. This changes backend, cost and possibly
trajectories; the legacy step is no longer a matrix-free implementation.

The `cubic_newton` signature retains its metric, objective-target and adaptive
search controls. It now checks solve status and accepted candidates, preserves
the iterate after an exhausted search, supports `trace=False`, and records the
gradient at the same point as the objective. Older plots used stale gradients,
and some failed/rejected steps could be applied. Those results require a rerun.
With `adaptive_search=False`, the outer majorization test is disabled; do not
interpret that setting as the globalized method described above.

## Experiment and figure mapping

The project reference is [arXiv:2602.08069v1](https://arxiv.org/abs/2602.08069v1),
Section 5.2 / Appendix E.2. GLAd is Algorithm 1 with quadratic subproblem (2);
these cubic methods are comparison baselines. The public MF objective is
(28); `oracles_mat_fac.py` defines the repository's ridge/penalty weights.
The ignored local revision and review PDFs are not needed to use this note.
No reviewer suggestion is represented here as an adopted paper change.

Settings snapshot from the notebook source on 2026-11-02:

| Notebook | Current data and stopping settings | Plot inputs |
| --- | --- | --- |
| `experiment_matrix_factorization.ipynb` | Seed 0; (m,n,r)=(200,100,12); noise 0; alpha .01, beta 200; initial factor std .5; legacy cubic H_0=1. | `histories`, including legacy cubic `func_at_lssolve`. |
| `experiment_matrix_factorization_multiple_cubic_new.ipynb` | Seed 483; (200,100,12); noise .5; alpha 5, beta 200; initial std 1.5; 3,000 GD / 2,000 Newton updates; no outer gradient tolerance. | `all_seeds_histories`, then `averaged_history`. |
| `experiment_matrix_factorization_multiple_cubic_new_parallel.ipynb` | Seeds 549,498,483,33,528,989; effective (50,25,4); noise .5, alpha 1, beta 500; initial std .5; 3,000 GD / 1,000 Newton updates; `grad_tol=1e-20`; six workers, one library thread each. | Same averages; combined, representative and final mean plots use `all_seeds_histories` directly. |

Both multiple-seed notebooks select GLRT with H_0=10, tolerance 1e-7, iteration
cap 6,000 and 120 backtracks. LeAP/GLAd use MINRES tolerance 1e-7 / cap 800.
The parallel settings cell contains an earlier (200,100,12) assignment that
is overridden by (50,25,4); the table records the effective value.

Paper Figure 3 maps to weighted `penalty`, Figure 4 to `grad_norm`, and Figure 6
to `func` minus the plotting reference. Runtime uses `time`; original means
align iterations and truncate at the shortest seed. The parallel `plot_threshold=1e-20`
trims objective/gradient displays. Penalty plots keep all recorded mean values;
their scale is linear up to this threshold and logarithmic above, showing zeros
and later rebounds. These settings differ from Appendix E.2, and the extra
cubic/combined plots are not claims of additional published figures. Parallel
elapsed times include shared-CPU contention, not isolated timing benchmarks.

The appended **Representative actual run** section reuses a completed batch.
Its tables require `pandas` in addition to the notebook's plotting dependencies.
It chooses one shared seed by the smallest average rank distance from the other
seeds at fixed gradient milestones, equally weighting GD, LeAP, GLAd p=.5 and
cubic GLRT. This applies the general
[medoid idea (Kaufman and Rousseeuw, 1987)](https://wis.kuleuven.be/stat/robust/papers/publications-1987/kaufmanrousseeuw-clusteringbymedoids-l1norm-1987.pdf)
to convergence histories; it is not a rule from the MF paper. Unreached targets
tie last and the smallest seed breaks ties. Runtime does not affect selection.
The selection settings and scores are displayed; `SAVE_FIGURES=True` writes
figures and a selection record to a new `parallel_results/representative_*`
directory. The same seed supplies all actual curves, with the best observed
objective for that seed as reference. Zeros remain visible; iteration defaults
to a labelled 0-150 zoom (`representative_max_iter=None` shows the full range).
An additional target-time plot includes every seed and distinguishes unreached
targets from successful first hits. Final gradients and exit statuses are also
shown; its reporting tolerance does not change solver stopping.

The local manuscript with PDF metadata dated 2026-10-22 explicitly reports
20-seed means in Appendix E.2 / Figure 7. This is later than the public v1
reference above, and its final acceptance/version identity is not independently
verified. The new plots are supplementary: the notebook's existing averages
remain, and neither version supplies a representative-seed selection rule.

The final **Mean with terminal values retained** section adds a gradient
figure for LeAP, GLAd p=.2,.4,.5,.6,.8, cubic GLRT and GD using `all_seeds_histories`
directly. Each available GN counterpart uses the same color with a dashed line;
LeAP and LeAP GN are black.
The current settings record GN p=.2,.5; missing variants are listed and appear
automatically after including their exponents in `gn_exponents` and rerunning
the batch. A method is shown only when it is recorded for every seed.
Every seed retains equal weight after any exit; its final recorded norm is
held constant, without replacing it by zero or stopping at an earlier target
hit. Each displayed curve ends when every seed's current norm is at or below
`mean_gradient_target`, or at that method's last recording if the target is
missed. The target defaults to the completed batch's `grad_tol`; this is only a
display cutoff. It tests all seeds, not the mean alone, and does not count a
previous hit followed by a rebound as currently below target. Missed targets
are listed explicitly. The full means and endpoint details are retained in
`mean_gradient_summaries`. Runtime
uses the latest recorded value on a shared timestamp grid, starting when all
runs have an observation; duplicate times use the last observation. This
describes the available returned solutions, not unobserved solver progress.
`mean_gradient_max_iter` and `mean_gradient_max_time` only zoom the view.
`SAVE_FIGURES=True` saves this figure and its numeric means/statuses separately
under `parallel_results/mean_gradient_*`. This is an additional averaging
convention for the Figure 4 metric, not a claim of exact paper reproduction.

## Validation

The current suite passes 100 unit tests, including penalty display,
representative-selection, mean-alignment and target-cutoff regressions.
Simplifying `methods.py` preserves all
62 small solver runs exactly in final iterates, statuses and non-timing histories,
including dense and GLRT. The dense solver and GLRT adapter are unchanged.
The earlier legacy cubic corrections intentionally altered its
backend, failure handling and gradient history; see [the audit](METHODS_AUDIT.md).

After simplification, a fresh kernel ran the parallel notebook with seeds 549/498,
dimensions (6,4,2), 20 GD and five Newton updates. All 22 method histories,
statuses and cubic diagnostics match the baseline exactly, excluding time; all
eight plots completed. With the supplementary section appended, a fresh check
of the same configuration completed all 12 figures and again matched all 22
non-timing solver histories exactly. Adding the final mean figure raises this
to 13 figures, with the solver histories still identical and its means verified
independently at every grid point. A fresh comparison notebook also ran all eight methods,
including alternating descent. Earlier checks ran the serial and legacy cubic
notebooks. These are small correctness checks, not full reproduction or a timing
benchmark.
Validation used Windows, Intel Core i7-11800H, Python 3.12.14, NumPy 2.4.6,
SciPy 1.18.0, GALAHAD 5.5.3 and float64; numerical comparisons used one library
thread and excluded timings. Refactor sources and baseline are preserved in the
ignored `_diagnostics_archive/simplify_methods_20261102_165942/` folder;
representative-plot snapshots, rendered previews and the executed notebook are
in `_diagnostics_archive/representative_plots_20261102_181623/`.
The added mean figure's validation is in
`_diagnostics_archive/mean_plot_20261102_184448/`.
