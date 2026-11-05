# SVM companion files

These are deliberately close copies of the existing SVM implementation. The
original `.py`, `.ipynb`, and `.pdf` files are unchanged. No reference results are
overwritten. Use `experiment_SVM_new.ipynb` for the corrected comparison and
`plot_SVM_from_data_new.ipynb` to replot one explicitly selected result export.

## Paper and figure mapping

Public reference: [arXiv:2602.08069v1](https://arxiv.org/abs/2602.08069v1),
Section 5.3 and Appendix E.3. Algorithm 1, equation (2), gives the regularized
Newton subproblem; equation (3) gives the acceptance inequalities; equation (4)
gives the Hessian refresh schedule. In the local October 22, 2026 revision,
these equations are numbered (1), (2), and (3), respectively. Editorial and
reviewer comments in that draft are not treated as adopted algorithm changes.

The source experiment for the local revision's Figure 5 is
`experiment_SVM.ipynb`: 10,000 samples, 1,000 features, C=10,000,
`make_classification(random_state=43, flip_y=0.2, class_sep=1.5,
n_clusters_per_class=1)`, and lazy frequencies 2, 4, 5, 10 in addition to m=1.
The other methods are gradient descent, SVRG-BB, cubic Newton and LeAP-SSN.
The current `experiment_SVM_510.ipynb` instead uses 500 features and only three
methods. Public v1 specifies 200 features; do not conflate these versions.

The old main notebook exports `svm_results_full.pkl` and calls
`SVM_utils.plot_for_paper_2`. The paper embeds an earlier `plot_SVM_func.pdf`
with a five-second time window; the checked-in PDF and main notebook use a
sixteen-second window. The exact final paper plotting state is not retained.
The old plotting-only notebook's saved curves also differ from Figure 5.

The new notebook uses the same model, data-generation parameters, data seed,
sampling seed 0, nine methods and maximum budgets. It imports
`SVM_oracles_new.py` and `SVM_utils_new.py`. It writes to a fresh
`results_new/paper_<timestamp>/` directory on each run:

- `svm_results_full_new.pkl`: full histories, statuses, final points,
  configuration, environment, data hash and source hashes.
- `plot_SVM_func_new.pdf`: updated objective comparison.

The plotting-only companion requires `DATA_FILE` to name one of these exports.
It writes `plot_SVM_func_replot_new.pdf` alongside that export. There is no
automatic selection of a potentially unrelated latest run.

## Numerical changes

- A failed/exhausted search returns the last accepted iterate and an explicit
  status. A failed Cholesky solve retries with increased regularization; a
  failed cubic subproblem is not consumed as a successful zero step.
- Exactly stationary starting points stop successfully. A numerically unchanged
  trial returns `stagnation`, not convergence. Nonfinite trials are rejected.
- GLAd-SSN now uses the paper constants alpha=0.5 and beta=0.25 instead of the
  notebook's 0.1 and 0.1. LeAP-SSN retains its original alpha=beta=0.1.
- The default `accept_tol` is zero, so neither SSN bypasses the acceptance
  inequalities. Passing a positive value explicitly restores the original
  absolute relaxation for diagnostic comparisons only. It is not used in the
  new experiment.
- Original starting points are retained: entries 0.5 for the deterministic
  methods and zero for SVRG-BB. This difference is explicit in the configuration;
  a full-size check found that switching SVRG-BB to 0.5 made it unstable. No BB
  stepsize clipping or alternate algorithm was introduced to conceal that result.
  All methods use absolute gradient tolerance 1e-5.
  Previously LeAP-SSN used 1e-8, GLAd-SSN used 1e-5, and the other methods had
  no gradient stopping test. Budget exhaustion is still reported separately.
- SVRG-BB checks zero displacement, nonpositive/nonfinite curvature and invalid
  stepsizes before division. It reports `stagnation`, `invalid_bb_curvature`
  or `invalid_bb_stepsize`; it does not silently change the BB formula or
  insert a different optimization algorithm. Uniform sampling, inner-loop
  length and initial stepsize 1e-12 are retained. Its history records the
  actual outgoing stepsize, with NaN when no outgoing epoch is performed.
- The objective, gradient, sample-gradient scaling and active-set Hessian are
  unchanged. The intercept is unregularized and unconstrained. Hessian reuse
  remains every m accepted outer steps; factorization is still performed for
  each regularization trial. GLAd-SSN keeps p=0.5, initial Lambda=1, factor-4
  backtracking and division by 4 after acceptance. LeAP-SSN retains factor 2.
- Newton subproblems still use direct Cholesky solves in floating-point
  arithmetic. Cubic Newton retains its scalar root solver and tolerance 1e-8;
  the zero-gradient division is fixed. These are numerical solves, not exact
  arithmetic or a new iterative/inexact subproblem method.

These changes can alter curves, stopping points and runtime. The updated figure
is a corrected experiment, not a bit-for-bit reproduction of the old figure.
Strict acceptance can expose stagnation at floating-point resolution; inspect
the printed statuses and final gradient norms instead of treating all endpoints
as successful convergence.

## API, diagnostics and plotting

- `trace=False` works without allocating histories. Optional B/Binv metrics
  work in the corrected cubic/SSN paths. Wrapper initialization uses `self.x`,
  and each wrapper returns `(point, status, history)`.
- Hessian-event histories are appended only when a Hessian is computed.
  Per-sample gradient calls are counted separately from full gradients.
  `matrix_inverses` is retained for compatibility and means successful linear
  solves, not actual matrix inversions. Cubic trial counts are not counts of
  its internal factorizations.
- Redundant objective/gradient evaluations were removed. Terminal counters and
  elapsed time include unsuccessful searches. Timing uses `time.perf_counter`
  and includes diagnostic tracing, but excludes data generation and plotting.
  The notebook explicitly limits BLAS to one thread for all methods and records
  the loaded BLAS implementation. Old timings cannot be compared directly.
- Iteration panels use accepted outer iterates, including x0. For SVRG-BB the
  corresponding point is an epoch snapshot; the axis explicitly says so.
  Trial histories remain available, and explicitly requested trial plots are
  labelled as trials including rejected steps. All methods appear in the
  accepted-iterate plot; its original 0--65 display window is retained.
- Objective gaps are relative to the best finite observed value, not a certified
  optimum. Log plots stop at the first threshold crossing without assuming
  monotonicity. The display floor is the larger of the requested threshold and
  `10 * eps_float64 * max(1, abs(reference))`. Nonfinite tails are excluded. Extreme finite divergent histories use bounded
  log tick exponents so Matplotlib cannot overflow while generating tick labels.
- The two-panel paper plot now follows the original SVM/MF paper legend:
  three columns, white box with alpha 0.8, overlapping the panel tops. The
  plot cells reload the plotting module so this style can be applied without
  rerunning the solvers. This is a presentation-only change.
- Plot functions honor `filename` and do not save implicitly. Runtime is not
  clipped by default; `clip_time=5` explicitly requests the old time window.

## Validation and use

Open the new experiment notebook in a Python kernel with NumPy, SciPy,
scikit-learn, Matplotlib and threadpoolctl. It supports execution from the
repository root or `SVM/`. Set `SMOKE_TEST=True` for a 200-sample, 12-feature
check with reduced budgets. That mode is not a paper reproduction.

The focused regression command actually used on this machine was:

```powershell
& 'C:/Users/cleme/anaconda3/python.exe' -B SVM/test_SVM_new.py
```

Use the equivalent interpreter path for your environment. These tests cover
derivatives, preserved valid legacy trajectories, failed/rejected steps,
factorization retries, stationary starts, BB denominator failures, optional
metrics, tracing, counters, wrapper dimensions and plot semantics. They do not
establish the paper's theoretical guarantees or benchmark runtimes.

Validation on November 5, 2026 passed all 15 regression tests. With the legacy
acceptance settings restored, the corrected solvers exactly matched the recorded
small baseline's final vectors and objectives. Both smoke notebooks ran in fresh
kernels. The final full-size experiment also executed every cell successfully:
SVRG-BB converged in 43 epochs, cubic Newton in 28 iterations, LeAP-SSN in 33,
and GLAd-SSN with m=1,2,4,5,10 in 17,24,24,27,33 iterations, respectively.
All those methods reached the absolute gradient tolerance 1e-5. Gradient descent
exhausted its original 400-step budget with gradient norm approximately 1.065.
The best observed objective was 32923455.946951862. This is one run with the
corrected settings, not a reproduction of the old timing figure or a multi-seed
performance study; other user experiments were running on the same machine.

See `validation_new.json` for the environment, original-file hashes, small
baseline and final run statistics. The full-size export and checked plot are
preserved separately under
`results_new/validation_paper_20261105_230304_338042/`. Executed notebook copies
remain in the validation temporary directory recorded in that JSON. The
delivered notebooks have clean outputs and retain their normal experiment defaults.
