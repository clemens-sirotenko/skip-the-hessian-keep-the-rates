# Opt-in NN corrections

Use `experiment_nn_lipschitz_constraint_new.ipynb`, which imports
`utils_new.py` and `GLAdSSNOptimizer_new.py`. The original modules,
experiment notebook, plotting notebook, saved outputs and PDFs are retained.
To return to the original behavior, open
`experiment_nn_lipschitz_constraint.ipynb` and restart its kernel. Old optimizer
states should not be loaded into the corrected optimizer.

## Paper reference and reproduction

The requested reference is the local current manuscript
`Lazy_Hessians_Neurips-20.pdf` (PDF metadata 2026-10-22), Algorithm 1,
equation (26), Section 5.1 and Appendix E.1/Table 2. The public reference is
[arXiv:2602.08069v1](https://arxiv.org/abs/2602.08069v1), where the NN objective
is equation (27) and the experiment settings are older. The local PDF stays
ignored; the formulas and settings below make these notes usable without it.
Inline manuscript discussions are not treated as adopted algorithm changes.

The scalar network is `2-16-16-1` with tanh activations and 337 parameters.
Training inputs are 100 uniform samples from `[0, 2*pi]^2`; targets are
`sin(x1)*cos(x2) + .2*sin(2*x1+x2) + .1*x1*x2` plus Gaussian noise with
standard deviation .05. The corrected objective is

```
mean((prediction - target)**2)
    + penalty_lambda/2 * mean(relu(squared_norm_estimate_i - 1)**2).
```

The full-run reference follows local Table 2: penalty lambda 1, five power
iterations, Lambda_0=1, p=.5, lazy periods `{1,5,100,1000,2500,5000}`, and
900,000 full-batch Adam updates at learning rate 1e-4. Its scheduler retains
factor .5, patience 5, threshold 1e-5 and minimum rate 1e-6, driven by the
pre-update loss. The 13,000 GLAd update budget is retained from the original
notebook. The manuscript's Adam prose still says 10,000 updates and 7e-4;
we follow its Table 2 rather than silently treating that disagreement as a
code error. The manuscript itself is not edited.

The notebook now defaults to a small precision comparison:
`ADAM_EPOCHS=10_000`, `GLAD_EPOCHS=300`, `K=5`, lazy periods `{1,5,20,50}`,
`DTYPE=torch.float64`, and `SOLVE_RTOL=1e-10`. The corresponding previous
float32 run took 23.9 minutes; 25-35 minutes is an estimate for the float64
configuration, not a measured full run. Restore 900,000 and 13,000 updates
and the six periods above for the historical full budgets. The recent
1,000-update runs used K=10, so their objective differs from this K=5
comparison; they do not isolate the precision change.

The training seed remains 26. Training/test samples, targets and the first
network's parameter snapshot are generated in float32, matching the preceding
experiments, then cast to the selected training precision. Merely changing the
default dtype before random generation would change both data and weights,
even with the same seed. All methods load the same snapshot. A separate
generator with seed 27 produces one shared 1,000-sample test set. Full-batch
order, CPU, one PyTorch thread and existing plot colors are retained. All
training and metric evaluation use float64 in the current configuration.
The notebook runs from a fresh kernel in the repository root or `NN`.

| Local figure | Notebook plotting input | Display window |
| --- | --- | --- |
| 1: gradient norm | `grad_norm` versus `time` and accepted-update index | Full histories; horizontal axes logarithmic above 1, linear near zero |
| 2: input-Hessian norm estimate | `hessian_norm`, the maximum of the per-sample estimates | Full histories; nonnegative linear norm axis and reference at 1 |

Index zero is the initial point. The gradient axis retains a symlog region
below 1e-12 to permit exact zeros. New data, metadata and PDFs go to timestamped
`NN/corrected_results/` directories. Save flags can disable exports. The old
plotting notebook still targets the old data. A table and export report first
observed crossings of gradient targets 1e-2, 1e-3 and 1e-4, with times and
accepted-update counts. These are reporting targets, not stopping criteria;
crossings may be transient and unmet targets are explicitly marked.

Exports also retain final model weights, per-attempt GLAd damping, relative
linear residual, trial count and cumulative solve/Hessian counts, plus the
learning rate used for each accepted Adam update. No additional derivative
evaluations are performed for these scalar diagnostics. Final model weights
are not complete optimizer checkpoints: resuming from them with a newly
constructed optimizer resets its state.

The 100-second runtime axis in Figure 1 is a display window, not a run
budget. The original notebook explicitly sets `set_xlim(0, 100)`; its
saved outputs report 3,550.07 seconds for fresh GLAd and 737.05 seconds
for period 5 over 13,000 updates. The local paper reports Apple M1 Pro
hardware, while the corrected-code timings were measured on an Intel
i7-11800H. The corrected derivative calculation also does more work than
the original detached power iteration, so historical runtimes do not
predict the corrected implementation's runtime.

## Numerical changes

- **Penalty:** normalize power vectors per sample, average the sample
  penalties, and apply lambda/2. Previously the code used one norm across
  the whole batch and a different coefficient. The second helper return
  remains scalar, but now means maximum sampled norm estimate.
- **Derivatives:** use `sqrt(sum(w_i**2) + 1e-24)` for normalization and
  differentiate through all power iterations. Squared norms enter the
  penalty directly to avoid unnecessary square-root derivatives at zero.
  The compatibility option `detach_power_iter=True` explicitly requests
  approximate derivatives; the new notebook never enables it.
- **Regularization:** with `lambda = 4**j * Lambda_k * ||g||**.5`, update
  `Lambda_{k+1} = 4**j * Lambda_k / 4`, excluding the gradient factor.
  The legacy state key `lambda_k` now consistently means Lambda; actual
  accepted damping is recorded separately as `accepted_lam`.
- **Numerical safeguards:** the previously unused `lam_min=1e-8` now floors
  the initial trial damping. The effective base coefficient is raised to
  this floor before backtracking. Set `lam_min=0` for the unfloored formula.
  Both direct and least-squares solves must satisfy
  `||(H+lambda*I)s-g|| <= solve_rtol*||g||`; the default is 100 machine eps
  (about 1.2e-5 for float32). Inaccurate solves retry with stronger damping.
  These are numerical approximations to stationarity, not exact solves or
  certified minimizers of an indefinite subproblem. No PSD projection is added.
  The optimizer's defaults are unchanged; the current notebook explicitly
  overrides them with `lam_min=1e-9` and `solve_rtol=1e-10` in float64.
- **Stopping and failures:** `grad_tol=0` handles exact stationarity before
  division by lambda, without adding a positive convergence threshold.
  A failed search, nonfinite computation or stagnation returns an explicit
  state status and retains the current parameters. The notebook stops and
  reports it. Only accepted steps advance the outer and Hessian-reuse
  counters. The parameter `m` remains a backtracking cap (25 in the notebook,
  allowing 26 trials); `number_of_steps_to_freeze` is the paper's lazy period.
- **Logging:** loss, gradient and norm are measured at the same returned
  iterate, including the initial point. The optimizer's returned f/g are
  now also at that point. CPU `perf_counter` timings include diagnostic
  evaluation and exclude model construction. Adding aligned diagnostics
  changes timing overhead, particularly for Adam.

Finite power iteration estimates spectral radius (largest absolute eigenvalue
for a symmetric Hessian), not necessarily its largest signed eigenvalue. It
can underestimate the true norm or miss a direction. Neither a finite set
of training inputs nor these estimates certifies a global Lipschitz bound.
The local manuscript's signed-eigenvalue wording remains a separate paper
correction. Author/dependency attribution remains in
[THIRD_PARTY.txt](../THIRD_PARTY.txt).

## Validation, 2026-11-03

Original sources were snapshotted and hashed before editing the copies.
The preceding audit established counterexamples for the original code:
next coefficient .5 instead of .25; two-sample penalty about 9 instead of
2.25; and a detached derivative 1.164289 instead of 1.411753. Numerical
equivalence to the original is deliberately not expected for these fixes.

Twelve focused tests pass: analytic penalties (including negative curvature),
batch splitting/duplication, finite-difference first and second derivatives,
zero-Hessian derivatives, coefficient updates, zero-gradient stopping,
rejected-search counters, lazy refreshes, consistent/inconsistent singular
systems, damping floors, and nonfinite gradients. Run from the repository root:

```powershell
python -m unittest discover -s NN -p test_nn_corrections.py -v
```

The verified local interpreter was `C:/Users/cleme/anaconda3/python.exe`
(Python 3.12.14, PyTorch 2.7.1+cu126, Windows 11, Intel Family 6 Model 141).
Tests use float64 and one library thread. Derivative checks use finite
differences of size 1e-6 and absolute tolerances 1e-6/1e-5; analytic penalty
checks use 11 decimal places.

A fresh-kernel execution from the repository root used the full 100 inputs
and original network, float32, and three updates for each of seven methods.
All completed their budgets; initial metrics agreed exactly across methods,
and each final recorded metric agreed exactly with recomputation at its
returned model. Fresh GLAd evaluated three Hessians; each lazy variant
evaluated one. Their final objective/gradient/max-norm triples were
(.808089, .513593, .148914) for fresh GLAd and
(1.115283, 2.219567, .114622) for the lazy variants; Adam gave
(2.260131, 5.113986, .054957). The longer lazy periods necessarily agree
over these first three steps. No timing comparison is claimed.

The export path was also exercised in a separate temporary directory: the
pickle reloads with seven histories and metadata, and both new PDF names
are produced. A forced nonfinite Adam update restores its preceding model
and reports failure with metrics still aligned to that model.

The original files remained byte-for-byte unchanged. Source snapshots,
hashes and the executed smoke notebook are retained in a local temporary
directory named `nn-corrections-20261103-*`. This is a correctness check,
not full paper reproduction; no corrected full-run curves are supplied.

## Small float64 configuration, 2026-11-05

Before changing the notebook, its executed float32 version, these notes and
the numerical modules were copied with SHA-256 hashes to
`corrected_results/notebook_snapshots/20261105_163836_811357/`. Existing result
directories are unchanged. Old cell outputs are cleared in the edited notebook
so they cannot be mistaken for new float64 results.

The numerical reference for the restored small budget is
`corrected_results/20261103_152357_680941/nn_results_new.pkl`: seed 26,
float32, K=5, 300 GLAd and 10,000 Adam updates, periods `{1,5,20,50}`.
Its total training time was 1,431.6 seconds, including diagnostics. Fresh GLAd
finished at objective 0.0078127803 and gradient norm 0.0025470434; Adam
finished at objective 0.0115917176 and gradient norm 0.0062292619. The environment
was Python 3.12.14, PyTorch 2.7.1+cu126 on CPU, one thread, Intel i7-11800H.
Timing is machine/load dependent and should not be used as a numerical
equivalence check.

The precision experiment changes the configured precision and solve tolerance,
apart from diagnostic/export and plotting additions. It retains the recent
run's lam_min=1e-9; the older 300-update baseline used 1e-8. Both floors are
far below the actual damping in that saved baseline. It retains lambda0=1,
p=.5, the two acceptance constants, the raw symmetrized parameter Hessian,
grad_tol=0, 25 maximum
backtracking increases, deterministic power vectors, the seeds and shared
initialization. Adam retains learning rate 1e-4 and its original scheduler
called after every update using the pre-update loss. No PSD projection,
adaptive Hessian schedule or Adam tuning is folded into this comparison.

For a paper comparison, distinguish objective minimization and stationarity
from test-set regression error. Compare common gradient targets and objective
values at common wall-clock budgets, give Adam a stated learning-rate/scheduler
tuning budget, and use paired seeds with variability reported. Keep K and the
penalty definition identical across methods. The earlier K=5 Adam results
were substantially better than its K=10 results; this sensitivity warrants
investigation before making claims across configurations. The small run is
not evidence of asymptotic superlinear convergence or a global Lipschitz bound.

Validation for this update: the 12 existing numerical tests pass. A fresh
Jupyter kernel executed all notebook cells from the repository root with
three updates per method (five methods), preserving the full dataset and
network. All 15 updates were accepted, and recomputed final metrics matched
the recorded histories exactly. Setup checks confirmed exact equality of
training/test data and initial weights to the saved float32 baseline after
promotion to float64. All methods had identical initial metrics. The solve
residuals met 1e-10, and the expected fresh/lazy Hessian counts were verified.
The pickle export reloads with float64 model weights, diagnostics and correct
target entries, including initial hits, later hits and unmet targets (using
test-only reporting thresholds). Imports/setup also passed from `NN`.
Plot layouts were inspected using the complete historical K=5 histories,
without presenting them as new float64 results. Smoke artifacts are retained
in the local temporary directory `nn-small-float64-check-w5kwbxza`.
The optimizer and penalty modules are byte-for-byte unchanged. The full
300/10,000-update float64 comparison has not been run as part of this check.

## Selective GLAd runs, 2026-11-05

The GLAd training cell now has a single list, `GLAD_PERIODS=[1,5,20,50]`.
Period 1 is no longer trained by a separate mandatory call. Setting the list
to `[10]` trains only period 10, using the same shared initial weights and
`GLAD_EPOCHS`; it does not continue the parameters of a previous method.

In an initialized kernel, edit and execute only the cell under **Train
selected GLAd periods**, followed by the evaluation/export/plot cells.
Existing histories, model weights and run diagnostics remain in memory;
only a selected method with the same label is replaced. The selection cell
also registers models from the previous notebook's `model_adam`, `model_glad`
and `models_lazy` variables, so an open pre-change kernel can be reused.
Do not rerun the data/setup cell when retaining existing results: that cell
intentionally initializes empty result dictionaries for a new experiment.

For a fresh kernel, set `RUN_ADAM=False` in the separate Adam training cell
and use `GLAD_PERIODS=[10]`, then Run All. Export, plotting and comparison
cells support the absence of Adam and period 1. Exports distinguish the
most recently requested periods from all actually stored periods. Colors
for existing methods are retained and additional periods do not depend on
a fixed number of plot styles. The solver, objective, precision, seeds,
stopping rules and numerical trainer functions are unchanged.

Keep data, initialization, precision, K and the penalty coefficient fixed
when accumulating methods in one comparison. Changing the optimization
problem calls for a separate experiment, not mixing old and new curves.
The preceding notebook and notes are archived under
`corrected_results/notebook_snapshots/20261105_172453_435173/`.

Validation: two fresh-kernel smoke scenarios used two updates per method.
The first trained Adam and periods `{1,5,20,50}`, then simulated the old
kernel variables and ran the selection cell with `[10]`. Only period 10
was called; every previous history, model weight, diagnostic and data tensor
was unchanged. The second executed the notebook with `RUN_ADAM=False` and
`GLAD_PERIODS=[10]`; neither Adam nor period 1 was called. Both scenarios
completed plotting, comparison tables and pickle export, with actual period
metadata and final model metrics verified. The Adam and GLAd trainer
definitions are unchanged. Artifacts are retained in the local temporary
directory `nn-period-selection-check-cn5yn0xh`; no full training run was
performed for this control-flow change.

## Audit of the penalty-5 plateau, 2026-11-05

The saved run `corrected_results/20261105_174744_637383/nn_results_new.pkl`
uses the corrected notebook with float64, K=5, penalty coefficient 5,
1,000 GLAd updates at periods 10 and 50, and 10,000 Adam updates. The data,
initialization and seeds remain those described above. Its histories are
the plotting inputs for the notebook's gradient and sampled-norm figures.
This is a small experimental variant of Section 5.1 / Appendix E.1, not
the full paper reproduction: both the budgets and penalty coefficient
differ from the reference. No optimizer, objective or notebook settings
were changed by this audit.

Independent checks used the three saved final models and all 100 training
inputs. A separate closed-form chain rule for the two tanh layers computes
each 2-by-2 input Hessian. Explicit matrix-vector products then reproduce
the finite-power objective, including differentiation through normalization.
Against the production nested-autograd implementation:

- Input Hessians agree to within 2.4e-16 relative Frobenius error.
- Full parameter gradients agree to within 3.6e-9 relative error, and the
  337-by-337 parameter Hessians to within 3.5e-12. Hessian asymmetry is below
  1e-15 relative error. Recomputed final loss and gradient norms exactly
  match the exported histories.
- Central differences of the production gradient, in a random direction,
  the gradient direction and the largest-eigenvalue direction, verify
  Hessian-vector products with best relative errors below 4.9e-8 over
  step sizes 1e-3 through 1e-8. The stiff GLAd endpoints require very small
  differences; this is not a claim of uniform finite-difference accuracy.
- All recorded accepted solve residuals meet 1e-10: maxima are 3.11e-12
  for period 10 and 3.50e-13 for period 50. Refresh counts match the requested
  schedules, and no accepted step increases the objective.

A fresh-process replay of all 1,000 period-50 updates reproduces every
saved loss/gradient norm and the final weights exactly. All accepted
regularized systems in this replay are positive definite, even though
the raw parameter Hessians are indefinite. The last cached Hessian agrees
with the independent derivative at its actual refresh point to 1.33e-12
relative error. On update 1,000, the damping-24.082 trial has relative
residual 6.33e-14 and sufficient objective decrease, but fails the gradient
acceptance inequality. Damping 96.328 passes both conditions, with residual
2.91e-14 and step norm 8.57e-5. Thus this rejection is not caused by an
inaccurate linear solve. The existing 12 numerical tests also pass.

The finite-power surrogate itself has extreme curvature at these endpoints:

| Endpoint | Largest parameter-Hessian eigenvalue, K=5 | Same weights, exact sampled norm objective | K=5 maximum input norm | Exact sampled maximum |
| --- | ---: | ---: | ---: | ---: |
| Adam | 235.14 | 124.30 | 1.01282 | 1.25249 |
| GLAd period 10 | 664323.20 | 216.50 | 1.03036 | 1.21414 |
| GLAd period 50 | 93363.21 | 223.15 | 1.02996 | 1.15325 |

For period 10, training sample index 50 contributes about 664284 to the
second directional derivative along the largest-curvature direction.
Its input-Hessian eigenvalues are approximately -1.12210 and 0.10524.
The fixed normalized all-ones starting vector has only 0.000151 alignment
with the dominant eigenvector. Five power iterations report norm 1.000021
at this sample. Small parameter changes rotating that eigenvector can
therefore change the penalty sharply. Period 50 exhibits the same effect.
The implementation accurately differentiates this poorly conditioned
surrogate; it does not accurately approximate the spectral norm everywhere.

Fresh Hessians at the saved endpoints permit some lower-damping steps, but
still reject sufficiently large steps because of the gradient condition.
Both surrogate curvature and reuse matter; the observations do not prove
a permanent gradient floor. Tightening the solves is not supported by this
audit. An appropriate next controlled experiment is exact sampled spectral
norms from the 2-by-2 input Hessians, applied identically to both optimizers.
That changes the implemented objective and requires separate results and
paper documentation; it is not a guarantee that GLAd outperforms Adam.
Exact norms on 100 inputs also do not certify a global Lipschitz bound.

Audit scripts and JSON reports are retained locally under
`corrected_results/diagnostics/20261105_hessian_audit/`: `audit.py`,
`curvature.py`, and `replay.py`. They use Python 3.12.14, PyTorch 2.7.1+cu126,
CPU float64 and one thread. Run each with the scientific Python environment
from the repository root. They read the named saved pickle and write only
diagnostic files in their own directory. Files prefixed `manual_mse_` retain
an earlier algebraically equivalent, manually reduced MSE check; only the
unprefixed replay uses the notebook's identical `F.mse_loss` operation and
reproduces its trajectory exactly. The replay omits routine plotting and
per-step norm measurement, so its runtime is not a training benchmark.

## Separate exact spectral-norm notebook, 2026-11-05

`experiment_nn_lipschitz_constraint_exact.ipynb` uses
`exact_spectral_norm.py` and the existing `GLAdSSNOptimizer_new.py`.
It evaluates the per-sample spectral-norm penalty in local equation (26),
Section 5.1 (public arXiv:2602.08069v1 equation (27)) directly, replacing
Appendix E.1's finite-power surrogate. This is an exploratory experiment,
not a claim to reproduce the existing paper figures or to satisfy every
assumption of the convergence theorems. The original and corrected power
notebooks, their numerical helpers and their result files are unchanged.

The copied defaults are the source notebook's actual current settings:
penalty coefficient 5, 1,000 GLAd updates, 10,000 Adam updates, selected
periods `[10, 50]`, CPU float64, one thread and relative solve tolerance
1e-10. These differ from the paper's penalty coefficient 1 and full budgets.
Changing the coefficient to 1 is a separate controlled comparison. Seeds
26/27, float32 generation followed by float64 promotion, the common initial
weights, data, architecture, Adam scheduler, Hessian reuse, raw symmetrized
parameter Hessian, damping, acceptance inequalities, backtracking budget,
stopping criteria and linear-solve safeguards are inherited unchanged.

For each of the 100 independent input rows, two differentiations construct
the 2-by-2 input Hessian. Symmetric eigendecomposition gives its eigenvalues;
their largest square is the squared spectral norm. Full parameter derivatives
through the active penalty are retained away from flagged ties; see the
continuation policy below. Neither a starting vector nor K is
used. The scalar maximum returned by the helper is a detached diagnostic.
The optimizer still constructs and reuses a separate 337-by-337 parameter
Hessian, and its linear solves remain numerical approximations to stationarity.

Inactive penalties have zero derivatives and are excluded from the
differentiable eigendecomposition. This avoids artificial NaNs in second
derivatives at zero or repeated eigenvalues inside the unit ball. At the
threshold the zero second-order branch is selected. When a penalty is active,
an absolute gap between squared eigenvalues <= 64 times machine epsilon
times max(1, squared spectral norm) identifies an unresolved active tie.
The initial implementation raised `ActiveSpectralTieError` there; the
continuation policy below now defaults to averaged derivatives instead.
Strict rejection remains available with `tie_policy="raise"`. No spectral
smoothing is silently added. An
active tie can make the exact objective non-C1; Appendix E.1's surrogate
semismooth-gradient argument does not automatically extend to this objective.

Start the new notebook in a fresh kernel. `RUN_ADAM=False` skips Adam;
`GLAD_PERIODS=[10]` trains only period 10; include 1 explicitly for a fresh
parameter Hessian every step. In the initialized exact-norm experiment,
rerunning only the period-selection cell preserves other stored methods.
Keep the objective and initialization fixed when accumulating results.
The old-kernel migration logic is intentionally not copied: finite-power
histories must not be mixed into the exact-objective comparison.

Exports go to `corrected_results/exact_spectral_norm/<timestamp>/`, using
`nn_results_exact.pkl` and `plot_NN_exact_*.pdf`. Metadata identifies the
exact objective, eigenvalue calculation, tie guard and lack of K. Figure 1
uses `grad_norm` against `time` and accepted-update index; Figure 2 uses
`hessian_norm`, the exact maximum on the training inputs up to floating-point
error. Neither is evidence of a certified global Lipschitz bound.

Validation used Python 3.12.14, PyTorch 2.7.1+cu126, CPU float64 and one
thread. Six focused tests check negative dominant eigenvalues, averaging,
batch duplication, an independent analytic nonlinear tanh Hessian and its
parameter gradient/Hessian, finite differences, inactive repeated/zero
eigenvalues, affine networks and explicit rejection of active ties. Run:

```powershell
python -m unittest discover -s NN -p test_nn_exact_spectral_norm.py -v
```

At the saved period-10 endpoint of the penalty-5 baseline, the complete
parameter Hessian agrees with an independent chain-rule implementation to
2.12e-16 relative error and its gradient to 7.26e-16. Three accepted Newton
updates starting from those active-penalty weights decreased the exact
objective; this was a validation continuation, not a new comparison result.
A fresh Jupyter kernel from the repository root executed 20 updates each
for Adam and periods 1, 5, 10 and 50, followed by adding period 20. A second
fresh kernel from `NN` executed two updates for period 10 with Adam disabled.
All updates were accepted. Tests verified shared data/initialization against
the saved baseline, final metrics, refresh counts, solve residuals, selective
preservation, metadata, pickle exports, target tables and plot generation.
The full configured experiment was not run, and no full-run timing or
performance claim follows from these checks. Validation artifacts are in
`corrected_results/diagnostics/exact_implementation_20261105_192541/`.

### Continuing past spectral ties, 2026-11-05

At the user's request, `DEFAULT_TIE_POLICY="average"` now permits the exact
notebook to continue past `ActiveSpectralTieError`. At a flagged active tie
in a symmetric 2-by-2 input Hessian B, the forward squared norm remains
q=max_j eigenvalue_j(B)^2, but differentiation uses
q_bar=||B||_F^2/2, the average of the two squared eigenvalues. Implementing
this as `q.detach() + q_bar - q_bar.detach()` (with the zero-valued difference
parenthesized) retains the exact spectral value. No sample or penalty term
is omitted. At an exact tie the first derivative selects an averaged
subgradient; the second-order information, and derivatives at numerically
unresolved near ties, are explicit approximations. In particular, this does
not establish a classical Hessian or the paper's convergence hypotheses at
those points. Repeated eigenvalues require no eigenvector differentiation,
so this fallback avoids NaNs from that operation. Outside flagged ties the
previous values, gradients and Hessians are unchanged.

`tie_policy="raise"` on a helper call, or setting the module's
`DEFAULT_TIE_POLICY="raise"`, restores strict rejection. The notebook exports
the policy and describes plotted `grad_norm` values as selected derivative
norms at ties. The damping, acceptance tests, nonfinite checks and linear
residual checks are still active; allowing ties does not guarantee that all
subsequent optimizer steps will be accepted.

For an already running kernel, reload `NN.exact_spectral_norm` and rebind
`exact_hessian_norm_penalty` as shown in the notebook introduction. Then
rerun only the failed training method; select only its period if other GLAd
methods completed. Reloading does not clear completed histories, models or
data, but an interrupted method starts again from the shared initialization.

Validation: the eight helper tests pass, including exact/near ties with both
equal and opposite signs, exact forward values, finite first/second
derivatives, strict mode, and identical derivatives away from ties. A ninth
test checks three accepted GLAd updates starting at an active spectral tie.
A fresh kernel also completed three updates each for Adam and periods 10/50,
including exports, plots, tie-policy metadata and a reload that preserved
the result registries. The user's exact failing live iterate was not saved
in the notebook, so validation used constructed ties and the fresh run.
Snapshots and validation artifacts are retained in
`corrected_results/diagnostics/exact_tie_fix_20261105_195150/`.

## Configurable normalization and Huber spectral smoothing, 2026-11-05

This change distinguishes two different epsilons. The paper's implementation
in Appendix E.1 uses a finite number of regularized power iterations. The
corrected power notebook `experiment_nn_lipschitz_constraint_new.ipynb` now
exposes `POWER_EPS=1e-12`, used in `sqrt(||w||^2 + POWER_EPS^2)` for every
normalization, including the starting vector. This default preserves the
previous numerical expression `sqrt(||w||^2 + 1e-24)`. Adam, GLAd and reported
metrics use the same configured value; exports and method records include it.
The notebook's existing budgets (15,000 Adam / 2,000 GLAd), coefficient 1,
K=5 and selected periods `[50, 100]` were retained. Larger epsilon changes
the surrogate and can increase norm underestimation; it is not merely a
linear-solver tolerance.

The separate `experiment_nn_lipschitz_constraint_exact.ipynb` now defaults
to `SPECTRAL_SMOOTHING_EPS=1e-3` and uses a Huber upper approximation of the
maximum of the two squared eigenvalues. Set this parameter to `0.0` to
recover the previous unsmoothed exact objective and its tie policy. With a
positive epsilon, `exact_spectral_norm.py` computes the differentiable
2-by-2 input Hessian B, forms A=B^2, and uses

```
r^2 = (A00 - A11)^2 + 4*A01^2
h_epsilon(r) = r^2/(2*epsilon) + epsilon/2   if r <= epsilon
               r                          otherwise
q_epsilon(B) = (trace(A) + h_epsilon(r))/2
penalty = coefficient/(2*N) * sum_i relu(q_epsilon(B_i) - 1)^2
```

Here r is the gap between the squared eigenvalues. This shifted Huber
absolute value agrees with the absolute value outside the transition band.
Consequently `||B||_2^2 <= q_epsilon(B) <= ||B||_2^2 + epsilon/4`, with exact
agreement outside that band. At epsilon=1e-3 the maximum squared-norm bias
is 2.5e-4. Inside the band, the implementation uses r^2 directly and never
differentiates a square root at zero. The otherwise unused square-root
branch also has a positive argument, preventing masked NaNs in autograd.
No eigenvector derivatives or averaged tie derivatives enter this penalty.

For fixed positive epsilon the matrix-entry expression is C1, generally
not C2 at r=epsilon, with a locally strongly semismooth gradient. The tanh
network composition and outer squared hinge retain a C1 objective with a
locally semismooth gradient. At transition boundaries autograd selects a
piecewise second-order branch. This addresses the regularity distinction
that matters for semismooth Newton; it does not establish the paper's other
global assumptions or positive semidefiniteness of the parameter Hessian.
The raw spectral maximum at an active tie need not be C1. Merely adding an
epsilon under a final square root would not remove that maximum's kink.

This is an explicit alternative to the finite-power construction in local
equation (26), Section 5.1 / Appendix E.1; the public reference is
[arXiv:2602.08069v1](https://arxiv.org/abs/2602.08069v1), equation (27).
It is not a reproduction of the existing manuscript plots. No change was
made to Hessian reuse, raw parameter curvature, damping, acceptance tests,
linear solves, tolerances, stopping criteria, seeds or initialization.
The direct notebook retains its existing coefficient 5, 10,000 Adam /
1,000 GLAd budgets and periods `[10, 50]`. Both methods use the same Huber
objective. The plotted `hessian_norm` remains the exact maximum on sampled
training inputs, not the biased surrogate or a global bound.

Use a fresh kernel for a new epsilon. Review rejects stored methods with
different epsilon, smoothing type or recorded penalty coefficient.
Positive-epsilon exports are separate:
`corrected_results/huber_spectral_norm/<timestamp>/nn_results_huber.pkl`,
with `plot_NN_huber_grad_norm.pdf` and `plot_NN_huber_hess_op_norm.pdf`.
Figure 1 inputs remain `grad_norm` versus `time` and accepted-update index;
Figure 2 uses `hessian_norm`. Zero epsilon retains the `exact_spectral_norm`
directory and exact filenames. Power exports retain `nn_results_new.pkl`.
Metadata records the objective, epsilon, upper bias and derivative policy.
Old outputs are preserved; the changed direct notebook's stale cell outputs
were cleared, and a snapshot of its previous contents was retained.

Validation used Python 3.12.14 / PyTorch 2.7.1+cu126, CPU float64, one thread,
Intel Core i7-11800H. All 26 tests pass:

```powershell
python -B -m unittest discover -s NN -p "test_nn*.py" -v
```

New checks cover the approximation bound, exact agreement outside the band,
finite first and second derivatives at both same-sign and opposite-sign
ties, finite differences, zero Hessians, and a continuous gradient with a
Hessian jump at the Huber boundary. A pre-existing singular least-squares
test was made deterministic with an SVD driver in that test alone: the
Windows default QR driver intermittently reported rank zero for its rank-one
fixture. The production optimizer and its residual safeguards are unchanged.

At saved power endpoints `20261105_174744_637383` (period 10, coefficient 5)
and `20261105_201559_033905` (period 50, coefficient 1), default-epsilon loss,
gradient and full parameter Hessian are bitwise identical to the archived
helper. An endpoint-only epsilon sweep found that 1e-6 through 1e-3 did not
reduce the very large curvature. At 1e-2 it dropped from approximately
664323 to 243 and from 891364 to 234, respectively, but the objective also
changed. This is not evidence of improved training speed.

At the saved direct period-100 endpoint `20261105_201629_341056`, coefficient
1, Huber epsilon=1e-3 reduced the largest parameter-Hessian eigenvalue from
2.87e11 to 285.86. Two 20-update diagnostic continuations started at those
same weights, each with a fresh optimizer, period 5 and initial Lambda=1.
The unsmoothed objective changed only at rounding level. The Huber objective
decreased from 0.0074160565 to 0.0071023701, and its final accepted damping
was 0.175 rather than roughly 4.64e10. Its final gradient norm was 0.00206,
so this check does not establish convergence or superiority to Adam. These
continuations omit normal notebook diagnostics from their timing and are
not runtime benchmarks. No full experiment was run.

Snapshots, scripts and reports are under
`corrected_results/diagnostics/epsilon_smoothing_20261105_210424/`.
`huber_results.json` contains the Huber diagnostics. The separately named
square-root prototype report predates the requested Huber change and must
not be used as Huber results. `validate_notebooks.py` runs disposable copies
with three updates per method, separate validation exports and fresh kernels;
the saved notebook budgets are not reduced by this check.
All three fresh-kernel checks passed: power and Huber modes each ran Adam
and periods 1/5 from the repository root; unsmoothed mode ran only period 10
from `NN`. Checks covered final metrics, unchanged samples and initial weights,
Hessian refresh counts, solve residuals, mixed-epsilon rejection, pickle
metadata and both PDF plots. Executed validation notebooks are retained
alongside the reports.

### Reloading updated helper signatures in existing kernels, 2026-11-05

Both notebooks now explicitly reload their penalty helper module and rebind
the imported function in the first code cell. A normal Python import could
retain the pre-epsilon function in a running kernel, producing an unexpected
`normalization_eps` or `spectral_smoothing_eps` keyword error. Each notebook's
introduction also provides a recovery cell that only reloads and rebinds the
helper, preserving data, settings, models and histories. Changing the objective
still requires a separate comparison as documented above.

This is an import repair only: the numerical implementation, settings, paper
alignment and figure inputs described above are unchanged. Existing outputs
were preserved. Validation reproduced each TypeError by loading the archived
pre-epsilon helper into the module cache, then executed the repaired notebook
in a fresh kernel with two updates each for Adam and GLAd periods 1/5. Both
passed, including the recovery cell preserving registry identities and exactly
matching before/after loss, gradient and sampled-norm metrics. Only disposable
copies used smaller budgets; no full experiments were run. Python/PyTorch,
CPU float64 and one-thread settings match the preceding validation. Snapshots
and executed checks are in
`corrected_results/diagnostics/import_reload_fix_20261105_212758/`.

### Central experiment settings, 2026-11-05

The first code cell in both corrected notebooks now owns `RUN_ADAM`,
`GLAD_PERIODS`, training sample count `N`, penalty coefficient and the relevant
epsilon; the power notebook also defines `no_of_power_iterations` there.
Budgets, learning rate, seeds, precision, stopping/solve tolerances and export
switches remain in that cell. Later data and training cells consume these
values without overwriting them. Rerunning the configuration cell preserves
existing data and result registries; rerunning the data/model setup cell still
starts a new comparison. The notebook instructions explain selective addition
of methods under this arrangement.

The power notebook retains the user's current settings: periods `[50, 100]`,
coefficient 1, K=5, `POWER_EPS=1e-7`, 15,000 Adam / 2,000 GLAd updates and
Adam learning rate 1e-5. The direct notebook retains the user's first-cell
choices `[20, 50, 200, 500]`, coefficient 1, Huber epsilon 0.05, 20,000 Adam /
4,000 GLAd updates and learning rate 5e-5. Its later training cell previously
overwrote the chosen periods with `[10, 50]`; removing that duplicate is an
intentional selection correction. Existing saved results are not relabeled.
Numerical routines, paper alignment, figure inputs and solver rules described
above are unchanged. Current executable configuration and export metadata
remain authoritative when the user adjusts these settings.

Fresh-kernel checks used disposable two-update budgets with Adam disabled,
period 5 followed by adding period 10, for both notebooks. Samples, initial
weights, initial metrics, two-update GLAd metrics and final weights match the
pre-edit baseline exactly. Checks also confirm that rerunning the first cell
preserves data/results, selection is not overwritten, and completed methods
survive adding a period. Saved notebook outputs were preserved; no full run
was performed. Environment matches the preceding checks. Snapshots, baseline
and executed notebooks are in
`corrected_results/diagnostics/central_config_20261105_221021/`.
