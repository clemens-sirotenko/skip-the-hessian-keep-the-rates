# Project guidance

## Purpose and layout

This repository contains code and experiments for "Skip the Hessian, Keep the
Rates: Globalized Semismooth Newton with Lazy Hessian Updates" (NeurIPS 2026).
The goal is readable, documented code that faithfully reproduces the paper.

- `MF/`: non-negative matrix factorization experiments and solvers.
- `NN/`: neural networks with Lipschitz constraints and the PyTorch optimizer.
- `SVM/`: support vector classification experiments and solvers.
- `THIRD_PARTY.txt`: sources and attribution for adapted code and dependencies.

## Paper alignment

Before changing an experiment or solver, identify its relationship to the paper's
algorithm, equations, Section 5, and Appendix E. Record which notebook, settings,
and plotting inputs reproduce each relevant figure.

The following ignored PDFs may be available locally as reference material:
- `Lazy_Hessians_Neurips-20.pdf`
- `Skip the Hessian, Keep the Rates_ Globalized Semismooth Newton with Lazy Hessian Updates _ OpenReview.pdf`

Keep these PDFs ignored. Public documentation must be usable without them; link
to the paper at https://arxiv.org/abs/2602.08069 and identify the version used.
If local references are absent, state the limitation instead of assuming their
contents. Distinguish reviewer suggestions from changes adopted in the revision.

## Numerical changes and cleanup

- Preserve numerical behavior during structural or documentation cleanup.
- Separate algorithm corrections from refactoring so their effects are reviewable.
- Flag discrepancies between the implementation and paper; explain any correction
  and its effect on reproduced results rather than silently preserving a bug.
- Explicitly document changes to Hessian reuse, regularization, acceptance tests,
  subproblem solvers, tolerances, stopping criteria, initialization, and seeds.
- Distinguish theoretical assumptions from implementation approximations,
  especially exact versus inexact subproblem solves.
- Work in small increments, preferably one experiment at a time. Check notebook
  imports and callers before moving, renaming, or removing code.
- Preserve author credits, license notices, and third-party attribution.

## Validation and reproducibility

Establish a baseline before refactoring numerical code. Record the environment,
configuration, seeds, and relevant objective, gradient, constraint, and solver
statistics. Compare results with justified numerical tolerances; runtime need
not be identical across machines or runs.

Use focused numerical checks and small end-to-end runs for affected experiments.
Check notebook execution from a fresh kernel. Clearly distinguish a quick check
from full reproduction of the paper's results, and report what was actually run.
Preserve reference outputs; write new experiment results separately.

No standard setup or test commands have been established yet. Inspect the working
environment and validate commands before documenting them. Record dependency
versions, hardware, precision, and timing methodology where they affect results.

## Documentation

Keep `README.md` focused on installation, a quick example, full experiment and
plot reproduction, citation, and practical limitations. Explain mathematical
parameters and array shapes in solver docstrings, and cross-reference verified
paper equations or algorithm steps. Keep long audit findings and task checklists
outside this file; retain only durable project guidance here.
