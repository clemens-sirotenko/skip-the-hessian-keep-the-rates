# Cubic baseline for the MF notebooks

Run [the final sequential experiment](experiment_matrix_factorization_multiple_final.ipynb)
or [the parallel version](experiment_matrix_factorization_multiple_parallel_final.ipynb).
All MF cubic comparisons now use GALAHAD GLRT.

Install the notebook dependencies in its Python environment, from the repository root:

```sh
python -m pip install numpy scipy matplotlib pandas joblib threadpoolctl -r MF/requirements-glrt.txt
```

Open the notebook from the repository root or `MF`, choose the data settings,
seeds and iteration budgets, then run all cells. Set `SAVE_FIGURES=True` to
export the plots. Check the reported solver exit statuses.

The paper reference is [arXiv:2602.08069v1](https://arxiv.org/abs/2602.08069v1),
Section 5.2 / Appendix E.2. Its MF figure inputs are:

| Figure | Recorded quantity |
| --- | --- |
| 3: nonnegativity violation | `penalty` |
| 4: gradient norm | `grad_norm` |
| 6: objective residual | `func` minus the plotting reference |

Current notebook defaults are exploratory. Match the paper's data, seeds,
budgets and aggregation before treating a run as figure reproduction; the
additional cubic and representative-run plots are supplementary comparisons.

The cubic model change is `g @ s + s @ A @ s / 2 + H_k * ||s||**3 / 6`.
GLRT uses Hessian-vector products and approximately solves this model.
Accept when the model change is nonpositive and `f(x+s) <= f(x)+model_change`;
rejection doubles `H_k`, acceptance halves it with floor `1e-5`.
`cubic_subproblem_tol` controls the model-gradient residual, while `grad_tol`
(if set) controls outer stopping. A successful inexact solve does not certify
a global model minimum. Parallel runtimes include shared CPU load.

References: [Cartis, Gould & Toint (2011), Section 6.2](https://doi.org/10.1007/s10107-009-0286-5)
for the subproblem method; [Doikov & Nesterov (2021), Algorithm 1](https://doi.org/10.1007/s10957-021-01838-7)
for the outer acceptance rule. Credits: [THIRD_PARTY.txt](../THIRD_PARTY.txt).
