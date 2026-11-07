# Skip the Hessian, Keep the Rates

Code for **[Skip the Hessian, Keep the Rates: Globalized Semismooth Newton with Lazy Hessian Updates](https://arxiv.org/abs/2602.08069)** (NeurIPS 2026).
This release contains the three standalone experiments for the November 2026 camera-ready version, Section 5 and Appendix E.

**Expected runtime with default settings:** approximately **1–2 hours for NMF** (six parallel workers), **4–5 hours for NN** (including fresh Hessians, `m=1`), and **3–5 minutes for SVM**. These estimates are based on recorded CPU runs; hardware, thread counts and settings affect runtime.

## Installation and use

Use Python 3.12. From the repository or extracted supplementary-code folder:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open a notebook below, restart its kernel, and select **Run All**. Run it from the repository root or its experiment folder. Each notebook generates its own synthetic data, runs the methods, and produces plots. No saved results or preparation script are required.

| Experiment | Notebook |
| --- | --- |
| Non-negative matrix factorization | [Parallel NMF experiment](MF/experiment_matrix_factorization_multiple_parallel_final.ipynb) |
| Neural network with a Lipschitz penalty | [NN experiment](NN/experiment_nn_lipschitz_constraint_new.ipynb) |
| Support vector classification | [SVM experiment](SVM/experiment_SVM_new.ipynb) |

The settings cells specify data sizes, seeds, method parameters and iteration budgets. Generated results are saved separately and excluded from Git.

The experiments use CPU computation and double precision; a GPU is not required. The NMF cubic Newton baseline uses GALAHAD GLRT, included in the dependencies. Subproblems use numerical tolerances, and wall-clock timings depend on hardware and thread settings.

Please cite the linked paper when using this code. See [THIRD_PARTY.txt](THIRD_PARTY.txt) for attribution.
