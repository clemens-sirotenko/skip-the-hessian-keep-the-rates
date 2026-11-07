# Skip the Hessian, Keep the Rates

Code and saved experiments for **[Skip the Hessian, Keep the Rates: Globalized
Semismooth Newton with Lazy Hessian Updates](https://arxiv.org/abs/2602.08069)**
(NeurIPS 2026). Saved runs follow the November 2026 camera-ready experimental
configuration (Section 5 and Appendix E).

## Setup

Use Python 3.12 and [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).
From the cloned repository:

```bash
python -m pip install numpy scipy matplotlib pandas joblib threadpoolctl scikit-learn torch jupyterlab
python -m pip install -r MF/requirements-glrt.txt
git lfs install
git lfs pull
python prepare_paper_runs.py
python -m jupyterlab
```

GALAHAD is required for the NMF cubic Newton baseline. Git LFS stores the full
NMF results file (about 207 MiB).

## Experiments

| Experiment | Main notebooks | Saved paper run | Figures |
| --- | --- | --- | --- |
| Non-negative matrix factorization | [Parallel experiment](MF/experiment_matrix_factorization_multiple_parallel_final.ipynb) | [MF/paper_run](MF/paper_run/) | 3, 4, 7, 8, 9 |
| Neural network | [Fresh experiment](NN/experiment_nn_lipschitz_constraint_new.ipynb), [saved runs and continuation](NN/experiment_nn_lipschitz_constraint_new_v2.ipynb) | [NN/paper_run](NN/paper_run/) | 1, 2, 6 |
| Support vector classification | [SVM experiment](SVM/experiment_SVM_new.ipynb) | [SVM/paper_run](SVM/paper_run/) | 5 |

The experiment notebooks are unchanged. Each `paper_run` folder contains one
results snapshot, the paper figures and a provenance manifest.

**Quick start:** `prepare_paper_runs.py` copies the saved NN results to the
existing path expected by NN v2, without overwriting different existing data.
Then run NN v2 to load and plot the saved comparison without training.

For NMF, view the included PDFs directly, or run imports and settings, select
`paper_run/results.pkl.gz` in **Reload saved results**, and continue at **Prepare
plots**, skipping initialization and solver cells. The manifest records the paper
plotting selections, which differ from notebook defaults: representative seed 107,
PALM seed 33, and all twelve methods in the 20-seed gradient average.

Fresh runs require the recorded paper budgets: NMF gradient descent used 320,000
iterations (notebook default: 80,000). NN `_new` produces an initial run; its
saved-run manifest specifies final budgets, including 200,000 Adam updates.
New outputs are separate from the reference results.

Experiments use seeded data, float64 arithmetic and finite-tolerance subproblem
solves. Wall-clock runtimes depend on hardware and thread settings.
Please cite the paper when using this code.
See [THIRD_PARTY.txt](THIRD_PARTY.txt) for attribution.
