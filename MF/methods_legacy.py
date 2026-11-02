"""Dense cubic entry point used by the older MF notebooks.

See CUBIC_SUBPROBLEM.md for references and numerical limitations.
Framework attribution remains in ../THIRD_PARTY.txt.
"""
from methods import _cubic_newton


def cubic_newton(oracle, x_0, n_iters=1000, H_0=1.0, adaptive_search=True,
                 trace=True, B=None, eps=1e-8, H_min=1e-5, f_star=None,
                 warnings=True, lss_maxiters=200, lss_tol=1e-9, lss_method="minres"):
    """Legacy API using the checked dense cubic reference and shared outer loop.

    x_0:(d,), optional SPD B:(d,d). Model coefficient is H_k/6 in the B-norm.
    eps controls both the dense subproblem tolerance and, when f_star is
    given, the objective residual stop. adaptive_search=False uses fixed H_k
    without the majorization test; finite successful nonpositive models are
    still required. lss_* keywords are retained but no longer used.

    This corrects ignored failures, rejected updates, stale gradient histories
    and missing indefinite hard cases in the old implementation. Dense
    cost is O(d**3); use cubic_newton_nonconvex_new with GLRT for large MF runs.
    See CUBIC_SUBPROBLEM.md for the reference and changed numerical behavior.
    """
    return _cubic_newton(
        oracle, x_0, n_iters=n_iters, H_0=H_0, trace=trace, H_min=H_min,
        subproblem_tol=eps, subproblem_solver="dense",
        max_backtracks=119 if adaptive_search else 0,
        warnings=warnings, B=B, f_star=f_star, objective_tol=eps,
        adaptive_search=adaptive_search, trace_trials=True,
    )
