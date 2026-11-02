"""Other MF baselines, retained for the older experiment notebooks.

The parallel notebook uses methods.py. These implementations were moved here
without numerical changes. See METHODS_AUDIT.md for the prior correctness audit
and CUBIC_SUBPROBLEM.md for the legacy dense cubic reference and its cost.
Framework attribution remains in ../THIRD_PARTY.txt.
"""

from collections import defaultdict
from datetime import datetime

import numpy as np
import scipy.linalg

from oracles import OracleCallsCounter
from methods import _cubic_newton
from cubic_subproblems import cubic_newton_step_nonconvex_new

__all__ = [
    "vanilla_newton", "super_newton", "fast_gradient_method",
    "cubic_newton", "cubic_newton_step",
]


def _validate_grad_tol(grad_tol):
    """Accept an optional finite, nonnegative absolute gradient tolerance."""
    if grad_tol is not None and (
        isinstance(grad_tol, (bool, np.bool_))
        or not isinstance(grad_tol, (int, float, np.integer, np.floating))
        or not np.isfinite(grad_tol)
        or grad_tol < 0
    ):
        raise ValueError("grad_tol must be a finite nonnegative real scalar, or None")


def vanilla_newton(oracle, x_0, n_iters=1000,
                   adaptive_search=True, trace=True, B=None, Binv=None, eps=1e-8,
                   f_star=None, grad_tol=None, warnings=False):
    """Take unregularized Newton steps using explicit Hessians.

    x_0 and gradients have shape (d,); Hessians have shape (d, d).
    grad_tol bounds the Euclidean gradient norm. There is no line search;
    adaptive_search, B and Binv are retained only for call compatibility.
    Failed solves and nonfinite trials return the last accepted point.
    """
    _validate_grad_tol(grad_tol)
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Initialization of the method
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""

    # Main loop
    for k in range(n_iters + 1):

        if trace:
            history['func'].append(f_k)
            history['grad_calls'].append(oracle.grad_calls)
            history['matrix_inverses'].append(matrix_inverses)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())

        if (f_star is not None and f_k - f_star < eps) or \
                (grad_tol is not None and np.linalg.norm(g_k) <= grad_tol):
            status = "success, %d iters" % k
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        Hess_k = oracle.hess(x_k)

        try:
            delta_x = scipy.linalg.solve(Hess_k, -g_k)
            matrix_inverses += 1
        except (np.linalg.LinAlgError, ValueError):
            if warnings:
                print('W: linalg_error', flush=True)
            status = "linear_solve_failed, %d iters" % k
            break
        candidate = x_k + delta_x
        if not np.all(np.isfinite(candidate)):
            status = "nonfinite_trial, %d iters" % k
            break
        f_new = oracle.func(candidate)
        g_new = oracle.grad(candidate)
        if not (np.isfinite(f_new) and np.all(np.isfinite(g_new))):
            status = "nonfinite_trial, %d iters" % k
            break
        if history is not None:
            history["func_at_lssolve"].append(f_new)
        # Update the point
        x_k += delta_x
        f_k = f_new
        g_k = g_new

    if history is not None:
        history['time'][-1] = (datetime.now() - start_timestamp).total_seconds()
        history['grad_calls'][-1] = oracle.grad_calls
        history['matrix_inverses'][-1] = matrix_inverses
    return x_k, status, history


def super_newton(oracle, x_0, n_iters=1000, H_0=1.0, alpha=1.0,
                 adaptive_search=True, trace=True, B=None, Binv=None, eps=1e-8,
                 H_min=1e-5, f_star=None, grad_tol=None, warnings=False, lazy_hessians=False, lazy_hessians_freq=5):
    """Dense Super-Universal Newton with shift H_k * ||g||_* ** alpha.

    x_0 and gradients have shape (d,); B is an SPD (d, d) metric and
    Binv its inverse. grad_tol bounds the dual gradient norm. Failed
    factorizations retry with larger regularization; a failed search
    preserves the last accepted point. H_min is a legacy unused argument.
    """
    _validate_grad_tol(grad_tol)
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Initialization of the dual norm
    if B is None:
        B = np.eye(x_0.shape[0])
        dual_norm_sqr = lambda x: x.dot(x)
    else:
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)

    # Initialization of the method
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    H_k = H_0

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""

    # Main loop
    no_backtracks = 0
    for k in range(n_iters + 1):

        if trace:
            history['func'].append(f_k)
            history['grad_norm'].append(g_k_norm)
            history['H_k'].append(H_k)
            history['grad_calls'].append(oracle.grad_calls)
            history['matrix_inverses'].append(matrix_inverses)
            history['new'].append(list(oracle.func_at_hess_eval))
            history['hess_calls'].append(oracle.hess_calls)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())

        if (f_star is not None and f_k - f_star < eps) or g_k_norm == 0 or \
                (grad_tol is not None and g_k_norm <= grad_tol):
            status = "success, %d iters" % k
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        if not lazy_hessians or k % lazy_hessians_freq == 0:
            Hess_k = oracle.hess(x_k)

        if history is not None:
            history["func_at_hess_eval"].append(oracle.func(x_k))
        accepted = False
        failure_status = "line_search_failed"
        for _ in range(40):
            lambda_k = H_k * g_k_norm ** alpha
            if not np.isfinite(lambda_k) or lambda_k <= 0:
                failure_status = "invalid_regularization"
                break
            try:
                delta_x = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
                    Hess_k + lambda_k * B, lower=False), -g_k)
                matrix_inverses += 1
            except (np.linalg.LinAlgError, ValueError):
                H_k *= 4
                continue
            candidate = x_k + delta_x
            if not np.all(np.isfinite(candidate)):
                H_k *= 4
                continue
            if np.array_equal(candidate, x_k):
                failure_status = "numerical_stagnation"
                break
            f_new = oracle.func(candidate)
            g_new = oracle.grad(candidate)
            g_new_norm_sqr = dual_norm_sqr(g_new)
            if not (np.isfinite(f_new) and np.all(np.isfinite(g_new))
                    and np.isfinite(g_new_norm_sqr)):
                H_k *= 4
                continue
            if history is not None:
                history["func_at_lssolve"].append(f_new)
            if not adaptive_search:
                accepted = True
                break

            if g_new.dot(-delta_x) >= g_new_norm_sqr / (4 * lambda_k):
                H_k *= 0.25
                accepted = True
                break
            H_k *= 4
            no_backtracks += 1

        if not accepted:
            status = "%s, %d iters" % (failure_status, k)
            if warnings:
                print("W:", status, flush=True)
            break
        # Update the point
        x_k += delta_x
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr ** 0.5
        if history is not None:
            history["reg_parameter"].append(lambda_k)
    if history is not None:
        history['time'][-1] = (datetime.now() - start_timestamp).total_seconds()
        history['grad_calls'][-1] = oracle.grad_calls
        history['hess_calls'][-1] = oracle.hess_calls
        history['matrix_inverses'][-1] = matrix_inverses
    if warnings:
        print("No of backtracks: ", no_backtracks)
    return x_k, status, history


def cubic_newton_step(g, H_mv, H, B=None, eps=1e-8,
                      lss_maxiters=200, lss_tol=1e-6, lss_method="cg"):
    """Compatibility entry point for the dense global cubic reference.

    Model: g.T@s + .5*s.T@A@s + H/3*||s||_B**3; g:(d,), A/B:(d,d).
    B must be SPD. Returns (step, model_change, status). This replaces the
    old scalar-root routine, which could select a nonglobal root for an
    indefinite Hessian. lss_* arguments remain accepted but are unused;
    eps controls the dense reference tolerance. Dense cost is O(d**3).
    For large problems use cubic_newton_nonconvex_new(...,
    subproblem_solver="glrt"). See CGT (2011), Theorem 3.1 / Section 6.1,
    https://doi.org/10.1007/s10107-009-0286-5, and CUBIC_SUBPROBLEM.md.
    """
    return cubic_newton_step_nonconvex_new(g, H_mv, H, B=B, eps=eps)


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


def fast_gradient_method(oracle, x_0, max_iter=1000, L_0=1.0, line_search=True,
                         trace=True, B=None, Binv=None, L_min=1e-5,
                         warnings=False):
    """Accelerated gradient descent with a quadratic upper-model line search.

    x_0 and gradients have shape (d,). B is an SPD (d, d) metric and Binv
    its inverse; either can be supplied alone. Histories describe accepted
    iterates, not the extrapolated points. A failed search keeps the last
    accepted iterate. The iteration budget is the only planned stopping rule.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    if B is None and Binv is not None:
        B = np.linalg.inv(Binv)
    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
        dual_norm_sqr = lambda x: x.dot(x)
        precond = lambda g: g
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
        precond = lambda g: Binv.dot(g)

    # Initialization of the method
    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    grad_k_norm_sqr = dual_norm_sqr(grad_k)
    L_k = L_0
    A_k = 0.0

    history = defaultdict(list) if trace else None
    status = ""

    # Main loop
    for k in range(max_iter + 1):

        if trace:
            history['func'].append(func_k)
            history['grad_sqr_norm'].append(grad_k_norm_sqr)
            history['func_calls'].append(oracle.func_calls)
            history['grad_calls'].append(oracle.grad_calls)
            history['L'].append(L_k)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())

        if k == max_iter:
            status = 'iterations_exceeded'
            break

        accepted = False
        failure_status = "line_search_failed"
        for _ in range(30):
            if not np.isfinite(L_k) or L_k <= 0:
                failure_status = "invalid_step_size"
                break

            # Choose a_k_new from the equation:
            #   L_k * a_k_new ** 2 = A_k + a_k_new
            a_k_new = (1 + (1 + 4 * A_k * L_k) ** 0.5) / (2 * L_k)
            y_k = (a_k_new * v_k + A_k * x_k) / (a_k_new + A_k)
            grad_y_k = oracle.grad(y_k)

            T = y_k - precond(grad_y_k) / L_k
            if not (np.all(np.isfinite(T)) and np.all(np.isfinite(grad_y_k))):
                L_k *= 2
                continue
            if np.array_equal(T, y_k) and np.any(grad_y_k != 0):
                failure_status = "numerical_stagnation"
                break
            func_T = oracle.func(T)
            func_y_k = oracle.func(y_k) if line_search else None
            if np.isfinite(func_T) and (not line_search or
                    func_T <= func_y_k + grad_y_k.dot(T - y_k)
                    + 0.5 * L_k * l2_norm_sqr(T - y_k)):
                grad_T = oracle.grad(T)
                if np.all(np.isfinite(grad_T)):
                    accepted = True
                    if line_search:
                        L_k = max(0.5 * L_k, L_min)
                    break
            L_k *= 2

        if not accepted:
            status = "%s, %d iters" % (failure_status, k)
            if warnings:
                print("W:", status, flush=True)
            break
        v_k = T + A_k / a_k_new * (T - x_k)
        A_k += a_k_new

        x_k = T
        func_k = func_T
        grad_k = grad_T
        grad_k_norm_sqr = dual_norm_sqr(grad_k)

    if history is not None:
        history['time'][-1] = (datetime.now() - start_timestamp).total_seconds()
        history['func_calls'][-1] = oracle.func_calls
        history['grad_calls'][-1] = oracle.grad_calls
    return x_k, status, history
