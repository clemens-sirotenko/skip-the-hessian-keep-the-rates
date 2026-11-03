"""Parallel matrix-factorization solvers and the alternating-gradient baseline.

Public solvers: gradient_method, alternating_gradient_method, mf_leapssn,
mf_gladssn and cubic_newton_nonconvex_new. The cubic baseline uses GALAHAD
through cubic_glrt.py. See CUBIC_SUBPROBLEM.md for notebook instructions.
Framework attribution remains in ../THIRD_PARTY.txt.

Inputs are assumed valid: real vectors, positive step/regularization parameters,
integer iteration budgets, and nonnegative tolerances (or None). SciPy's rtol
interface is used directly. Numerical failures still return the last accepted point.
"""

from collections import defaultdict
from datetime import datetime
from time import perf_counter

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator, cg, minres

from cubic_glrt import GLRTCubicSubproblem
from oracles import OracleCallsCounter


__all__ = [
    "gradient_method", "alternating_gradient_method", "mf_leapssn", "mf_gladssn",
    "cubic_newton_nonconvex_new",
]


def _stable_gradient_norm(g):
    """Keep ordinary norm arithmetic; rescale only on underflow/overflow."""
    norm = np.linalg.norm(g)
    if np.all(np.isfinite(g)) and (not np.isfinite(norm) or (norm == 0 and np.any(g))):
        return scipy.linalg.norm(g, check_finite=False)
    return norm


def mf_leapssn(oracle, x_0, n_iters=1000, H_0=1.0,
                  adaptive_search=True, trace=True, eps=1e-8,
                  H_min=1e-5, f_star=None, grad_tol=None, warnings=False,
                  lazy_Hessians=False, update_freq=5, lss_maxiters=400,
                  lss_tol=1e-3, lss_method="cg", use_gauss_newton=False):
    """LeAP-SSN with an adaptive Hessian shift, initially H_0.

    x_0 and oracle.grad(x) have shape (d,); oracle.hess(x) supplies a (d, d)
    matrix or Hessian-vector callable. Rejection doubles the shift; acceptance
    halves it. H_min is retained for compatibility but is not enforced.
    See _mf_ssn for common stopping, Hessian-reuse and inner-solve controls.
    """
    return _mf_ssn(
        oracle, x_0, n_iters=n_iters, H_0=H_0, p=None, glad=False,
        adaptive_search=adaptive_search, trace=trace, eps=eps, f_star=f_star,
        grad_tol=grad_tol, warnings=warnings, lazy_Hessians=lazy_Hessians,
        update_freq=update_freq, lss_maxiters=lss_maxiters, lss_tol=lss_tol,
        lss_method=lss_method, use_gauss_newton=use_gauss_newton,
    )


def mf_gladssn(oracle, x_0, n_iters=1000, H_0=1.0,
                  adaptive_search=True, trace=True, eps=1e-8,
                  H_min=1e-5, p=0.99, f_star=None, grad_tol=None, warnings=False,
                  lazy_Hessians=False, update_freq=5, lss_maxiters=400, lss_tol=1e-3,
                  lss_method="cg", use_gauss_newton=False):
    """GLAd-SSN with shift H_k * ||grad f(x)||_2**p, initially H_0.

    x_0 and oracle.grad(x) have shape (d,); oracle.hess(x) supplies a (d, d)
    matrix or Hessian-vector callable. H_k starts at H_0 / ||grad f(x_0)||**p
    unless the initial point meets grad_tol. Rejection multiplies H_k by four;
    acceptance divides it by four. H_min is retained but is not enforced.
    See _mf_ssn for common stopping, Hessian-reuse and inner-solve controls.
    """
    return _mf_ssn(
        oracle, x_0, n_iters=n_iters, H_0=H_0, p=p, glad=True,
        adaptive_search=adaptive_search, trace=trace, eps=eps, f_star=f_star,
        grad_tol=grad_tol, warnings=warnings, lazy_Hessians=lazy_Hessians,
        update_freq=update_freq, lss_maxiters=lss_maxiters, lss_tol=lss_tol,
        lss_method=lss_method, use_gauss_newton=use_gauss_newton,
    )


def _mf_ssn(oracle, x_0, *, n_iters, H_0, adaptive_search, trace, eps,
            p, glad, f_star, grad_tol, warnings, lazy_Hessians, update_freq,
            lss_maxiters, lss_tol, lss_method, use_gauss_newton):
    """Shared LeAP/GLAd loop with an explicit choice of regularization policy.

    Stop when ||g|| <= grad_tol (zero if None), or f-f_star < eps if supplied.
    lazy_Hessians reuses curvature for update_freq iterations; use_gauss_newton
    selects oracle.hess_gn. CG/MINRES use lss_tol and lss_maxiters; a finite
    iteration-limited MINRES step may still pass the acceptance test.
    Failed searches keep the last point and include their work in its history.
    """
    linear_solve = {"cg": cg, "minres": minres}[lss_method]
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = g_k.dot(g_k) ** 0.5
    stationarity_tol = grad_tol if grad_tol is not None else 0.0
    H_k = H_0
    if glad and g_k_norm > stationarity_tol:
        H_k = H_0 / (g_k_norm**p)
    update_factor = 4.0 if glad else 2.0

    history = defaultdict(list) if trace else None
    status = "iterations_exceeded"

    no_backtracks = 0
    for k in range(n_iters + 1):
        if warnings and k % 500 == 0:
            print("Iteration ", k)
        if trace:
            history['func'].append(f_k)
            history['grad_norm'].append(g_k_norm)
            history['H_k'].append(H_k)
            history['grad_calls'].append(oracle.grad_calls)
            history['matrix_inverses'].append(0)
            history['hess_calls'].append(oracle.hess_calls)
            history['penalty'].append(oracle.penalty(x_k))
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())

        if (f_star is not None and f_k - f_star < eps) or g_k_norm <= stationarity_tol:
            status = f"success, {k} iters"
            break

        if k == n_iters:
            break
        if not lazy_Hessians or k % update_freq == 0:
            Hess_k = oracle.hess_gn(x_k) if use_gauss_newton else oracle.hess(x_k)
        f_current = oracle.func(x_k)
        H_mv = Hess_k if callable(Hess_k) else (lambda v: Hess_k @ v)

        if history is not None:
            history["func_at_hess_eval"].append(oracle.func(x_k))
        step_accepted = False
        failure_status = "line_search_failed"
        for i in range(120):
            lambda_k = H_k * g_k_norm ** p if glad else H_k
            if not np.isfinite(lambda_k) or lambda_k <= 0:
                failure_status = "invalid_regularization"
                break
            def A_mv(v):
                return H_mv(v) + lambda_k * v
            Aop = LinearOperator((x_k.size, x_k.size), matvec=A_mv, dtype=np.float64)

            try:
                delta_x, info = linear_solve(Aop, -g_k, maxiter=lss_maxiters, rtol=lss_tol)
            except ZeroDivisionError:
                # GLAd can fall back to a direct solve for an explicit Hessian.
                if not glad or callable(Hess_k):
                    H_k *= update_factor
                    continue
                try:
                    regularized_hessian = Hess_k + lambda_k * np.eye(x_k.size)
                    delta_x = scipy.linalg.solve(regularized_hessian, -g_k)
                    info = 0
                    if warnings:
                        print("W: linear-solver breakdown; used a direct solve", flush=True)
                except (np.linalg.LinAlgError, ValueError):
                    H_k *= update_factor
                    continue
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
                if warnings:
                    print("W: linear solve failed:", error)
                H_k *= update_factor
                continue
            if info < 0 or (lss_method == "cg" and info != 0):
                if warnings:
                    print("W: linear solver did not converge, info =", info)
                H_k *= update_factor
                continue
            if not np.all(np.isfinite(delta_x)):
                if warnings:
                    print("W: not all finite delta_x at k =", k, "i =", i)
                H_k *= update_factor
                continue

            # A finite direction can still overflow or be lost to rounding.
            with np.errstate(over="ignore", invalid="ignore"):
                x_trial = x_k + delta_x
            if not np.all(np.isfinite(x_trial)):
                H_k *= update_factor
                continue
            if np.array_equal(x_trial, x_k):
                failure_status = "numerical_stagnation"
                break

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                f_new = oracle.func(x_trial)
                g_new = oracle.grad(x_trial)
                g_new_norm_sqr = g_new.dot(g_new)
            if not (np.isfinite(f_new) and np.all(np.isfinite(g_new))
                    and np.isfinite(g_new_norm_sqr)):
                H_k *= update_factor
                continue
            if history is not None:
                history["func_at_lssolve"].append(f_new)
                history["grad_norm_at_lssolve"].append(g_new_norm_sqr ** 0.5)

            if not adaptive_search:
                step_accepted = True
                break

            sufficient_decrease = f_current - f_new >= 0.25 * lambda_k * np.linalg.norm(delta_x)**2
            gradient_condition = g_new.dot(-delta_x) >= 0.5 * g_new_norm_sqr / lambda_k
            if sufficient_decrease and gradient_condition:
                step_accepted = True
                H_k = H_k * (1.0 / update_factor)
                break
            H_k = H_k * update_factor
            no_backtracks += 1

        if not step_accepted:
            status = "%s, %d iters" % (failure_status, k)
            if warnings:
                print("W:", status, "-- returning the last accepted iterate", flush=True)
            break
        if history is not None:
            history["reg_parameter"].append(lambda_k)

        x_k, f_k, g_k = x_trial, f_new, g_new
        g_k_norm = g_new_norm_sqr ** 0.5
    # Include work from a final failed search without changing its accepted point.
    if history is not None and history["time"]:
        history["time"][-1] = (datetime.now() - start_timestamp).total_seconds()
        history["grad_calls"][-1] = oracle.grad_calls
        history["hess_calls"][-1] = oracle.hess_calls
    if warnings:
        print("No of backtracks: ", no_backtracks)
    return x_k, status, history


def cubic_newton_nonconvex_new(oracle, x_0, n_iters=1000, H_0=1.0,
                              trace=True, H_min=1e-5, grad_tol=None,
                              subproblem_tol=1e-8, glrt_max_iter=400,
                              max_backtracks=120, warnings=False):
    """Cubic Newton baseline using GALAHAD GLRT and fresh generalized Hessians.

    x_0 and gradients have shape (d,); oracle supplies func, grad, penalty
    and hess (a symmetric matrix or v -> A v). The model change is
    g@s + s@A@s/2 + H_k*||s||**3/6. Accept when f(x+s) <= f(x)+model;
    halve H_k after acceptance (floor H_min), double it after rejection.
    The Hessian is fixed within a search; each trial starts a new GLRT solve.
    grad_tol bounds ||g||. subproblem_tol controls the relative model-gradient
    residual; glrt_max_iter caps each inexact solve, not total HVPs.
    Return (x, status, history); failures retain the last accepted point.
    References and limitations: CUBIC_SUBPROBLEM.md.
    """
    x = np.array(x_0, dtype=np.float64, copy=True)

    start = perf_counter()
    oracle = OracleCallsCounter(oracle)
    f = oracle.func(x)
    g = oracle.grad(x)
    g_norm = _stable_gradient_norm(g)
    H_k = float(H_0)
    history = defaultdict(list) if trace else None
    subproblem_calls = 0
    hvp_calls = 0
    glrt_iterations = 0
    last_glrt_status = None
    last_subproblem_residual = np.nan
    status = "iterations_exceeded"

    for k in range(n_iters + 1):
        if trace:
            for key, value in (("func", f), ("grad_norm", g_norm),
                               ("penalty", oracle.penalty(x)), ("H_k", H_k),
                               ("grad_calls", oracle.grad_calls),
                               ("hess_calls", oracle.hess_calls),
                               ("subproblem_calls", subproblem_calls),
                               ("time", perf_counter() - start)):
                history[key].append(value)
            history["hess_vec_calls"].append(hvp_calls)
            history["glrt_iterations"].append(glrt_iterations)
            history["glrt_status"].append(last_glrt_status)
            history["subproblem_relative_residual"].append(last_subproblem_residual)
        if not (np.isfinite(f) and np.all(np.isfinite(g)) and np.isfinite(g_norm)):
            status = f"nonfinite_initial_point, {k} iters"
            break
        if grad_tol is not None and g_norm <= grad_tol:
            status = f"success, {k} iters"
            break
        if k == n_iters:
            break
        if warnings:
            def report(message):
                print(f"Cubic {k + 1}/{n_iters}: {message} "
                      f"[elapsed {perf_counter() - start:.1f}s]", flush=True)

            report(f"starting, f={f:.6e}, ||g||={g_norm:.3e}, H_k={H_k:.3g}")

        hessian = oracle.hess(x)
        # Fixed x and g throughout this search; discard after accepting a step.
        glrt_model = GLRTCubicSubproblem(g, hessian, progress=report if warnings else None)
        accepted = False
        reason = "line_search_failed"
        for trial in range(max_backtracks + 1):
            if not np.isfinite(H_k):
                reason = "invalid_regularization"
                break
            subproblem_calls += 1
            step, model, message = glrt_model.solve(
                0.5 * H_k,
                eps=subproblem_tol,
                max_iter=glrt_max_iter,
            )
            if message != "success":
                reason = "subproblem_" + message
                break  # A larger weight must not conceal a failed GLRT solve.
            elif not (np.all(np.isfinite(step)) and np.isfinite(model)):
                reason = "nonfinite_subproblem"
            elif model > 0:
                reason = "line_search_failed"
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate = x + step
                if np.array_equal(candidate, x):
                    reason = "success" if g_norm == 0 and not np.any(step) else "numerical_stagnation"
                    break
                if np.all(np.isfinite(candidate)):
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        f_new = oracle.func(candidate)
                        g_new = oracle.grad(candidate)
                        g_new_norm = _stable_gradient_norm(g_new)
                    if (np.isfinite(f_new) and np.all(np.isfinite(g_new))
                            and np.isfinite(g_new_norm)
                            and f_new <= f + model):
                        x, f, g, g_norm = candidate, f_new, g_new, g_new_norm
                        H_k = max(H_min, 0.5 * H_k)
                        accepted = True
                        if warnings:
                            report(f"accepted trial {trial + 1}, f={f:.6e}, "
                                   f"||g||={g_norm:.3e}, next H_k={H_k:.3g}")
                        break
                reason = "line_search_failed"
            if warnings:
                report(f"rejected trial {trial + 1}, H_k={H_k:.3g}, {reason}")
            if trial < max_backtracks:
                H_k *= 2.0
        hvp_calls += glrt_model.hvp_calls
        glrt_iterations += glrt_model.iterations
        last_glrt_status = glrt_model.native_status
        last_subproblem_residual = glrt_model.relative_residual
        del glrt_model  # Release workspace before preparing the next point.
        if not accepted:
            status = f"{reason}, {k} iters"
            break

    if trace:
        history["time"][-1] = perf_counter() - start
        history["grad_calls"][-1] = oracle.grad_calls
        history["hess_calls"][-1] = oracle.hess_calls
        history["subproblem_calls"][-1] = subproblem_calls
        history["hess_vec_calls"][-1] = hvp_calls
        history["glrt_iterations"][-1] = glrt_iterations
        history["glrt_status"][-1] = last_glrt_status
        history["subproblem_relative_residual"][-1] = last_subproblem_residual
    if warnings and not (status.startswith("success") or status == "iterations_exceeded"):
        print("W:", status, "-- returning the last accepted iterate", flush=True)
    return x, status, history


def gradient_method(oracle, x_0, max_iter=1000, L_0=1.0, line_search=True,
                    trace=True, B=None, Binv=None, L_min=1e-5, warnings=False,
                    grad_tol=None):
    """Gradient descent with optional backtracking and an SPD metric.

    x_0 and gradients have shape (d,); B and Binv are (d,d), default identity.
    Either the metric or its inverse can be supplied. grad_tol bounds the
    dual norm sqrt(g.T@Binv@g); None uses only the iteration budget.
    Keep the stopping point in history. MF baseline: arXiv:2602.08069v1,
    Section 5.2 / Appendix E.2.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    if B is None and Binv is not None:
        B = np.linalg.inv(Binv)
    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
        dual_norm_sqr = lambda x: x.dot(x)
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
    if Binv is None:
        precond = lambda g: g
    else:
        precond = lambda g: Binv.dot(g)

    # Initialization of the method
    x_k = np.copy(x_0)
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    grad_k_norm_sqr = dual_norm_sqr(grad_k)
    L_k = L_0

    history = defaultdict(list) if trace else None
    status = ""

    # Main loop
    for k in range(max_iter + 1):

        if trace:
            history['func'].append(func_k)
            history['grad_sqr_norm'].append(grad_k_norm_sqr)
            history['grad_norm'].append(grad_k_norm_sqr**0.5)
            history['L'].append(L_k)
            history['func_calls'].append(oracle.func_calls)
            history['grad_calls'].append(oracle.grad_calls)
            history['penalty'].append(oracle.penalty(x_k))
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())

        if grad_tol is not None and grad_k_norm_sqr**0.5 <= grad_tol:
            status = "success, %d iters" % k
            break

        if k == max_iter:
            status = 'iterations_exceeded'
            break

        accepted = False
        failure_status = "line_search_failed"
        for _ in range(30):
            if not np.isfinite(L_k) or L_k <= 0:
                failure_status = "invalid_step_size"
                break
            T = x_k - precond(grad_k) / L_k
            if not np.all(np.isfinite(T)):
                L_k *= 2
                continue
            if np.array_equal(T, x_k) and np.any(grad_k != 0):
                failure_status = "numerical_stagnation"
                break
            func_T = oracle.func(T)
            if np.isfinite(func_T) and (not line_search or
                    func_T <= func_k + grad_k.dot(T - x_k)
                    + 0.5 * L_k * l2_norm_sqr(T - x_k)):
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
        x_k = T
        func_k = func_T
        grad_k = grad_T
        grad_k_norm_sqr = dual_norm_sqr(grad_k)

    if history is not None:
        history['time'][-1] = (datetime.now() - start_timestamp).total_seconds()
        history['func_calls'][-1] = oracle.func_calls
        history['grad_calls'][-1] = oracle.grad_calls
    return x_k, status, history


def alternating_gradient_method(oracle, z_0, max_iter=200, L_u0=None, L_v0=None,
                                line_search=True, L_min=1e-12, trace=True,
                                tol=None, warnings=False):
    """Alternate a U-step and a V-step on oracle.func; tol bounds ||grad f||.

    U:(m,r), V:(r,n), residual U@V-M. Ridge/penalty weights are alpha/2, beta/2,
    as in the revised MF equation (27). L_u0/L_v0 are initial block constants;
    None uses spectral estimates. Each block has its own backtracking search.
    If V fails after U succeeds, keep and record that accepted U update.
    """
    M, alpha, beta = oracle.M, oracle.alpha, oracle.beta
    U, V = oracle.unpack(np.asarray(z_0, dtype=np.float64).copy())

    func_k = oracle.func(oracle.pack(U, V))
    L_u = float(L_u0) if L_u0 is not None else (np.linalg.norm(V, 2) ** 2 + alpha + beta)
    L_v = float(L_v0) if L_v0 is not None else (np.linalg.norm(U, 2) ** 2 + alpha + beta)

    history = defaultdict(list) if trace else None
    start_timestamp = datetime.now()
    status = "iterations_exceeded"

    def record(grad_norm=None):
        if not trace:
            return
        z = oracle.pack(U, V)
        if grad_norm is None:
            grad_norm = np.linalg.norm(oracle.grad(z))
        values = dict(func=func_k, grad_norm=grad_norm, grad_sqr_norm=grad_norm**2,
                      penalty=oracle.penalty(z), L_u=L_u, L_v=L_v,
                      time=(datetime.now() - start_timestamp).total_seconds())
        for key, value in values.items():
            history[key].append(float(value))

    def update_block(block, gradient, L, evaluate):
        for _ in range(30):
            if not np.isfinite(L) or L <= 0:
                break
            trial = block - gradient / L
            delta = trial - block
            value = evaluate(trial)
            if np.all(np.isfinite(trial)) and np.isfinite(value) and (not line_search or
                    value <= func_k + np.sum(gradient * delta) + 0.5 * L * np.sum(delta * delta)):
                if line_search:
                    L = max(L_min, 0.5 * L)
                return trial, value, L, True
            L *= 2.0
        return block, func_k, L, False

    for k in range(max_iter + 1):
        grad_norm = np.linalg.norm(oracle.grad(oracle.pack(U, V)))
        record(grad_norm)
        if tol is not None and grad_norm <= tol:
            status = "success"
            break
        if k == max_iter:
            break

        # U-step with V fixed.
        R = U @ V - M
        gU = R @ V.T + alpha * U + beta * np.minimum(U, 0.0)

        U, func_k, L_u, accepted = update_block(
            U, gU, L_u, lambda trial: oracle.func(oracle.pack(trial, V)))
        if not accepted:
            status = f"line_search_failed_u, {k} iters"
            break

        # V-step using the updated U.
        R = U @ V - M
        gV = U.T @ R + alpha * V + beta * np.minimum(V, 0.0)

        V, func_k, L_v, accepted = update_block(
            V, gV, L_v, lambda trial: oracle.func(oracle.pack(U, trial)))
        if not accepted:
            status = f"line_search_failed_v, {k} iters"
            record()
            break

    if warnings and status.startswith("line_search_failed"):
        print("W:", status, flush=True)
    if history is not None:
        history['time'][-1] = (datetime.now() - start_timestamp).total_seconds()
    return oracle.pack(U, V), status, history
