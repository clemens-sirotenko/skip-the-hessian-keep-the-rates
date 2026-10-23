import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

from oracles_mat_fac import *

from scipy import sparse


import inspect
from scipy.sparse.linalg import LinearOperator, cg
from scipy.optimize import linear_sum_assignment

try:
    from scipy.sparse.linalg import LinearOperator, cg

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# from scipy.sparse import diags

import time


def cg_compat(Aop, b, maxiter, rtol, callback=None):
    sig = inspect.signature(cg)
    kwargs = dict(maxiter=maxiter, callback=callback)
    if "tol" in sig.parameters:  # older SciPy
        kwargs["tol"] = rtol
    else:  # newer SciPy
        kwargs["rtol"] = rtol
        kwargs["atol"] = 0.0
    return cg(Aop, b, **kwargs)


def vanilla_newton(
    oracle,
    p_0,
    n_iters=100,
    trace=True,
    eps=1e-8,
    f_star=None,
    grad_tol=None,
    warnings=False,
):
    """
    Run Vanilla Newton Method
    for 'n_iters' iterations, minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    p_k = np.copy(p_0)
    f_k = oracle.func(p_k)
    g_k = oracle.grad(p_k)

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""

    # Main loop
    for k in range(n_iters + 1):
        if trace:
            history["func"].append(f_k)
            history["grad_calls"].append(oracle.grad_calls)
            history["matrix_inverses"].append(matrix_inverses)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())

        if (f_star is not None and f_k - f_star < eps) or (grad_tol is not None):
            status = "success, %d iters" % k
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        Hess_k = oracle.hess(p_k)

        try:
            # Compute the regularized Newton step
            global delta_p
            # delta_x = scipy.linalg.solve(Hess_k, -g_k)
            delta_p = sparse.linalg.spsolve(Hess_k, -1 * g_k)
            matrix_inverses += 1
        except (np.linalg.LinAlgError, ValueError) as e:
            raise RuntimeError("Cholesky solve failed") from e
            if warnings:
                print("W: linalg_error", flush=True)
        f_new = oracle.func(p_k + delta_p)
        g_new = oracle.grad(p_k + delta_p)

        # Update the point
        p_k += delta_p
        f_k = f_new
        g_k = g_new

    return p_k, status, history


# ============================================================
# Alignment for "distance to ground truth" (perm + scaling)
# ============================================================
def _normalize_cols(A):
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)


def align_factors(U, V, U_true, V_true):
    """Align columns of U and matching rows of V, preserving U @ V."""
    r = U_true.shape[1]
    Un, Vn = _normalize_cols(U), _normalize_cols(V.T)
    Utn, Vtn = _normalize_cols(U_true), _normalize_cols(V_true.T)

    # Match each column of U together with the corresponding row of V.
    C = np.abs(Utn.T @ Un) + np.abs(Vtn.T @ Vn)  # (r,r)
    row, col = linear_sum_assignment(-C)
    perm = col

    U_p = U[:, perm]
    V_p = V[perm, :]

    # per-component scaling to best match U_true
    U_al = np.zeros_like(U_p)
    V_al = np.zeros_like(V_p)
    for k in range(r):
        u = U_p[:, k]
        v = V_p[k, :]
        ut = U_true[:, k]
        dk = (ut @ u) / ((u @ u) + 1e-12)
        if abs(dk) < 1e-12:
            dk = 1.0
        U_al[:, k] = dk * u
        V_al[k, :] = (1.0 / dk) * v

    return U_al, V_al


# ============================================================
# Metrics to log each iteration
# ============================================================
def mf_terms_and_metrics(
    oracle: PositiveMatrixFactorization, z, M_clean=None, U_true=None, V_true=None
):
    U, V = oracle.unpack(z)
    M_obs = oracle.M

    R = U @ V - M_obs
    rec = 0.5 * np.sum(R * R)
    pen = 0.5 * oracle.beta * (
        np.sum(np.maximum(-U, 0.0) ** 2) + np.sum(np.maximum(-V, 0.0) ** 2)
    )

    rel_obs = np.linalg.norm(U @ V - M_obs) / (np.linalg.norm(M_obs) + 1e-12)

    rel_clean = None
    if M_clean is not None:
        rel_clean = np.linalg.norm(U @ V - M_clean) / (
            np.linalg.norm(M_clean) + 1e-12
        )

    negU = np.maximum(-U, 0.0).mean()
    negV = np.maximum(-V, 0.0).mean()

    U_err = None
    V_err = None
    if U_true is not None and V_true is not None:
        U_al, V_al = align_factors(U, V, U_true, V_true)
        U_err = np.linalg.norm(U_al - U_true) / (np.linalg.norm(U_true) + 1e-12)
        V_err = np.linalg.norm(V_al - V_true) / (np.linalg.norm(V_true) + 1e-12)

    return dict(
        rec=rec,
        pen=pen,
        rel_obs=rel_obs,
        rel_clean=rel_clean if rel_clean is not None else np.nan,
        negU=negU,
        negV=negV,
        U_err=U_err if U_err is not None else np.nan,
        V_err=V_err if V_err is not None else np.nan,
    )


# ============================================================
# Leap-SSN GN, logs time + metrics
# ============================================================
def leap_ssn_gn(
    oracle,
    p_0,
    n_iters=100,
    trace=True,
    eps=1e-8,
    f_star=None,
    grad_tol=None,
    warnings=False,
    alpha=0.5,
    beta=0.25,
    adaptive_search_max_iter=60,
    lambda0=1.0,
    cg_maxiters=200,
    cg_tol=1e-10,
    use_grad_reg=True,
    metric_fn=None,
    lazy_Hessians=False,
    update_freq=5,
):
    oracle = OracleCallsCounter(oracle)
    start = time.perf_counter()

    p_k = np.copy(p_0).astype(np.float64)
    f_k = float(oracle.func(p_k))
    g_k = oracle.grad(p_k)

    history = defaultdict(list) if trace else None
    status = ""
    lam_base = float(lambda0)

    for k in range(n_iters + 1):
        if trace:
            history["iter"].append(k)
            history["func"].append(f_k)
            history["grad_norm"].append(float(np.linalg.norm(g_k)))
            history["time"].append(float(time.perf_counter() - start))
            history["func_calls"].append(oracle.func_calls)
            history["grad_calls"].append(oracle.grad_calls)
            history["hess_calls"].append(oracle.hess_calls)

            if metric_fn is not None:
                md = metric_fn(p_k)
                for kk, vv in md.items():
                    history[kk].append(float(vv))

        # stopping
        if (f_star is not None and f_k - f_star < eps) or (
            grad_tol is not None and np.linalg.norm(g_k) <= grad_tol
        ):
            status = f"success, {k} iters"
            break
        if k == n_iters:
            status = "iterations_exceeded"
            break
        if lazy_Hessians is True and k % update_freq != 0:
            pass
            # print("Didn't update Hessian at iteration " + str(k))
        else:
            Hk = oracle.hess(p_k)
        H_mv = Hk if callable(Hk) else (lambda v: Hk @ v)

        gnorm = np.linalg.norm(g_k)
        g_scale = (gnorm**0.5) if use_grad_reg else 1.0

        accepted = False
        chosen_lam = None
        chosen_cg = None

        for j in range(adaptive_search_max_iter + 1):
            if j == adaptive_search_max_iter:
                if warnings:
                    print(("W: adaptive_iterations_exceeded, k = %d" % k), flush=True)
                break
            lam = (4.0**j) * lam_base * g_scale

            def A_mv(v):
                return H_mv(v) + lam * v

            Aop = LinearOperator((p_k.size, p_k.size), matvec=A_mv, dtype=np.float64)

            it_counter = {"it": 0}

            def _cb(_xk):
                it_counter["it"] += 1

            try:
                d, info = cg_compat(
                    Aop, -g_k, maxiter=cg_maxiters, rtol=cg_tol, callback=_cb
                )
            except Exception as e:
                if warnings:
                    print("W: CG exception:", e)
                continue

            if not np.all(np.isfinite(d)):
                continue

            p_new = p_k + d
            f_new = float(oracle.func(p_new))
            g_new = oracle.grad(p_new)

            history["func_at_lssolve"].append(f_k)
            history["grad_norm_at_lssolve"].append(float(np.linalg.norm(g_new)))

            step_vec = p_k - p_new  # = -d
            Fprime_term = float(g_new @ step_vec)
            gnew_norm2 = float(g_new @ g_new)

            cond1 = Fprime_term >= (alpha / lam) * gnew_norm2
            cond2 = (f_k - f_new) >= beta * lam * float(d @ d)

            if np.isfinite(f_new) and cond1 and cond2:
                accepted = True
                chosen_lam = lam
                chosen_cg = it_counter["it"]
                p_k, f_k, g_k = p_new, f_new, g_new
                print("First here at accepted at iter ", k, " and inner iter", j)
                break

        if trace:
            history["accepted"].append(bool(accepted))
            history["chosen_lam"].append(chosen_lam if accepted else np.nan)
            history["cg_iters"].append(chosen_cg if accepted else np.nan)

        if accepted:
            # lam_base = lam_base / 4.0
            lam_base = lam / 4.0
            print("Second here at accepted at iter ", k, " and inner iter", j)
        else:
            # lam_base = lam_base * 4.0
            print("HERE!!!")
            lam_base = lam_base * 4.0 ** (j + 1)

    return p_k, status, history


# ============================================================
# Adam baseline (NumPy), logs time + metrics
# ============================================================
def adam_optimize(
    oracle,
    p_0,
    n_iters=100,
    trace=True,
    lr=1e-2,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    grad_tol=None,
    metric_fn=None,
):
    oracle = OracleCallsCounter(oracle)
    start = time.perf_counter()

    p = np.copy(p_0).astype(np.float64)
    m1 = np.zeros_like(p)
    m2 = np.zeros_like(p)

    f = float(oracle.func(p))
    g = oracle.grad(p)

    history = defaultdict(list) if trace else None
    status = ""

    for k in range(n_iters + 1):
        if trace:
            history["iter"].append(k)
            history["func"].append(f)
            history["grad_norm"].append(float(np.linalg.norm(g)))
            history["time"].append(float(time.perf_counter() - start))
            history["func_calls"].append(oracle.func_calls)
            history["grad_calls"].append(oracle.grad_calls)

            if metric_fn is not None:
                md = metric_fn(p)
                for kk, vv in md.items():
                    history[kk].append(float(vv))

        if grad_tol is not None and np.linalg.norm(g) <= grad_tol:
            status = f"success, {k} iters"
            break
        if k == n_iters:
            status = "iterations_exceeded"
            break

        # Adam step
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * (g * g)
        m1hat = m1 / (1 - b1 ** (k + 1))
        m2hat = m2 / (1 - b2 ** (k + 1))
        p = p - lr * m1hat / (np.sqrt(m2hat) + eps)

        f = float(oracle.func(p))
        g = oracle.grad(p)

    return p, status, history


# ============================================================
# Plot helper (no length mismatch: use each history's own x)
# ============================================================
def plot_metric(hist_leap, hist_adam, key, title, ylog=True):
    x1 = np.asarray(hist_leap["iter"])
    y1 = np.asarray(hist_leap[key], dtype=float)
    x2 = np.asarray(hist_adam["iter"])
    y2 = np.asarray(hist_adam[key], dtype=float)

    plt.figure()
    plt.plot(x1, y1, label="LeapSSN-GN-CG")
    plt.plot(x2, y2, label="Adam")
    if ylog:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

    t1 = np.asarray(hist_leap["time"], dtype=float)
    t2 = np.asarray(hist_adam["time"], dtype=float)

    plt.figure()
    plt.plot(t1, y1, label="LeapSSN-GN-CG")
    plt.plot(t2, y2, label="Adam")
    if ylog:
        plt.yscale("log")
    plt.title(title + " (vs time)")
    plt.xlabel("seconds")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()


def plot_metric_many(hists, labels, key, title, ylog=True):
    # --- vs iteration ---
    plt.figure()
    for hist, lab in zip(hists, labels):
        if key not in hist:
            continue
        x = np.asarray(hist.get("iter", np.arange(len(hist[key]))))
        y = np.asarray(hist[key], dtype=float)
        n = min(len(x), len(y))
        plt.plot(x[:n], y[:n], label=lab)
    if ylog:
        plt.yscale("log")
    plt.title(title + " (vs iteration)")
    plt.xlabel("iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

    # --- vs time ---
    plt.figure()
    for hist, lab in zip(hists, labels):
        if key not in hist or "time" not in hist:
            continue
        t = np.asarray(hist["time"], dtype=float)
        y = np.asarray(hist[key], dtype=float)
        n = min(len(t), len(y))
        plt.plot(t[:n], y[:n], label=lab)
    if ylog:
        plt.yscale("log")
    plt.title(title + " (vs time)")
    plt.xlabel("seconds")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()


def gd_armijo(
    oracle,
    z0,
    n_iters=60,
    trace=True,
    grad_tol=None,
    armijo_c=1e-4,
    backtrack=0.5,
    step0=1.0,
    min_step=1e-16,
    metric_fn=None,
):
    """
    Gradient descent with Armijo backtracking.

    History keys:
      - iter, time, f, grad_norm
      - plus keys produced by metric_fn(z) (e.g. rec, pen, rel_obs, rel_clean, ...)
    """
    z = z0.copy().astype(np.float64)
    f = float(oracle.func(z))
    g = oracle.grad(z)

    hist = defaultdict(list) if trace else None
    t0 = time.perf_counter()
    status = ""

    for k in range(n_iters + 1):
        if trace:
            hist["iter"].append(k)
            hist["time"].append(time.perf_counter() - t0)
            hist["f"].append(f)
            hist["grad_norm"].append(float(np.linalg.norm(g)))

            if metric_fn is not None:
                md = metric_fn(z)
                for kk, vv in md.items():
                    hist[kk].append(float(vv))

        if grad_tol is not None and np.linalg.norm(g) <= grad_tol:
            status = f"success, {k} iters"
            break
        if k == n_iters:
            status = "iterations_exceeded"
            break

        d = -g
        g2 = float(g @ g)

        # backtracking
        t = step0
        accepted = False
        while t >= min_step:
            z_new = z + t * d
            f_new = float(oracle.func(z_new))
            if f_new <= f - armijo_c * t * g2:
                accepted = True
                break
            t *= backtrack

        if not accepted:
            status = "line_search_failed"
            break

        z = z_new
        f = f_new
        g = oracle.grad(z)

    return z, status, hist


def nag_armijo(
    oracle,
    z0,
    n_iters=200,
    trace=True,
    grad_tol=None,
    armijo_c=1e-4,
    backtrack=0.5,
    step0=1.0,
    min_step=1e-16,
    metric_fn=None,
    restart=True,
):
    """
    Nesterov accelerated gradient with Armijo backtracking + (optional) adaptive restart.

    History keys: iter, time, func, grad_norm, plus metric_fn(z) keys.
    """
    z = z0.copy().astype(np.float64)
    y = z.copy()
    tk = 1.0

    fz = float(oracle.func(z))
    hist = defaultdict(list) if trace else None
    t_start = time.perf_counter()
    status = ""

    for k in range(n_iters + 1):
        # log at current iterate z
        if trace:
            hist["iter"].append(k)
            hist["time"].append(time.perf_counter() - t_start)
            hist["func"].append(fz)
            # (use grad at y for algorithm, but log norm at z for consistency)
            gz = oracle.grad(z)
            hist["grad_norm"].append(float(np.linalg.norm(gz)))
            if metric_fn is not None:
                md = metric_fn(z)
                for kk, vv in md.items():
                    hist[kk].append(float(vv))

        # stopping
        if grad_tol is not None:
            if np.linalg.norm(oracle.grad(z)) <= grad_tol:
                status = f"success, {k} iters"
                break
        if k == n_iters:
            status = "iterations_exceeded"
            break

        # gradient at lookahead point y
        gy = oracle.grad(y)
        gy2 = float(gy @ gy)
        fy = float(oracle.func(y))

        # Armijo backtracking on y
        step = step0
        accepted = False
        while step >= min_step:
            z_new = y - step * gy
            f_new = float(oracle.func(z_new))
            if f_new <= fy - armijo_c * step * gy2:
                accepted = True
                break
            step *= backtrack

        if not accepted:
            status = "line_search_failed"
            break

        # Nesterov momentum update
        tk_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        beta = (tk - 1.0) / tk_new
        y_new = z_new + beta * (z_new - z)

        # Restart to stabilize nonconvex case:
        # if objective increased compared to previous iterate, kill momentum
        if restart and f_new > fz:
            tk_new = 1.0
            y_new = z_new.copy()

        z, y, tk = z_new, y_new, tk_new
        fz = f_new

    return z, status, hist
