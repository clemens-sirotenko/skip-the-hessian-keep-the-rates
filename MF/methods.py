import scipy
import numpy as np
from collections import defaultdict
from datetime import datetime
from oracles import OracleCallsCounter
import inspect

# from oracles_sparse import OracleCallsCounter

from scipy import sparse

# from scipy.sparse import diags
import scipy.sparse.linalg
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import LinearOperator, cg


def vanilla_newton(oracle, x_0, n_iters=1000,
                   adaptive_search=True, trace=True, B=None, Binv=None, eps=1e-8,
                   f_star=None, grad_tol=None, warnings=False):
    """
    Run Vanilla Newton Method
    for 'n_iters' iterations, minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.
    """
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
                (grad_tol is not None):
            status = "success, %d iters" % k
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        Hess_k = oracle.hess(x_k)

        try:
            # Compute the regularized Newton step
            delta_x = scipy.linalg.solve(Hess_k, -g_k)
            matrix_inverses += 1
        except (np.linalg.LinAlgError, ValueError) as e:
            if warnings:
                print('W: linalg_error', flush=True)
            raise RuntimeError("Cholesky solve failed") from e
        f_new = oracle.func(x_k + delta_x)
        g_new = oracle.grad(x_k + delta_x)
        history["func_at_lssolve"].append(f_new)
        # Update the point
        x_k += delta_x
        f_k = f_new
        g_k = g_new

    return x_k, status, history

def cg_compat(Aop, b, maxiter, rtol, callback=None, warnings=True):
    sig = inspect.signature(cg)
    kwargs = dict(maxiter=maxiter, callback=callback)
    if "tol" in sig.parameters:      # older SciPy
        kwargs["tol"] = rtol
    else:                            # newer SciPy
        kwargs["rtol"] = rtol
        kwargs["atol"] = 0.0
    x, info = cg(Aop, b, **kwargs)
    if warnings and info > 0:
        print("W: CG failed at CG iteration =", info, ". Tolerance was set as ", rtol)
    return x, info

def minres_compat(Aop, b, maxiter, rtol, callback=None, warnings=True):
    """
    Solve A x = b using MINRES in a CG-like interface.
    
    Parameters
    ----------
    Aop : array-like or LinearOperator
        The matrix or linear operator representing the Hessian.
    b : ndarray
        Right-hand side.
    maxiter : int
        Maximum number of iterations.
    rtol : float
        Relative tolerance.
    callback : callable, optional
        Called after each iteration with the current solution vector.
    
    Returns
    -------
    x : ndarray
        Solution vector.
    info : int
        Solver information (0 = converged, >0 = iterations, <0 = failure)
    """
    sig = inspect.signature(minres)
    kwargs = dict(maxiter=maxiter, callback=callback)
    
    # SciPy changed minres signature in different versions
    if "tol" in sig.parameters:
        # print("Using old minres tol, check this part of the code!")
        kwargs["tol"] = rtol
    else:
        kwargs["rtol"] = rtol
        # kwargs["atol"] = 0.0

    x, info = minres(Aop, b, **kwargs)
    if warnings and info < 0:
        print("W: MINRES failed with info =", info)
    return x, info


  

def mf_leapssn(oracle, x_0, n_iters=1000, H_0=1.0,
                  adaptive_search=True, trace=True, eps=1e-8,
                  H_min=1e-5, f_star=None, grad_tol=None, warnings=False, 
                  lazy_Hessians=False, update_freq=5, lss_maxiters=400, 
                  lss_tol=1e-3, lss_method="cg", use_gauss_newton=False): 
    """
    Run our  algorithm
    for 'n_iters' iterations, minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.

    Only finite steps passing the acceptance tests update the iterate. Failed
    inner solves retry with larger regularization. An exhausted search or a
    step too small to change x returns the last accepted point with a failure
    status; it is not reported as convergence. Set grad_tol explicitly to stop
    at approximate stationarity (which need not be a local minimum).
    """

    
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    dual_norm_sqr = lambda x: x.dot(x)
    
    # Initialization of the method
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    H_k = H_0

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""
    lambda_k = H_0 # just to make it print.

    # Main loop
    no_backtracks = 0
    for k in range(n_iters + 1):
        if k % 500 == 0:
            # pass
            print("Iteration ", k)#, " with f_k = ", f_k, " and ||g_k|| = ", g_k_norm, "lambda_k = ", lambda_k)
        if trace:
            history['func'].append(f_k)
            history['grad_norm'].append(g_k_norm)
            history['H_k'].append(H_k)
            history['grad_calls'].append(oracle.grad_calls)
            history['matrix_inverses'].append(matrix_inverses)
            history['hess_calls'].append(oracle.hess_calls)
            history['penalty'].append(oracle.penalty(x_k))
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())

        if (f_star is not None and f_k - f_star < eps) or \
                (g_k_norm == 0 or (grad_tol is not None and g_k_norm < grad_tol)):
            status = "success, %d iters" % k
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break
        if lazy_Hessians is True and k % update_freq != 0:
            print("Didn't update Hessian at iteration " + str(k))
        else:
            if use_gauss_newton:
                Hess_k = oracle.hess_gn(x_k)
            else:
                Hess_k = oracle.hess(x_k)
        f_current = oracle.func(x_k)
        H_mv = Hess_k if callable(Hess_k) else (lambda v: Hess_k @ v)

        
        if history is not None:
            history["func_at_hess_eval"].append(oracle.func(x_k))
        adaptive_search_max_iter = 120
        step_accepted = False
        failure_status = "line_search_failed"
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(('W: adaptive_iterations_exceeded, k = %d' % k),
                          flush=True)
                break

            lambda_k = H_k 
            if not np.isfinite(lambda_k) or lambda_k <= 0:
                failure_status = "invalid_regularization"
                break
            def A_mv(v):
                return H_mv(v) + lambda_k * v
            Aop = LinearOperator((x_k.size, x_k.size), matvec=A_mv, dtype=np.float64)

            it_counter = {"it": 0}
            def _cb(_xk): it_counter["it"] += 1


            try:
                if lss_method == "cg":
                    delta_x, info = cg_compat(Aop, -g_k, maxiter=lss_maxiters, rtol=lss_tol, callback=_cb, warnings=warnings)
                else:
                    delta_x, info = minres_compat(Aop, -g_k, maxiter=lss_maxiters, rtol=lss_tol, callback=_cb, warnings=warnings)
            except Exception as e:
                if warnings:
                    print("W: General exception:", e)
                H_k *= 2.0
                continue
            if lss_method == "cg" and info != 0:
                if warnings:
                    print("W: CG linear solver did not converge, info =", info)
                H_k *= 2.0
                continue
            if lss_method == "minres" and info < 0:
                if warnings:
                    print("W: MINRES linear solver did not converge, info =", info)
                H_k *= 2.0
                continue
            if not np.all(np.isfinite(delta_x)):
                if warnings:
                    print("W: not all finite delta_x at k =", k, "i =", i)
                H_k *= 2.0
                continue

            # A finite direction can still overflow or be lost to rounding.
            with np.errstate(over="ignore", invalid="ignore"):
                x_trial = x_k + delta_x
            if not np.all(np.isfinite(x_trial)):
                H_k *= 2.0
                continue
            if np.array_equal(x_trial, x_k):
                failure_status = "numerical_stagnation"
                break

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                f_new = oracle.func(x_trial)
                g_new = oracle.grad(x_trial)
                g_new_norm_sqr = dual_norm_sqr(g_new)
            if not (np.isfinite(f_new) and np.all(np.isfinite(g_new))
                    and np.isfinite(g_new_norm_sqr)):
                H_k *= 2.0
                continue
            if history is not None:
                history["func_at_lssolve"].append(f_new)
                history["grad_norm_at_lssolve"].append(g_new_norm_sqr ** 0.5)

            if not adaptive_search:
                step_accepted = True
                break

            # Check condition for H_k
            if f_current - f_new >= 0.25*lambda_k*np.linalg.norm(delta_x)**2 and g_new.dot(-delta_x) >= 0.5*g_new_norm_sqr / (lambda_k):
                step_accepted = True
                H_k = H_k * 0.5             
                break
            H_k = H_k * 2
            no_backtracks += 1

        if not step_accepted:
            status = "%s, %d iters" % (failure_status, k)
            if warnings:
                print("W:", status, "-- returning the last accepted iterate", flush=True)
            break
        if history is not None:
            history["reg_parameter"].append(lambda_k)

        # Update the point
        x_k = x_trial
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr ** 0.5
    print("No of backtracks: ", no_backtracks)
    return x_k, status, history

def mf_gladssn(oracle, x_0, n_iters=1000, H_0=1.0,
                  adaptive_search=True, trace=True, eps=1e-8,
                  H_min=1e-5, p=0.99, f_star=None, grad_tol=None, warnings=False, 
                  lazy_Hessians=False, update_freq=5, lss_maxiters=400, lss_tol=1e-3,
                  lss_method="cg", use_gauss_newton=False): 
    """
    Run our new gradient-regularised algorithm
    for 'n_iters' iterations, minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.

    Only finite steps passing the acceptance tests update the iterate. Failed
    inner solves retry with larger regularization. An exhausted search or a
    step too small to change x returns the last accepted point with a failure
    status; it is not reported as convergence. Set grad_tol explicitly to stop
    at approximate stationarity (which need not be a local minimum).
    """
    
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    dual_norm_sqr = lambda x: x.dot(x)
    
    # Initialization of the method
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    stationarity_tol = grad_tol if grad_tol is not None else 0.0
    H_k = H_0 / (g_k_norm**p) if g_k_norm > stationarity_tol else H_0

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""
    lambda_k = H_0 # just to make it print.

    # Main loop
    no_backtracks = 0
    for k in range(n_iters + 1):
        if k % 500 == 0:
            # pass
            print("Iteration ", k)#, " with f_k = ", f_k, " and ||g_k|| = ", g_k_norm, "lambda_k = ", lambda_k)
        if trace:
            history['func'].append(f_k)
            history['grad_norm'].append(g_k_norm)
            history['H_k'].append(H_k)
            history['grad_calls'].append(oracle.grad_calls)
            history['matrix_inverses'].append(matrix_inverses)
            history['hess_calls'].append(oracle.hess_calls)
            history['penalty'].append(oracle.penalty(x_k))
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())

        if (f_star is not None and f_k - f_star < eps) or \
            g_k_norm <= stationarity_tol:
            status = "success, %d iters" % k
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break
        if lazy_Hessians is True and k % update_freq != 0:
            print("Didn't update Hessian at iteration " + str(k))
        else:
            if use_gauss_newton:
                Hess_k = oracle.hess_gn(x_k)
            else:
                Hess_k = oracle.hess(x_k)
        f_current = oracle.func(x_k)
        H_mv = Hess_k if callable(Hess_k) else (lambda v: Hess_k @ v)

        
        if history is not None:
            history["func_at_hess_eval"].append(oracle.func(x_k))
        adaptive_search_max_iter = 120
        step_accepted = False
        failure_status = "line_search_failed"
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(('W: adaptive_iterations_exceeded, k = %d' % k),
                          flush=True)
                break
            #if k == 0:
            #    lambda_k = H_k
            #else: # normal
            #    lambda_k = H_k* g_k_norm ** 0.5
            lambda_k = H_k* g_k_norm ** p
            if not np.isfinite(lambda_k) or lambda_k <= 0:
                failure_status = "invalid_regularization"
                break
            def A_mv(v):
                return H_mv(v) + lambda_k * v
            Aop = LinearOperator((x_k.size, x_k.size), matvec=A_mv, dtype=np.float64)

            it_counter = {"it": 0}
            def _cb(_xk): it_counter["it"] += 1


            try:
                if lss_method == "cg":
                    delta_x, info = cg_compat(Aop, -g_k, maxiter=lss_maxiters, rtol=lss_tol, callback=_cb, warnings=warnings)
                else:
                    delta_x, info = minres_compat(Aop, -g_k, maxiter=lss_maxiters, rtol=lss_tol, callback=_cb, warnings=warnings)
            except ZeroDivisionError:
                try:
                    if callable(Hess_k):
                        raise np.linalg.LinAlgError(
                            "direct fallback requires an explicit Hessian"
                        )
                    regularized_hessian = Hess_k + lambda_k * np.eye(x_k.size)
                    delta_x = scipy.linalg.solve(regularized_hessian, -g_k)
                    info = 0
                    if warnings:
                        print(
                            "W: MINRES breakdown at iteration",
                            k,
                            "; used direct regularized solve",
                            flush=True,
                        )
                except (np.linalg.LinAlgError, ValueError) as fallback_error:
                    if warnings:
                        print(
                            "W: MINRES breakdown and direct fallback failed:",
                            fallback_error,
                            flush=True,
                        )
                    H_k *= 4
                    continue
            except Exception as e:
                if warnings:
                    print("W: General exception:", e)
                H_k *= 4.0
                continue
            if lss_method == "cg" and info != 0:
                if warnings:
                    print("W: CG linear solver did not converge, info =", info)
                H_k *= 4.0
                continue
            if lss_method == "minres" and info < 0:
                if warnings:
                    print("W: MINRES linear solver did not converge, info =", info)
                H_k *= 4.0
                continue
            if not np.all(np.isfinite(delta_x)):
                if warnings:
                    print("W: not all finite delta_x at k =", k, "i =", i)
                H_k *= 4.0
                continue

            # A finite direction can still overflow or be lost to rounding.
            with np.errstate(over="ignore", invalid="ignore"):
                x_trial = x_k + delta_x
            if not np.all(np.isfinite(x_trial)):
                H_k *= 4.0
                continue
            if np.array_equal(x_trial, x_k):
                failure_status = "numerical_stagnation"
                break

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                f_new = oracle.func(x_trial)
                g_new = oracle.grad(x_trial)
                g_new_norm_sqr = dual_norm_sqr(g_new)
            if not (np.isfinite(f_new) and np.all(np.isfinite(g_new))
                    and np.isfinite(g_new_norm_sqr)):
                H_k *= 4.0
                continue
            if history is not None:
                history["func_at_lssolve"].append(f_new)
                history["grad_norm_at_lssolve"].append(g_new_norm_sqr ** 0.5)

            if not adaptive_search:
                step_accepted = True
                break

            # Check condition for H_k
            if f_current - f_new >= 0.25*lambda_k*np.linalg.norm(delta_x)**2 and g_new.dot(-delta_x) >= 0.5*g_new_norm_sqr / (lambda_k):
                step_accepted = True
                H_k = H_k * 0.25                
                break
            H_k = H_k * 4
            no_backtracks += 1

        if not step_accepted:
            status = "%s, %d iters" % (failure_status, k)
            if warnings:
                print("W:", status, "-- returning the last accepted iterate", flush=True)
            break
        if history is not None:
            history["reg_parameter"].append(lambda_k)

        # Update the point
        x_k = x_trial
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr ** 0.5
    print("No of backtracks: ", no_backtracks)
    return x_k, status, history


 
def super_newton(oracle, x_0, n_iters=1000, H_0=1.0, alpha=1.0,
                 adaptive_search=True, trace=True, B=None, Binv=None, eps=1e-8,
                 H_min=1e-5, f_star=None, grad_tol=None, warnings=False, lazy_hessians=False, lazy_hessians_freq=5):
    """
    Run Super-Universal Newton Method
    for 'n_iters' iterations, minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.
    """
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
            history['new'].append(oracle.func_at_hess_eval)
            history['hess_calls'].append(oracle.hess_calls)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())

        if (f_star is not None and f_k - f_star < eps) or \
                (grad_tol is not None and g_k_norm < grad_tol):
            status = "success, %d iters" % k
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        if lazy_hessians is True and k % lazy_hessians_freq != 0:
            print("skipped!")
        else:
            Hess_k = oracle.hess(x_k)

        # Hess_k = oracle.hess(x_k)
        history["func_at_hess_eval"].append(oracle.func(x_k))
        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(('W: adaptive_iterations_exceeded, k = %d' % k),
                          flush=True)
                break

            lambda_k = H_k * g_k_norm ** alpha
            print('lambda_k', lambda_k)
            try:
                # Compute the regularized Newton step
                delta_x = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
                    Hess_k + lambda_k * B, lower=False), -g_k)
                matrix_inverses += 1

            except (np.linalg.LinAlgError, ValueError) as e:
                if warnings:
                    print('W: linalg_error', flush=True)
            f_new = oracle.func(x_k + delta_x)
            history["func_at_lssolve"].append(f_new)
            

            g_new = oracle.grad(x_k + delta_x)
            g_new_norm_sqr = dual_norm_sqr(g_new)

            if not adaptive_search:
                break

            # Check condition for H_k
            if g_new.dot(-delta_x) >= g_new_norm_sqr / (4 * lambda_k):
                H_k *= 0.25
                # H_k = max(H_k, H_min)
                break

            H_k *= 4
            no_backtracks += 1

        # Update the point
        x_k += delta_x
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr ** 0.5
        history["reg_parameter"].append(lambda_k)
    print("No of backtracks: ", no_backtracks)
    return x_k, status, history


def cubic_newton_step(g, H_mv, H, B=None, eps=1e-8,
                      lss_maxiters=200, lss_tol=1e-6, lss_method="cg"):
    """
    Cubic Newton step using Hessian-vector products.

    Solves: min_x <g, x> + 1/2 <x, H_mv(x)> + (H/3) ||x||^3
    via the scalar root r: r - ||(H + HrB)^{-1} g|| = 0
    """

    n = g.shape[0]
    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
    else:
        l2_norm_sqr = lambda x: (x @ B).dot(x)

    def f(T, T_norm):
        Hv = H_mv(T)
        quad = 0.5 * T.dot(Hv)
        cubic = H * T_norm**3 / 3.0
        return g.dot(T) + quad + cubic

    def h(r, der=False):
        # Define linear operator for (H + H*r*B)
        def A_mv(v):
            Hv = H_mv(v)
            if B is None:
                return Hv + H * r * v
            else:
                return Hv + H * r * (B @ v)

        Aop = LinearOperator((n, n), matvec=A_mv, dtype=np.float64)

        # Solve (H + HrB) T = -g
        if lss_method == "cg":
            T, info = cg_compat(Aop, -g, maxiter=lss_maxiters, rtol=lss_tol)
        else:
            T, info = minres_compat(Aop, -g, maxiter=lss_maxiters, rtol=lss_tol)
        if info != 0:
            raise ValueError("Linear solve failed")

        T_norm = l2_norm_sqr(T)**0.5
        h_r = r - T_norm

        h_r_prime = None
        if der:
            # Solve (H + HrB) u = BT
            if B is None:
                BT = T
            else:
                BT = B @ T
            if lss_method == "cg":
                u, info = cg_compat(Aop, BT, maxiter=lss_maxiters, rtol=lss_tol)
            else:
                u, info = minres_compat(Aop, BT, maxiter=lss_maxiters, rtol=lss_tol)
            if info != 0:
                raise ValueError("Derivative linear solve failed")
            h_r_prime = 1 + H / T_norm * (u @ BT)

        return h_r, T_norm, T, h_r_prime

    try:
        max_r = 1.0
        max_iters = 50
        # Find max_r such that h(max_r) is nonnegative
        for i in range(max_iters):
            h_r, T_norm, T, _ = h(max_r)
            if h_r < -eps:
                max_r *= 2
            elif -eps <= h_r <= eps:
                return T, f(T, T_norm), "success"
            else:
                break

        # Univariate Newton's method over r
        r = max_r
        for i in range(max_iters):
            h_r, T_norm, T, h_r_prime = h(r, der=True)
            if -eps <= h_r <= eps:
                return T, f(T, T_norm), "success"
            r -= h_r / h_r_prime

    except (np.linalg.LinAlgError, ValueError) as e:
        return np.zeros(n), 0.0, "linalg_error"

    return np.zeros(n), 0.0, "iterations_exceeded"


def cubic_newton(oracle, x_0, n_iters=1000, H_0=1.0, adaptive_search=True,
                 trace=True, B=None, eps=1e-8, H_min=1e-5, f_star=None,
                 warnings=True, lss_maxiters=200, lss_tol=1e-9, lss_method="minres"):
    """
    Cubic Newton method using Hessian-vector products (matrix-free).

    Parameters
    ----------
    oracle : BaseSmoothOracle
        Oracle providing func, grad, hess (or hess_gn)
    x_0 : ndarray
        Initial point
    H_0 : float
        Cubic regularization parameter
    adaptive_search : bool
        Whether to adaptively adjust H_k
    trace : bool
        Whether to record history
    B : ndarray or None
        Optional B in cubic term
    eps : float
        Tolerance
    H_min : float
        Minimum H_k allowed
    f_star : float or None
        Optional optimal value for stopping
    warnings : bool
        Whether to print warnings
    lss_maxiters, lss_tol, lss_method : solver params
        Passed to Krylov linear solver
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    dual_norm_sqr = lambda x: x.dot(x)

    # Initialization
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k)**0.5
    H_k = H_0

    history = defaultdict(list) if trace else None
    status = ""

    for k in range(n_iters + 1):

        if trace:
            history['func'].append(f_k)
            history['grad_norm'].append(g_k_norm)
            history['grad_calls'].append(oracle.grad_calls)
            history['H_k'].append(H_k)
            history['hess_calls'].append(oracle.hess_calls)
            history['penalty'].append(oracle.penalty(x_k))
            history['time'].append((datetime.now() - start_timestamp).total_seconds())

        if f_star is not None and f_k - f_star < eps:
            status = f"success, {k} iters"
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        g_k = oracle.grad(x_k)
        g_k_norm = dual_norm_sqr(g_k)**0.5

        # Lazy Hessians / HVP
        Hess_k = oracle.hess(x_k)
        H_mv = Hess_k if callable(Hess_k) else (lambda v: Hess_k @ v)

        f_current = oracle.func(x_k)

        adaptive_search_max_iter = 120
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(f'W: adaptive_iterations_exceeded, k = {k}', flush=True)
                break

            try:
                # Cubic Newton step
                x_delta, model_value, message = cubic_newton_step(
                    g_k, H_mv, 0.5 * H_k, B=B,
                    eps=eps, lss_maxiters=lss_maxiters,
                    lss_tol=lss_tol, lss_method=lss_method
                )
            except Exception as e:
                if warnings:
                    print(f"W: cubic_newton_step failed at iter {k}, {e}")
                x_delta = np.zeros_like(x_k)
                model_value = 0.0

            f_new = oracle.func(x_k + x_delta)
            history["func_at_lssolve"].append(f_new)

            g_new = oracle.grad(x_k + x_delta)
            g_new_norm_sqr = dual_norm_sqr(g_new)
            history["grad_norm_at_lssolve"].append(g_new_norm_sqr ** 0.5)
            if not adaptive_search:
                break

            # Adaptive search condition
            if f_new <= f_current + model_value:
                H_k = max(H_k * 0.5, H_min)
                break

            H_k *= 2  # Increase regularization if step too aggressive

        # Update iterate
        x_k += x_delta
        f_k = f_new

    return x_k, status, history


def gradient_method(oracle, x_0, max_iter=1000, L_0=1.0, line_search=True,
                    trace=True, B=None, Binv=None, L_min=1e-5, warnings=False):
    """
    Run the Gradient Method for 'n_iters' iterations, 
    minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Initialization of the Euclidean metric
    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
        dual_norm_sqr = lambda x: x.dot(x)
        to_dual = lambda x: x
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)
        to_dual = lambda x: B.dot(x)
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

        if k == max_iter:
            status = 'iterations_exceeded'
            break

        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                if warnings:
                    print('W: line_search_max_iter_reached', flush=True)
                break

            T = x_k - precond(grad_k) / L_k
            func_T = oracle.func(T)

            if not line_search:
                break

            if func_T <= func_k + grad_k.dot(T - x_k) + \
                    0.5 * L_k * l2_norm_sqr(T - x_k):
                L_k *= 0.5
                L_k = max(L_k, L_min)
                break

            L_k *= 2

        x_k = T
        func_k = func_T
        grad_k = oracle.grad(T)
        grad_k_norm_sqr = dual_norm_sqr(grad_k)

    return x_k, status, history


def fast_gradient_method(oracle, x_0, max_iter=1000, L_0=1.0, line_search=True,
                         trace=True, B=None, Binv=None, L_min=1e-5,
                         warnings=False):
    """
    Run the Fast Gradient Method for 'n_iters' iterations, 
    minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Initialization of the Euclidean metric
    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
        dual_norm_sqr = lambda x: x.dot(x)
        to_dual = lambda x: x
        precond = lambda g: g
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)
        to_dual = lambda x: B.dot(x)
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

        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                if warnings:
                    print('W: line_search_max_iter_reached', flush=True)
                break

            # Choose a_k_new from the equation:
            #   L_k * a_k_new ** 2 = A_k + a_k_new
            a_k_new = (1 + (1 + 4 * A_k * L_k) ** 0.5) / (2 * L_k)
            y_k = (a_k_new * v_k + A_k * x_k) / (a_k_new + A_k)
            grad_y_k = oracle.grad(y_k)

            T = y_k - precond(grad_y_k) / L_k
            func_T = oracle.func(T)

            if not line_search:
                break

            func_y_k = oracle.func(y_k)
            if func_T <= func_y_k + grad_y_k.dot(T - y_k) + \
                    0.5 * L_k * l2_norm_sqr(T - y_k):
                L_k *= 0.5
                L_k = max(L_k, L_min)
                break

            L_k = 2 * L_k

        grad_T = oracle.grad(T)

        v_k = T + A_k / a_k_new * (T - x_k)
        A_k += a_k_new

        x_k = T
        func_k = func_T
        grad_k_norm_sqr = dual_norm_sqr(grad_k)

    return x_k, status, history


def _make_history(trace):
    return defaultdict(list) if trace else None



def _append_history(history, oracle, z, func_k, grad_norm=None, proj_grad_norm=None,
                    L_u=None, L_v=None, start_timestamp=None):
    if history is None:
        return
    if grad_norm is None:
        grad_norm = np.linalg.norm(oracle.grad(z))

    history['func'].append(float(func_k))
    history['grad_norm'].append(float(grad_norm))
    history['grad_sqr_norm'].append(float(grad_norm ** 2))
    history['penalty'].append(float(oracle.penalty(z)))

    if proj_grad_norm is not None:
        history['proj_grad_norm'].append(float(proj_grad_norm))
    if L_u is not None:
        history['L_u'].append(float(L_u))
    if L_v is not None:
        history['L_v'].append(float(L_v))
    if start_timestamp is not None:
        history['time'].append((datetime.now() - start_timestamp).total_seconds())


def alternating_gradient_method(
    oracle,
    z_0,
    max_iter=200,
    L_u0=None,
    L_v0=None,
    line_search=True,
    L_min=1e-12,
    trace=True,
    tol=None,
    warnings=False,
):
    """
    Block gradient descent on the same smooth penalized objective as oracle.func.
    One U-step with V fixed, then one V-step with U fixed.
    U has shape (m, r), V has shape (r, n), and the residual is U @ V - M.
    Ridge and penalty weights follow the revised equation (27): alpha/2, beta/2.

    This is usually the cleanest 'factorization-aware' baseline exact
    penalty formulation.
    """
    M, alpha, beta = oracle.M, oracle.alpha, oracle.beta
    U, V = oracle.unpack(np.asarray(z_0, dtype=np.float64).copy())

    z = oracle.pack(U, V)
    func_k = oracle.func(z)
    L_u = float(L_u0) if L_u0 is not None else (np.linalg.norm(V, 2) ** 2 + alpha + beta)
    L_v = float(L_v0) if L_v0 is not None else (np.linalg.norm(U, 2) ** 2 + alpha + beta)

    history = _make_history(trace)
    start_timestamp = datetime.now()
    status = ""

    for k in range(max_iter + 1):
        z = oracle.pack(U, V)
        g = oracle.grad(z)
        grad_norm = np.linalg.norm(g)

        _append_history(
            history, oracle, z, func_k,
            grad_norm=grad_norm, L_u=L_u, L_v=L_v,
            start_timestamp=start_timestamp
        )

        if tol is not None and grad_norm <= tol:
            status = "success"
            break
        if k == max_iter:
            status = "iterations_exceeded"
            break

        # ---- U update ----
        R = U @ V - M
        gU = R @ V.T + alpha * U + beta * np.minimum(U, 0.0)

        ls_ok = False
        for _ in range(30):
            U_trial = U - gU / L_u
            D = U_trial - U
            z_trial = oracle.pack(U_trial, V)
            f_trial = oracle.func(z_trial)

            if (not line_search) or (
                f_trial <= func_k + np.sum(gU * D) + 0.5 * L_u * np.sum(D * D)
            ):
                U = U_trial
                func_k = f_trial
                if line_search:
                    L_u = max(L_min, 0.5 * L_u)
                ls_ok = True
                break
            L_u *= 2.0

        if (not ls_ok) and warnings:
            print("W: U line-search maxed out", flush=True)

        # ---- V update ----
        R = U @ V - M
        gV = U.T @ R + alpha * V + beta * np.minimum(V, 0.0)

        ls_ok = False
        for _ in range(30):
            V_trial = V - gV / L_v
            D = V_trial - V
            z_trial = oracle.pack(U, V_trial)
            f_trial = oracle.func(z_trial)

            if (not line_search) or (
                f_trial <= func_k + np.sum(gV * D) + 0.5 * L_v * np.sum(D * D)
            ):
                V = V_trial
                func_k = f_trial
                if line_search:
                    L_v = max(L_min, 0.5 * L_v)
                ls_ok = True
                break
            L_v *= 2.0

        if (not ls_ok) and warnings:
            print("W: V line-search maxed out", flush=True)

    return oracle.pack(U, V), status, history